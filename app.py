"""
Swaram Chord Detection Service
FastAPI microservice that detects chords from audio files using librosa
chroma analysis with beat-synchronized chord template matching.

Usage:
    uvicorn app:app --host 0.0.0.0 --port 7860

API:
    POST /analyze-upload  (multipart file upload)
    GET  /health
"""

import os
import re
import tempfile
import logging
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ===== Configuration =====
MAX_DURATION_SEC = 600  # 10 minutes max
MAX_UPLOAD_MB = 30
SAMPLE_RATE = 22050
HOP_LENGTH = 512

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chord-service")

app = FastAPI(title="Swaram Chord Detection", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


# ===== Response models =====
class ChordEntry(BaseModel):
    time: float
    duration: float
    chord: str


class AnalyzeResponse(BaseModel):
    videoId: str
    key: str
    bpm: int
    timeSignature: str
    chords: list[ChordEntry]


# ===== Chord Templates =====
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

CHORD_QUALITIES = {
    "":     [0, 4, 7],          # major
    "m":    [0, 3, 7],          # minor
    "7":    [0, 4, 7, 10],      # dominant 7th
    "m7":   [0, 3, 7, 10],      # minor 7th
    "M7":   [0, 4, 7, 11],      # major 7th
    "dim":  [0, 3, 6],          # diminished
    "aug":  [0, 4, 8],          # augmented
    "sus4": [0, 5, 7],          # suspended 4th
    "sus2": [0, 2, 7],          # suspended 2nd
}


def build_chord_templates() -> dict[str, np.ndarray]:
    """Build normalized 12-dimensional chroma template for each chord."""
    templates = {}
    for root_idx, root_name in enumerate(NOTE_NAMES):
        for quality_name, intervals in CHORD_QUALITIES.items():
            chord_name = f"{root_name}{quality_name}"
            template = np.zeros(12)
            for interval in intervals:
                template[(root_idx + interval) % 12] = 1.0
            template[root_idx] = 1.5  # Weight root note higher
            template = template / np.linalg.norm(template)
            templates[chord_name] = template
    return templates


CHORD_TEMPLATES = build_chord_templates()


# ===== Audio Analysis Functions =====
def analyze_audio(y: np.ndarray, sr: int) -> dict:
    """Run full chord analysis on loaded audio. Returns dict with all results."""
    if len(y) < sr * 2:
        raise ValueError("Audio too short (< 2 seconds)")

    # Detect BPM and beat positions
    logger.info("Detecting BPM and beats...")
    bpm, beat_frames = detect_bpm(y, sr)

    # Detect chords (beat-synchronized)
    logger.info("Detecting chords...")
    chords = detect_chords(y, sr, beat_frames)
    if not chords:
        raise ValueError("No chords detected in this audio")

    # Detect key and time signature
    key = detect_key(y, sr)
    time_sig = estimate_time_signature(y, sr, beat_frames)

    logger.info("Done: %d chords, key=%s, bpm=%d, time=%s",
                len(chords), key, bpm, time_sig)

    return {
        "key": key,
        "bpm": bpm,
        "timeSignature": time_sig,
        "chords": chords,
    }


def detect_chords(y: np.ndarray, sr: int, beat_frames: np.ndarray) -> list[dict]:
    """Detect chords using beat-synchronized chroma analysis."""
    y_harmonic = librosa.effects.harmonic(y, margin=4.0)

    chroma = librosa.feature.chroma_cqt(
        y=y_harmonic, sr=sr, hop_length=HOP_LENGTH, n_chroma=12
    )

    # Sync chroma to beats
    if len(beat_frames) < 2:
        frame_length = int(sr * 0.5 / HOP_LENGTH)
        n_frames = chroma.shape[1] // frame_length
        if n_frames < 1:
            return []
        beat_chroma = np.array([
            chroma[:, i * frame_length:(i + 1) * frame_length].mean(axis=1)
            for i in range(n_frames)
        ])
        beat_times = np.arange(n_frames) * 0.5
    else:
        beat_chroma = librosa.util.sync(chroma, beat_frames, aggregate=np.median).T
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP_LENGTH)

    # Match each beat to best chord template
    template_names = list(CHORD_TEMPLATES.keys())
    template_matrix = np.array(list(CHORD_TEMPLATES.values()))

    chord_indices = []
    confidences = []
    for frame_chroma in beat_chroma:
        if np.linalg.norm(frame_chroma) < 1e-6:
            chord_indices.append(-1)
            confidences.append(0.0)
            continue
        frame_norm = frame_chroma / np.linalg.norm(frame_chroma)
        similarities = template_matrix @ frame_norm
        best_idx = np.argmax(similarities)
        chord_indices.append(best_idx)
        confidences.append(similarities[best_idx])

    chord_indices = np.array(chord_indices)
    confidences = np.array(confidences)

    # Median smoothing
    if len(chord_indices) >= 3:
        from scipy.ndimage import median_filter
        chord_indices = median_filter(chord_indices, size=3).astype(int)

    # Build chord list
    total_duration = len(y) / sr
    chords = []
    for i, (idx, conf) in enumerate(zip(chord_indices, confidences)):
        if idx < 0 or conf < 0.3:
            continue

        time_start = float(beat_times[i]) if i < len(beat_times) else 0.0
        if i + 1 < len(beat_times):
            duration = float(beat_times[i + 1] - beat_times[i])
        else:
            duration = float(total_duration - time_start)

        chords.append({
            "time": round(time_start, 2),
            "duration": round(max(duration, 0.1), 2),
            "chord": template_names[idx],
        })

    # Merge consecutive identical chords
    merged = []
    for chord in chords:
        if merged and merged[-1]["chord"] == chord["chord"]:
            merged[-1]["duration"] = round(
                merged[-1]["duration"] + chord["duration"], 2
            )
        else:
            merged.append(dict(chord))

    return merged


def detect_key(y: np.ndarray, sr: int) -> str:
    """Detect musical key using Krumhansl-Kessler key profiles."""
    y_harmonic = librosa.effects.harmonic(y, margin=4.0)
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
    chroma_mean = chroma.mean(axis=1)

    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                              2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                              2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    major_corr = np.array([
        np.corrcoef(chroma_mean, np.roll(major_profile, i))[0, 1]
        for i in range(12)
    ])
    minor_corr = np.array([
        np.corrcoef(chroma_mean, np.roll(minor_profile, i))[0, 1]
        for i in range(12)
    ])

    best_major_idx = np.argmax(major_corr)
    best_minor_idx = np.argmax(minor_corr)

    if major_corr[best_major_idx] >= minor_corr[best_minor_idx]:
        return NOTE_NAMES[best_major_idx]
    else:
        return NOTE_NAMES[best_minor_idx] + "m"


def detect_bpm(y: np.ndarray, sr: int) -> tuple[int, np.ndarray]:
    """Detect BPM and beat frames."""
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
    bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    return round(bpm), beat_frames


def estimate_time_signature(y: np.ndarray, sr: int, beat_frames: np.ndarray) -> str:
    """Estimate time signature (3/4 vs 4/4) from beat strength patterns."""
    if len(beat_frames) < 6:
        return "4/4"

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    beat_strengths = onset_env[beat_frames[beat_frames < len(onset_env)]]

    if len(beat_strengths) < 9:
        return "4/4"

    def grouping_score(group_size):
        n = len(beat_strengths) - (len(beat_strengths) % group_size)
        if n == 0:
            return 0
        groups = beat_strengths[:n].reshape(-1, group_size)
        return np.mean(groups[:, 0]) / (np.mean(groups[:, 1:]) + 1e-8)

    score_3 = grouping_score(3)
    score_4 = grouping_score(4)

    return "3/4" if score_3 > score_4 * 1.15 else "4/4"


# ===== API Endpoints =====
@app.get("/")
def root():
    return {"name": "Swaram Chord Detection", "version": "3.0.0", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze-upload")
async def analyze_upload(
    file: UploadFile = File(...),
    video_id: str = Form(default=""),
):
    """Analyze an uploaded audio file for chords."""
    # Validate file size
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(413, f"File too large ({size_mb:.1f}MB). Max {MAX_UPLOAD_MB}MB.")

    # Validate file type
    allowed_types = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma", ".webm"}
    ext = Path(file.filename).suffix.lower() if file.filename else ""
    if ext not in allowed_types:
        raise HTTPException(
            415,
            f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(allowed_types))}"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save uploaded file
        audio_path = os.path.join(tmpdir, f"upload{ext}")
        with open(audio_path, "wb") as f:
            f.write(contents)

        # Load audio
        logger.info("Loading uploaded audio: %s (%.1fMB)", file.filename, size_mb)
        try:
            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True,
                                 duration=MAX_DURATION_SEC)
        except Exception as e:
            logger.error("Audio load failed: %s", e)
            raise HTTPException(422, f"Could not read audio file: {e}")

        # Analyze
        try:
            result = analyze_audio(y, sr)
        except ValueError as e:
            raise HTTPException(422, str(e))
        except Exception as e:
            logger.error("Analysis failed: %s", e)
            raise HTTPException(500, f"Chord detection failed: {e}")

        return AnalyzeResponse(
            videoId=video_id or "upload",
            key=result["key"],
            bpm=result["bpm"],
            timeSignature=result["timeSignature"],
            chords=[ChordEntry(**c) for c in result["chords"]],
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
