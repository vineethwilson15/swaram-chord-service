"""
Swaram Chord Detection Service
FastAPI microservice — detects chords from uploaded audio files using
librosa chroma analysis with beat-synchronized template matching.

Optimized for low-CPU environments (Render free tier = 0.1 vCPU).
"""

import os
import tempfile
import logging
from pathlib import Path

import numpy as np
import librosa
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ===== Configuration (tuned for speed on 0.1 vCPU) =====
MAX_DURATION_SEC = 240   # 4 minutes max (keeps processing under 60s)
MAX_UPLOAD_MB = 30
SAMPLE_RATE = 16000      # lower SR = faster loading + less data to process
HOP_LENGTH = 2048        # larger hops = fewer frames = faster

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chord-service")

app = FastAPI(title="Swaram Chord Detection", version="3.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
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
    "":     [0, 4, 7],
    "m":    [0, 3, 7],
    "7":    [0, 4, 7, 10],
    "m7":   [0, 3, 7, 10],
    "M7":   [0, 4, 7, 11],
    "dim":  [0, 3, 6],
    "aug":  [0, 4, 8],
    "sus4": [0, 5, 7],
    "sus2": [0, 2, 7],
}


def build_chord_templates() -> tuple[list[str], np.ndarray]:
    """Build chord templates. Returns (names, matrix) for fast vectorized matching."""
    names = []
    templates = []
    for root_idx, root_name in enumerate(NOTE_NAMES):
        for quality_name, intervals in CHORD_QUALITIES.items():
            chord_name = f"{root_name}{quality_name}"
            template = np.zeros(12)
            for interval in intervals:
                template[(root_idx + interval) % 12] = 1.0
            template[root_idx] = 1.5
            template = template / np.linalg.norm(template)
            names.append(chord_name)
            templates.append(template)
    return names, np.array(templates)


TEMPLATE_NAMES, TEMPLATE_MATRIX = build_chord_templates()


# ===== Audio Analysis (optimized for speed) =====
def analyze_audio(y: np.ndarray, sr: int) -> dict:
    """Full chord analysis pipeline. Target: < 60s on 0.1 vCPU for 4min audio."""
    if len(y) < sr * 2:
        raise ValueError("Audio too short (< 2 seconds)")

    logger.info("Audio loaded: %.1fs at %dHz", len(y) / sr, sr)

    # 1. Beat tracking (needed for chord sync + BPM + time sig)
    logger.info("Beat tracking...")
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
    bpm = round(float(tempo) if np.isscalar(tempo) else float(tempo[0]))
    logger.info("BPM: %d, beats: %d", bpm, len(beat_frames))

    # 2. Chroma features (using STFT — faster than CQT, skipping HPSS)
    logger.info("Computing chroma...")
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP_LENGTH, n_chroma=12)

    # 3. Sync chroma to beats
    if len(beat_frames) >= 2:
        beat_chroma = librosa.util.sync(chroma, beat_frames, aggregate=np.median).T
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP_LENGTH)
    else:
        frame_len = int(sr * 0.5 / HOP_LENGTH)
        n_frames = max(1, chroma.shape[1] // frame_len)
        beat_chroma = np.array([
            chroma[:, i * frame_len:(i + 1) * frame_len].mean(axis=1)
            for i in range(n_frames)
        ])
        beat_times = np.arange(n_frames) * 0.5

    # 4. Match to chord templates (vectorized)
    logger.info("Matching chords...")
    norms = np.linalg.norm(beat_chroma, axis=1, keepdims=True)
    norms[norms < 1e-6] = 1.0
    beat_chroma_norm = beat_chroma / norms
    similarities = beat_chroma_norm @ TEMPLATE_MATRIX.T  # (n_beats, n_chords)
    chord_indices = np.argmax(similarities, axis=1)
    confidences = np.max(similarities, axis=1)

    # Mark silent frames
    is_silent = np.linalg.norm(beat_chroma, axis=1) < 1e-6
    chord_indices[is_silent] = -1
    confidences[is_silent] = 0.0

    # 5. Median smoothing
    if len(chord_indices) >= 3:
        from scipy.ndimage import median_filter
        valid_mask = chord_indices >= 0
        if valid_mask.sum() >= 3:
            smoothed = median_filter(chord_indices, size=3)
            chord_indices[valid_mask] = smoothed[valid_mask]

    # 6. Build chord list
    total_dur = len(y) / sr
    chords = []
    for i in range(len(chord_indices)):
        idx = int(chord_indices[i])
        if idx < 0 or confidences[i] < 0.3:
            continue
        t_start = float(beat_times[i]) if i < len(beat_times) else 0.0
        dur = float(beat_times[i + 1] - beat_times[i]) if i + 1 < len(beat_times) else float(total_dur - t_start)
        chords.append({
            "time": round(t_start, 2),
            "duration": round(max(dur, 0.1), 2),
            "chord": TEMPLATE_NAMES[idx],
        })

    # 7. Merge consecutive identical chords
    merged = []
    for chord in chords:
        if merged and merged[-1]["chord"] == chord["chord"]:
            merged[-1]["duration"] = round(merged[-1]["duration"] + chord["duration"], 2)
        else:
            merged.append(dict(chord))

    if not merged:
        raise ValueError("No chords detected in this audio")

    # 8. Key detection (fast — just chroma mean + correlation)
    chroma_mean = chroma.mean(axis=1)
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    major_corr = np.array([np.corrcoef(chroma_mean, np.roll(major_profile, i))[0, 1] for i in range(12)])
    minor_corr = np.array([np.corrcoef(chroma_mean, np.roll(minor_profile, i))[0, 1] for i in range(12)])

    best_maj = np.argmax(major_corr)
    best_min = np.argmax(minor_corr)
    key = NOTE_NAMES[best_maj] if major_corr[best_maj] >= minor_corr[best_min] else NOTE_NAMES[best_min] + "m"

    # 9. Time signature estimation
    time_sig = "4/4"
    if len(beat_frames) >= 9:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
        bs = onset_env[beat_frames[beat_frames < len(onset_env)]]
        if len(bs) >= 9:
            def gscore(g):
                n = len(bs) - (len(bs) % g)
                if n == 0: return 0
                groups = bs[:n].reshape(-1, g)
                return np.mean(groups[:, 0]) / (np.mean(groups[:, 1:]) + 1e-8)
            if gscore(3) > gscore(4) * 1.15:
                time_sig = "3/4"

    logger.info("Done: %d chords, key=%s, bpm=%d, time=%s", len(merged), key, bpm, time_sig)

    return {"key": key, "bpm": bpm, "timeSignature": time_sig, "chords": merged}


# ===== API Endpoints =====
@app.get("/")
def root():
    return {"name": "Swaram Chord Detection", "version": "3.1.0", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze-upload")
async def analyze_upload(
    file: UploadFile = File(...),
    video_id: str = Form(default=""),
):
    """Analyze an uploaded audio file for chords."""
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(413, f"File too large ({size_mb:.1f}MB). Max {MAX_UPLOAD_MB}MB.")

    allowed = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma", ".webm"}
    ext = Path(file.filename).suffix.lower() if file.filename else ""
    if ext not in allowed:
        raise HTTPException(415, f"Unsupported type '{ext}'. Use: {', '.join(sorted(allowed))}")

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, f"upload{ext}")
        with open(audio_path, "wb") as f:
            f.write(contents)

        logger.info("Uploaded: %s (%.1fMB)", file.filename, size_mb)

        try:
            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True, duration=MAX_DURATION_SEC)
        except Exception as e:
            logger.error("Load failed: %s", e)
            raise HTTPException(422, f"Could not read audio file: {e}")

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
