"""
Swaram Chord Detection Service
FastAPI microservice that detects chords from YouTube videos using librosa
chroma analysis with beat-synchronized chord template matching.

Usage:
    uvicorn app:app --host 0.0.0.0 --port 7860

API:
    POST /analyze  { "url": "https://youtube.com/watch?v=..." }
    GET  /health
"""

import os
import re
import tempfile
import subprocess
import logging
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ===== Configuration =====
MAX_DURATION_SEC = 600  # 10 minutes max
SAMPLE_RATE = 22050
HOP_LENGTH = 512

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chord-service")

app = FastAPI(title="Swaram Chord Detection", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


# ===== Request / Response models =====
class AnalyzeRequest(BaseModel):
    url: str


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


# ===== YouTube helpers =====
VIDEO_ID_RE = re.compile(
    r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^\?&\s]+)"
)


def extract_video_id(url: str) -> str | None:
    m = VIDEO_ID_RE.search(url)
    return m.group(1) if m else None


def download_audio(url: str, output_dir: str) -> str:
    """Download YouTube audio using yt-dlp. Returns path to downloaded file."""
    output_template = os.path.join(output_dir, "audio.%(ext)s")
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "5",  # medium quality is fine for chord detection
        "--output", output_template,
        "--max-filesize", "50M",
        "--socket-timeout", "30",
        "--no-check-certificates",
        "--js-runtimes", "node",
        # Use mobile/android clients to avoid bot detection on cloud IPs
        "--extractor-args", "youtube:player_client=mweb,android",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        logger.error("yt-dlp stderr: %s", result.stderr)
        raise RuntimeError(f"yt-dlp failed: {result.stderr[:300]}")

    # Find the output file
    audio_files = list(Path(output_dir).glob("audio.*"))
    if not audio_files:
        raise RuntimeError("yt-dlp produced no output file")
    return str(audio_files[0])


# ===== Chord Templates =====
# 12 pitch classes: C, C#, D, D#, E, F, F#, G, G#, A, A#, B
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Chord quality intervals (semitones from root)
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
            # Weight the root note higher
            template[root_idx] = 1.5
            # Normalize
            template = template / np.linalg.norm(template)
            templates[chord_name] = template
    return templates


CHORD_TEMPLATES = build_chord_templates()


# ===== Chord Detection =====
def detect_chords(y: np.ndarray, sr: int, beat_frames: np.ndarray) -> list[dict]:
    """
    Detect chords using beat-synchronized chroma analysis.

    Process:
    1. Separate harmonic content (removes drums/percussion)
    2. Compute CQT chroma features
    3. Sync chroma to beats (one chord per beat)
    4. Match each beat's chroma to chord templates
    5. Apply median smoothing to avoid rapid changes
    6. Merge consecutive identical chords
    """
    # Harmonic-percussive separation — keeps only tonal content
    y_harmonic = librosa.effects.harmonic(y, margin=4.0)

    # Compute chroma using CQT (better frequency resolution for chords)
    chroma = librosa.feature.chroma_cqt(
        y=y_harmonic, sr=sr, hop_length=HOP_LENGTH, n_chroma=12
    )

    # Sync chroma to beats (average chroma per beat interval)
    if len(beat_frames) < 2:
        # Not enough beats — use fixed-size frames
        frame_length = int(sr * 0.5 / HOP_LENGTH)  # 0.5s frames
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
    template_matrix = np.array(list(CHORD_TEMPLATES.values()))  # (n_chords, 12)

    chord_indices = []
    confidences = []
    for frame_chroma in beat_chroma:
        if np.linalg.norm(frame_chroma) < 1e-6:
            chord_indices.append(-1)  # silence
            confidences.append(0.0)
            continue
        # Normalize frame
        frame_norm = frame_chroma / np.linalg.norm(frame_chroma)
        # Cosine similarity with all templates
        similarities = template_matrix @ frame_norm
        best_idx = np.argmax(similarities)
        chord_indices.append(best_idx)
        confidences.append(similarities[best_idx])

    chord_indices = np.array(chord_indices)
    confidences = np.array(confidences)

    # Median smoothing (window=3 beats) to reduce rapid chord flickering
    if len(chord_indices) >= 3:
        from scipy.ndimage import median_filter
        chord_indices = median_filter(chord_indices, size=3).astype(int)

    # Build chord list with timestamps
    total_duration = len(y) / sr
    chords = []
    for i, (idx, conf) in enumerate(zip(chord_indices, confidences)):
        if idx < 0 or conf < 0.3:
            continue  # skip silence / low confidence

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


# ===== Key detection =====
def detect_key(y: np.ndarray, sr: int) -> str:
    """Detect musical key using Krumhansl-Kessler key profiles."""
    y_harmonic = librosa.effects.harmonic(y, margin=4.0)
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
    chroma_mean = chroma.mean(axis=1)

    # Krumhansl-Kessler key profiles
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


# ===== BPM detection =====
def detect_bpm(y: np.ndarray, sr: int) -> tuple[int, np.ndarray]:
    """Detect BPM and beat frames."""
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
    bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    return round(bpm), beat_frames


# ===== Time signature estimation =====
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
    return {"name": "Swaram Chord Detection", "status": "ok", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    video_id = extract_video_id(req.url)
    if not video_id:
        raise HTTPException(400, "Invalid YouTube URL")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Download audio
        logger.info("Downloading audio for %s ...", video_id)
        try:
            audio_file = download_audio(req.url, tmpdir)
        except Exception as e:
            logger.error("Download failed: %s", e)
            raise HTTPException(502, f"Failed to download audio: {e}")

        # Load audio once (reuse for all analysis)
        logger.info("Loading audio: %s", audio_file)
        try:
            y, sr = librosa.load(audio_file, sr=SAMPLE_RATE, mono=True,
                                 duration=MAX_DURATION_SEC)
        except Exception as e:
            logger.error("Audio load failed: %s", e)
            raise HTTPException(500, f"Failed to load audio: {e}")

        if len(y) < sr * 2:
            raise HTTPException(422, "Audio too short (< 2 seconds)")

        # Detect BPM and beat positions (used by chord detection + time sig)
        logger.info("Detecting BPM and beats...")
        bpm, beat_frames = detect_bpm(y, sr)

        # Detect chords (beat-synchronized)
        logger.info("Detecting chords...")
        try:
            chords = detect_chords(y, sr, beat_frames)
        except Exception as e:
            logger.error("Chord detection failed: %s", e)
            raise HTTPException(500, f"Chord detection failed: {e}")

        if not chords:
            raise HTTPException(422, "No chords detected in this audio")

        # Detect key and time signature
        key = detect_key(y, sr)
        time_sig = estimate_time_signature(y, sr, beat_frames)

        logger.info(
            "Done: %d chords, key=%s, bpm=%d, time=%s",
            len(chords), key, bpm, time_sig,
        )

        return AnalyzeResponse(
            videoId=video_id,
            key=key,
            bpm=bpm,
            timeSignature=time_sig,
            chords=[ChordEntry(**c) for c in chords],
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
