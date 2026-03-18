FROM python:3.11-slim

# System dependencies for yt-dlp + librosa audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY app.py .

# Hugging Face Spaces requires port 7860 and non-root user
EXPOSE 7860
RUN useradd -m -u 1000 user
USER user

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
