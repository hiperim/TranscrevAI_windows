# TranscrevAI Docker Container - Production Ready (CPU-Only)
# Universal Portuguese Brazilian transcription and diarization
# Compatible with: Windows/Linux/macOS (including Silicon)
# Minimum specs: 4 cores, 8GB RAM

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    wget \
    git \
    build-essential \
    cmake \
    pkg-config \
    libsndfile1 \
    portaudio19-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# FASE 10: Pre-download models during build to eliminate cold start download time
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='Systran/faster-whisper-medium', \
    cache_dir='/root/.cache/huggingface'); \
    print('✓ Faster-Whisper model pre-downloaded to cache')"

# Copy application source code
COPY src/ ./src/
COPY main.py .

COPY config/ ./config/
COPY static/ ./static/
COPY templates/ ./templates/

# Create necessary directories
RUN mkdir -p data/uploads data/transcripts data/srt data/recordings temp

# Pre-download and convert models for immediate use


# Set proper permissions
RUN chmod +x main.py

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Start command
CMD ["gunicorn", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "-w", "1", \
     "--timeout", "300", \
     "--graceful-timeout", "300", \
     "-b", "0.0.0.0:8000", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "main:app"]