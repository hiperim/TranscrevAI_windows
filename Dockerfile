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
# Build-time argument for HuggingFace token (not exposed in runtime)
ARG HUGGING_FACE_HUB_TOKEN

# Set HuggingFace cache location explicitly
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers
ENV HF_DATASETS_CACHE=/root/.cache/huggingface/datasets

# Create cache directories explicitly
RUN mkdir -p /root/.cache/huggingface/hub

# Download Whisper model (public, no token needed)
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='Systran/faster-whisper-medium', \
    cache_dir='/root/.cache/huggingface'); \
    print('✓ Faster-Whisper model pre-downloaded to cache')" && \
    ls -la /root/.cache/huggingface/hub/ && \
    echo "Whisper cache size:" && du -sh /root/.cache/huggingface/

# Copy model download script
COPY download_models.py /tmp/download_models.py

# Download Pyannote models (requires token at build time only)
RUN if [ -n "$HUGGING_FACE_HUB_TOKEN" ]; then \
    echo "Cache bust: 2025-10-29-v5" && \
    echo "Starting fresh Pyannote model download with HF_HOME cache..." && \
    python3 /tmp/download_models.py && \
    echo "" && echo "Total cache size after Pyannote:" && du -sh /root/.cache/huggingface/ && \
    rm /tmp/download_models.py; \
else \
    echo "⚠️  HUGGING_FACE_HUB_TOKEN not provided - Pyannote models will be downloaded at runtime" && \
    rm /tmp/download_models.py; \
fi

# FASE 11: Models are loaded at runtime using hf_hub_download() from cache
# No config rewriting needed - all path resolution happens in memory at runtime
# The Hub will automatically use local cache if models are present.
# Models are already downloaded during build, so they will be loaded from cache automatically.

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