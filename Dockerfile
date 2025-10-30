# =====================================================================================
# Single-Stage Dockerfile - Optimized for 8GB RAM Systems
# - All dependencies and models in one stage
# - PyTorch CPU from requirements.txt via --index-url
# - ~17GB final image
# =====================================================================================
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/root/.cache/huggingface

# Install all system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
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

# Set working directory
WORKDIR /app

# Copy and install Python dependencies (includes PyTorch CPU at top)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download Whisper model (public, no token needed)
RUN python -c "from huggingface_hub import snapshot_download; \
    print('Downloading Whisper model...'); \
    snapshot_download(repo_id='Systran/faster-whisper-medium')"

# Download Pyannote models (requires token at build-time)
ARG HUGGING_FACE_HUB_TOKEN
COPY download_models.py .
RUN if [ -n "$HUGGING_FACE_HUB_TOKEN" ]; then \
    echo "Downloading Pyannote models with token..." && \
    python3 download_models.py && \
    rm download_models.py; \
    else \
    echo "WARNING: HUGGING_FACE_HUB_TOKEN not provided. Pyannote models will not be embedded." && \
    rm download_models.py; \
    fi

# Copy application source code
COPY src/ ./src/
COPY main.py .
COPY config/ ./config/
COPY static/ ./static/
COPY templates/ ./templates/

# Create necessary directories for the application
RUN mkdir -p data/uploads data/transcripts data/srt data/recordings temp

# Set proper permissions
RUN chmod +x main.py

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose the application port
EXPOSE 8000

# Start command for the application
CMD ["gunicorn", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "-w", "1", \
     "--timeout", "300", \
     "--graceful-timeout", "300", \
     "-b", "0.0.0.0:8000", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "main:app"]
