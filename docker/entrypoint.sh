#!/bin/bash

# TranscrevAI Docker Entrypoint Script

set -e

echo "Starting TranscrevAI container..."

# Check if data directories exist and create if needed
echo "Setting up data directories..."
mkdir -p /app/data/{inputs,outputs,transcripts,recordings,logs,models,temp,processed}

# Check FFmpeg availability
echo "Checking FFmpeg availability..."
if command -v ffmpeg >/dev/null 2>&1; then
    echo "✓ FFmpeg is available: $(ffmpeg -version | head -n1)"
else
    echo "⚠ FFmpeg not found, but continuing..."
fi

# Check Python dependencies
echo "Checking Python environment..."
python -c "import torch, whisper, fastapi; print('✓ Core dependencies loaded')" || {
    echo "❌ Failed to import core dependencies"
    exit 1
}

# Set up environment variables for Docker
export DATA_DIR="/app/data"
export TEMP_DIR="/app/data/temp"
export WHISPER_MODEL_DIR="/app/data/models/whisper"

# Create logs directory and start logging
mkdir -p /app/data/logs
echo "$(date): Container started" >> /app/data/logs/container.log

echo "✓ Setup complete. Starting application..."

# Execute the provided command
exec "$@"