#!/bin/bash
# Script to build Docker image with embedded Pyannote models
# This script reads HUGGING_FACE_HUB_TOKEN from .env file

set -e

echo "🔨 Building TranscrevAI Docker image with embedded ML models..."

# Load HF token from .env
if [ -f .env ]; then
    export $(grep HUGGING_FACE_HUB_TOKEN .env | xargs)
    if [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
        echo "⚠️  Warning: HUGGING_FACE_HUB_TOKEN not found in .env"
        echo "Pyannote models will NOT be embedded in the image"
        echo "You can still build, but diarization won't work without runtime token"
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo "✅ Found HF token in .env"
    fi
else
    echo "❌ Error: .env file not found"
    echo "Please create .env with HUGGING_FACE_HUB_TOKEN=your_token_here"
    exit 1
fi

# Build with docker compose
echo "🚀 Building with docker compose..."
docker compose build --build-arg HUGGING_FACE_HUB_TOKEN="$HUGGING_FACE_HUB_TOKEN"

echo "✅ Build complete!"
echo "📦 Image size:"
docker images transcrevai_windows-transcrevai:latest --format "{{.Size}}"

echo ""
echo "To run the container:"
echo "  docker compose up -d"
