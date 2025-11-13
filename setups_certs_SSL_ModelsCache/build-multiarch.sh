#!/bin/bash
# Script to build Multi-Architecture Docker image (AMD64 + ARM64)
# Supports Intel/AMD CPUs and Apple Silicon (M1/M2/M3)
# Pushes to Docker Hub with platform manifests

set -e

echo "🌍 Building TranscrevAI Multi-Architecture Docker image..."
echo "   Platforms: linux/amd64, linux/arm64"
echo ""

# Load HF token from .env
if [ -f .env ]; then
    export $(grep HUGGING_FACE_HUB_TOKEN .env | xargs)
    if [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
        echo "⚠️  Warning: HUGGING_FACE_HUB_TOKEN not found in .env"
        echo "Pyannote models will NOT be embedded in the image"
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

# Check if buildx is available
if ! docker buildx version > /dev/null 2>&1; then
    echo "❌ Error: Docker Buildx is required for multi-arch builds"
    echo "Please install Docker Desktop or enable buildx"
    exit 1
fi

# Create buildx builder if it doesn't exist
BUILDER_NAME="transcrevai-multiarch"
if ! docker buildx inspect $BUILDER_NAME > /dev/null 2>&1; then
    echo "📦 Creating buildx builder: $BUILDER_NAME"
    docker buildx create --name $BUILDER_NAME --use
else
    echo "✅ Using existing builder: $BUILDER_NAME"
    docker buildx use $BUILDER_NAME
fi

# Bootstrap builder (downloads QEMU for cross-compilation)
echo "🔧 Bootstrapping builder (may take a few minutes first time)..."
docker buildx inspect --bootstrap

# Get Docker Hub username (default: hiperim)
read -p "Enter Docker Hub username [hiperim]: " DOCKER_USERNAME
DOCKER_USERNAME=${DOCKER_USERNAME:-hiperim}

# Get image tag (default: latest)
read -p "Enter image tag [latest]: " IMAGE_TAG
IMAGE_TAG=${IMAGE_TAG:-latest}

IMAGE_NAME="$DOCKER_USERNAME/transcrevai:$IMAGE_TAG"

echo ""
echo "🚀 Building multi-arch image: $IMAGE_NAME"
echo "   This will build for AMD64 (x86_64) and ARM64 (Apple Silicon)"
echo ""

# Build and push multi-arch image
# --platform: specifies target architectures
# --push: automatically pushes to Docker Hub
# --build-arg: passes HF token for model download
docker buildx build \
    --platform linux/amd64,linux/arm64 \
    --file Dockerfile.multiarch \
    --build-arg HUGGING_FACE_HUB_TOKEN="$HUGGING_FACE_HUB_TOKEN" \
    --tag $IMAGE_NAME \
    --push \
    .

echo ""
echo "✅ Multi-arch build complete!"
echo ""
echo "📦 Image pushed to Docker Hub: $IMAGE_NAME"
echo "   - Supports: linux/amd64 (Intel/AMD)"
echo "   - Supports: linux/arm64 (Apple Silicon M1/M2/M3)"
echo ""
echo "To pull and run on any architecture:"
echo "  docker pull $IMAGE_NAME"
echo "  docker run -p 8000:8000 $IMAGE_NAME"
echo ""
echo "Docker will automatically select the correct architecture!"
