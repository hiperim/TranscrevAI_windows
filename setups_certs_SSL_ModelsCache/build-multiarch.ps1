# PowerShell script to build Multi-Architecture Docker image (AMD64 + ARM64)
# Supports Intel/AMD CPUs and Apple Silicon
# Pushes to Docker Hub with platform manifests

$ErrorActionPreference = 'Stop'

Write-Host "Building TranscrevAI Multi-Architecture Docker image..."
Write-Host "   Platforms: linux/amd64, linux/arm64"
Write-Host ""

# Load HF token from .env
$hf_token = $null
if (Test-Path ".env") {
    $envContent = Get-Content .env
    $hfLine = $envContent | Where-Object { $_ -match 'HUGGING_FACE_HUB_TOKEN' }
    if ($hfLine) {
        $tokenValue = ($hfLine -split '=')[1].Trim()
        if (-not [string]::IsNullOrEmpty($tokenValue)) {
            $hf_token = $tokenValue
            Write-Host "Found HF token in .env"
        }
    }
}

if ([string]::IsNullOrEmpty($hf_token)) {
    Write-Host "Warning: HUGGING_FACE_HUB_TOKEN not found in .env"
    Write-Host "Pyannote models will NOT be embedded in the image"
    $continue = Read-Host "Continue anyway? (y/N)"
    if ($continue -ne 'y' -and $continue -ne 'Y') {
        exit 1
    }
}

# Check if buildx is available
try {
    docker buildx version | Out-Null
    Write-Host "Docker Buildx is available"
} catch {
    Write-Error "Docker Buildx is required for multi-arch builds. Please install Docker Desktop."
    exit 1
}

# Create buildx builder if it doesn't exist
$builderName = "transcrevai-multiarch"
try {
    docker buildx inspect $builderName 2>&1 | Out-Null
    Write-Host "Using existing builder: $builderName"
    docker buildx use $builderName
} catch {
    Write-Host "Creating buildx builder: $builderName"
    docker buildx create --name $builderName --use
}

# Bootstrap builder (downloads QEMU for cross-compilation)
Write-Host "Bootstrapping builder (may take a few minutes first time)..."
docker buildx inspect --bootstrap

# Get Docker Hub username (default: hiperim)
$dockerUsername = Read-Host "Enter Docker Hub username [hiperim]"
if ([string]::IsNullOrEmpty($dockerUsername)) {
    $dockerUsername = "hiperim"
}

# Get image tag (default: latest)
$imageTag = Read-Host "Enter image tag [latest]"
if ([string]::IsNullOrEmpty($imageTag)) {
    $imageTag = "latest"
}

$imageName = "$dockerUsername/transcrevai:$imageTag"

Write-Host ""
Write-Host "Building multi-arch image: $imageName"
Write-Host "   This will build for AMD64 (x86_64) and ARM64 (Apple Silicon)"
Write-Host ""

# Build and push multi-arch image
docker buildx build `
    --platform linux/amd64,linux/arm64 `
    --file Dockerfile.multiarch `
    --build-arg HUGGING_FACE_HUB_TOKEN="$hf_token" `
    --tag $imageName `
    --push `
    .

Write-Host ""
Write-Host "Multi-arch build complete!"
Write-Host ""
Write-Host "Image pushed to Docker Hub: $imageName"
Write-Host "   - Supports: linux/amd64 (Intel/AMD)"
Write-Host "   - Supports: linux/arm64 (Apple Silicon)"
Write-Host ""
Write-Host "To pull and run on any architecture:"
Write-Host "  docker pull $imageName"
Write-Host "  docker run -p 8000:8000 $imageName"
Write-Host ""
Write-Host "Docker will automatically select the correct architecture"
