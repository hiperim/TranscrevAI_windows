# PowerShell script to build Multi-Architecture Docker image (AMD64 + ARM64)
# Supports Intel/AMD CPUs and Apple Silicon
# Pushes to Docker Hub with platform manifests

$ErrorActionPreference = 'Stop'

Write-Host ""
Write-Host "Building TranscrevAI Multi-Architecture Docker Image" -ForegroundColor Cyan
Write-Host "   Platforms: linux/amd64, linux/arm64" -ForegroundColor Gray
Write-Host "   Embedded ML Models with offline support" -ForegroundColor Gray
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
            Write-Host "Found HF token in .env" -ForegroundColor Green
        }
    }
}

if ([string]::IsNullOrEmpty($hf_token)) {
    Write-Host "Warning: HUGGING_FACE_HUB_TOKEN not found in .env" -ForegroundColor Yellow
    Write-Host "   Pyannote models will NOT be embedded in the image" -ForegroundColor Yellow
    Write-Host "   Diarization will not work without runtime token" -ForegroundColor Yellow
    $continue = Read-Host "`nContinue anyway? (y/N)"
    if ($continue -ne 'y' -and $continue -ne 'Y') {
        exit 1
    }
}

# Ask build mode: local test or multi-arch push
Write-Host ""
Write-Host "Select build mode:" -ForegroundColor Cyan
Write-Host "  1) Local test build (single platform, faster)" -ForegroundColor White
Write-Host "  2) Multi-arch build and push to Docker Hub (slower)" -ForegroundColor White
$buildMode = Read-Host "`nEnter choice [1]"
if ([string]::IsNullOrEmpty($buildMode)) {
    $buildMode = "1"
}

if ($buildMode -eq "1") {
    # Local single-platform build
    Write-Host ""
    Write-Host "Building local test image..." -ForegroundColor Cyan
    Write-Host "   Platform: linux/amd64 only" -ForegroundColor Gray
    Write-Host ""

    docker build `
        --no-cache `
        --platform linux/amd64 `
        --file Dockerfile.multiarch `
        --build-arg HUGGING_FACE_HUB_TOKEN="$hf_token" `
        --tag transcrevai:latest `
        .

    Write-Host ""
    Write-Host "Build complete!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Image size:" -ForegroundColor Cyan
    docker images transcrevai:latest --format "{{.Size}}"
    Write-Host ""
    Write-Host "To test the container:" -ForegroundColor Cyan
    Write-Host "  docker run -p 8000:8000 transcrevai:latest" -ForegroundColor White

} else {
    # Multi-arch build with push

    # Check if buildx is available
    try {
        docker buildx version | Out-Null
        Write-Host "Docker Buildx is available" -ForegroundColor Green
    } catch {
        Write-Error "Docker Buildx is required for multi-arch builds. Please install Docker Desktop."
        exit 1
    }

    # Create buildx builder if it doesn't exist
    $builderName = "transcrevai-multiarch"
    try {
        docker buildx inspect $builderName 2>&1 | Out-Null
        Write-Host "Using existing builder: $builderName" -ForegroundColor Green
        docker buildx use $builderName
    } catch {
        Write-Host "Creating buildx builder: $builderName" -ForegroundColor Yellow
        docker buildx create --name $builderName --use
    }

    # Bootstrap builder (downloads QEMU for cross-compilation)
    Write-Host "Bootstrapping builder (may take a few minutes first time)..." -ForegroundColor Yellow
    docker buildx inspect --bootstrap

    # Get Docker Hub username (default: hiperim)
    Write-Host ""
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
    Write-Host "Building multi-arch image: $imageName" -ForegroundColor Cyan
    Write-Host "   AMD64 (Intel/AMD) + ARM64 (Apple Silicon)" -ForegroundColor Gray
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
    Write-Host "Multi-arch build complete!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Image pushed to Docker Hub: $imageName" -ForegroundColor Cyan
    Write-Host "   Supports: linux/amd64 (Intel/AMD)" -ForegroundColor Gray
    Write-Host "   Supports: linux/arm64 (Apple Silicon)" -ForegroundColor Gray
    Write-Host ""
    Write-Host "To pull and run on any architecture:" -ForegroundColor Cyan
    Write-Host "  docker pull $imageName" -ForegroundColor White
    Write-Host "  docker run -p 8000:8000 $imageName" -ForegroundColor White
    Write-Host ""
    Write-Host "Docker will automatically select the correct architecture" -ForegroundColor Gray
}
