# PowerShell script to build the Docker image, reading the HF token from .env

# Stop script on any error
$ErrorActionPreference = 'Stop'

Write-Host "🔨 Building TranscrevAI Docker image with embedded ML models..."

# Load HF token from .env file
$hf_token = $null
if (Test-Path ".env") {
    $envContent = Get-Content .env
    $hfLine = $envContent | Where-Object { $_ -match 'HUGGING_FACE_HUB_TOKEN' }
    if ($hfLine) {
        $tokenValue = ($hfLine -split '=')[1].Trim()
        if (-not [string]::IsNullOrEmpty($tokenValue)) {
            $hf_token = $tokenValue
            Write-Host "✅ Found HF token in .env"
        }
    }
}

# Check if token was found
if ([string]::IsNullOrEmpty($hf_token)) {
    Write-Error "HUGGING_FACE_HUB_TOKEN not found or is empty in .env file. Cannot proceed with the build."
    exit 1
}

# Build with docker compose, passing the token as a build argument
Write-Host "🚀 Building with docker compose..."
docker compose build --no-cache --build-arg HUGGING_FACE_HUB_TOKEN="$hf_token"

Write-Host "`n✅ Build complete!"
Write-Host "📦 Image size:"
docker images transcrevai_windows-transcrevai:latest --format "{{.Size}}"

Write-Host "`nTo run the container:"
Write-Host "  docker compose up -d"
