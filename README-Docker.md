# TranscrevAI Docker Setup

## Quick Start for Reviewers

### Prerequisites
- Docker and Docker Compose installed
- At least 4GB RAM available for Docker

### One-Command Launch
```bash
# Clone and run in one command
git clone <repository-url>
cd transcrevai_windows
docker-compose up --build
```

The application will be available at: http://localhost:8000

## Development Workflow

### For Project Owner (Windows)
```bash
# Continue developing normally on Windows
python main.py
# or
uvicorn main:app --reload
```

### For Testing with Docker
```bash
# Production build
docker-compose up --build

# Development mode with hot reload
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

## Features
- ✅ **Cross-platform**: Runs on any OS with Docker
- ✅ **Fast performance**: Optimized container build
- ✅ **Persistent data**: Models and data survive container restarts
- ✅ **Health monitoring**: Built-in health checks
- ✅ **Resource limits**: Memory and CPU controls

## Architecture
- **Base Image**: Python 3.11 slim (optimized for size)
- **System Dependencies**: FFmpeg, audio libraries pre-installed
- **Data Persistence**: Docker volumes for models and user data
- **Security**: Non-root user execution
- **Networking**: Isolated Docker network

## Ports
- `8000`: Main application (production)
- `8001`: Additional port (development mode)

## Volumes
- `transcrevai_data`: Application data persistence
- `transcrevai_models`: Whisper models cache

## Environment Variables
- `DATA_DIR`: Data directory path (default: `/app/data`)
- `WHISPER_MODEL_DIR`: Whisper models directory
- `TEMP_DIR`: Temporary files directory
- `LOG_LEVEL`: Logging level (INFO/DEBUG)
- `ENVIRONMENT`: Runtime environment (production/development)

## Health Check
The application includes a health endpoint at `/health` that Docker uses to monitor container status.

## Troubleshooting

### Container won't start
```bash
# Check logs
docker-compose logs

# Remove and rebuild
docker-compose down
docker system prune -a
docker-compose up --build
```

### Performance issues
```bash
# Check resource usage
docker stats

# Adjust memory limits in docker-compose.yml
```

### Model download issues
First container startup may take longer as Whisper models download automatically.