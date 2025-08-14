# TranscrevAI Configuration - Fixed Version
import os
from pathlib import Path

# Application package name
APP_PACKAGE_NAME = "TranscrevAI"

# Base directory (project root) - Fixed path
BASE_DIR = Path(r"C:\transcrevai_android\TranscrevAI_commit34")

# Data directories
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = DATA_DIR / "models"
TEMP_DIR = DATA_DIR / "temp"
RECORDINGS_DIR = DATA_DIR / "recordings"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
PROCESSED_DIR = DATA_DIR / "processed"

# Ensure directories exist
for directory in [DATA_DIR, MODEL_DIR, TEMP_DIR, RECORDINGS_DIR, TRANSCRIPTS_DIR, PROCESSED_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Audio settings
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHUNK_SIZE = 4096

# Supported languages and their model URLs
LANGUAGE_MODELS = {
    "en": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
    "pt": "https://alphacephei.com/vosk/models/vosk-model-small-pt-0.3.zip",
    "es": "https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip"
}

# FastAPI Configuration
FASTAPI_CONFIG = {
    "title": "TranscrevAI",
    "description": "Real-time Audio Transcription and Speaker Diarization Service",
    "version": "2.0.0",
    "host": "0.0.0.0",
    "port": 8000,
    "reload": True
}

# WebSocket Configuration
WEBSOCKET_CONFIG = {
    "ping_interval": 20,
    "ping_timeout": 10,
    "max_size": 1024 * 1024  # 1MB max message size
}

# Logging configuration for FastAPI
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
        "access": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "access": {
            "formatter": "access",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "formatter": "default",
            "class": "logging.FileHandler",
            "filename": "logs/app.log",
            "mode": "a",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["default", "file"],
    },
    "loggers": {
        "uvicorn.access": {
            "handlers": ["access"],
            "level": "INFO",
            "propagate": False,
        },
        "fastapi": {
            "handlers": ["default", "file"],
            "level": "INFO",
            "propagate": False,
        },
    },
}

# Session Configuration
SESSION_CONFIG = {
    "timeout": 3600,  # 1 hour session timeout
    "cleanup_interval": 300,  # 5 minutes cleanup interval
    "max_sessions": 100  # Maximum concurrent sessions
}

# Audio Processing Configuration
AUDIO_CONFIG = {
    "max_recording_duration": 3600,  # 1 hour max recording
    "chunk_size": 4096,
    "sample_rate": 16000,
    "channels": 1,
    "format": "wav"
}

# Model Configuration
MODEL_CONFIG = {
    "cache_models": True,
    "max_cached_models": 3,
    "download_timeout": 300,  # 5 minutes
    "extraction_timeout": 180,  # 3 minutes
    "auto_download": True,  # Automatically download missing models
    "validate_on_startup": True,
}