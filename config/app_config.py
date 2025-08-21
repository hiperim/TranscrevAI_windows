# TranscrevAI Configuration - Fixed Version
import os
from pathlib import Path

# Application package name
APP_PACKAGE_NAME = "TranscrevAI"

# Base directory (project root) - Fixed path
BASE_DIR = Path(r"c:\\TranscrevAI_windows")

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
# Using larger, more accurate models for better transcription quality
LANGUAGE_MODELS = {
    "en": "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip",  # Large model ~1.8GB (was small ~50MB)
    "pt": "https://alphacephei.com/vosk/models/vosk-model-pt-fb-v0.1.1-pruned.zip",  # Large model ~1.2GB (was small ~39MB)
    "es": "https://alphacephei.com/vosk/models/vosk-model-es-0.42.zip"  # Large model ~1.4GB (was small ~39MB)
}

# Alternative smaller models for faster download/testing (optional)
LANGUAGE_MODELS_SMALL = {
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
    "download_timeout": 600,  # 10 minutes (increased for large models)
    "extraction_timeout": 300,  # 5 minutes (increased for large models)
    "auto_download": True,  # Automatically download missing models
    "validate_on_startup": True,
    "use_large_models": True,  # Set to False to use smaller, faster models
    "model_preference": "accuracy"  # "accuracy" for large models, "speed" for small models
}

# ENHANCED: Speaker Diarization Configuration
DIARIZATION_CONFIG = {
    "method": "frequency_analysis",  # Primary method: "frequency_analysis", "pyaudioanalysis", "sklearn"
    "min_speakers": 2,  # Minimum number of speakers to detect
    "max_speakers": 5,  # Maximum number of speakers to detect
    "window_duration": 1.0,  # Analysis window duration in seconds
    "hop_duration": 0.5,  # Hop duration for overlapping windows
    "frequency_features": {
        "f0_range": [50, 500],  # Fundamental frequency range in Hz
        "formant_analysis": True,  # Enable formant (F1, F2) analysis
        "spectral_analysis": True,  # Enable spectral centroid and energy ratio
        "low_freq_threshold": 1000,  # Threshold for low/high frequency energy ratio
    },
    "clustering": {
        "algorithm": "kmeans",  # "kmeans", "agglomerative"
        "n_components_pca": 3,  # PCA components for dimensionality reduction
        "random_state": 42,  # For reproducible results
        "silhouette_threshold": 0.3  # Minimum silhouette score for cluster quality
    },
    "temporal_smoothing": {
        "enabled": True,
        "min_segment_duration": 0.5,  # Minimum segment duration in seconds
        "median_filter_size": 3,  # Size of median filter for smoothing
        "merge_gap_threshold": 0.1  # Maximum gap between segments to merge
    },
    "fallback_methods": ["sklearn", "energy_based", "single_speaker"]
}

# ENHANCED: Audio Preprocessing Configuration  
AUDIO_PREPROCESSING_CONFIG = {
    "enabled": True,
    "lufs_normalization": {
        "enabled": True,
        "target_lufs": -23.0,  # Broadcast standard
        "fallback_peak_level": 0.8
    },
    "noise_reduction": {
        "spectral_subtraction": True,  # Advanced noise reduction with noisereduce
        "high_pass_filter": {
            "enabled": True,
            "cutoff_freq": 80,  # Hz
            "order": 4
        },
        "low_pass_filter": {
            "enabled": True, 
            "cutoff_freq": 8000,  # Hz
            "order": 6
        }
    },
    "stereo_processing": {
        "channel_separation_analysis": True,
        "energy_difference_threshold": 0.3,  # Threshold for using single channel
        "balanced_mixing": True  # Use weighted mixing for balanced stereo
    },
    "dynamic_range": {
        "compression_enabled": True,
        "threshold": 0.3,
        "ratio": 3.0,
        "final_amplitude_limit": 0.9
    },
    "resampling": {
        "target_sample_rate": 16000,
        "quality": "high"  # librosa resampling quality
    }
}

# ENHANCED: Transcription Enhancement Configuration
TRANSCRIPTION_CONFIG = {
    "vosk_optimization": {
        "word_timestamps": True,  # Enable word-level timestamps
        "partial_words": True,  # Enable partial word results
        "max_alternatives": 3,  # Multiple recognition alternatives
        "nlsml_output": True,  # Structured output format
        "speaker_adaptation": False  # Speaker adaptation (if available)
    },
    "post_processing": {
        "text_enhancement": True,
        "duplicate_filtering": True,
        "confidence_threshold": 0.0,  # Minimum confidence (keep current behavior)
        "common_word_fixes": True,  # Fix common transcription errors
        "repetition_removal": {
            "enabled": True,
            "max_repetitions": 2
        },
        "punctuation": {
            "auto_punctuation": True,
            "sentence_boundaries": True
        }
    },
    "chunk_processing": {
        "chunk_size": 16384,  # Audio chunk size for processing
        "overlap_duration": 0.1  # Overlap between chunks
    }
}

# ENHANCED: Alignment Configuration
ALIGNMENT_CONFIG = {
    "method": "enhanced_temporal",  # Alignment method
    "temporal_window": {
        "extension_ms": 200,  # Extend diarization windows by ±200ms
        "proximity_threshold_ms": 300,  # Proximity threshold for boundary detection
        "boundary_bonus": 0.3  # Score bonus for boundary words
    },
    "scoring": {
        "overlap_weight": 0.4,
        "proximity_weight": 0.3,
        "boundary_weight": 0.2,
        "coverage_weight": 0.1
    },
    "filtering": {
        "min_overlap_ratio": 0.3,
        "min_weighted_score": 0.4,
        "min_coverage_score": 0.6,
        "min_text_length": 2  # Minimum text length to include
    },
    "timing_adjustment": {
        "prefer_transcription_boundaries": True,
        "max_boundary_adjustment_ms": 500  # Maximum timing adjustment
    }
}