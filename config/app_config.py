# TranscrevAI Configuration - Cross-Platform Version
import os
import sys
from pathlib import Path

# Application package name
APP_PACKAGE_NAME = "TranscrevAI"

# Cross-platform base directory detection
def get_base_directory():
    """Get the application base directory based on platform and environment"""
    # Check if running from source directory
    current_dir = Path(__file__).parent.parent
    if (current_dir / "src").exists() and (current_dir.name.startswith("TranscrevAI")):
        return current_dir
    
    # Platform-specific application data directories
    if sys.platform == "win32":
        # Windows: Use AppData/Local
        base = Path(os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))) / APP_PACKAGE_NAME
    elif sys.platform == "darwin":
        # macOS: Use ~/Library/Application Support
        base = Path.home() / "Library" / "Application Support" / APP_PACKAGE_NAME
    else:
        # Linux and other Unix-like: Use ~/.local/share
        base = Path.home() / ".local" / "share" / APP_PACKAGE_NAME
    
    return base

BASE_DIR = get_base_directory()

# Data directories
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = DATA_DIR / "models"
TEMP_DIR = DATA_DIR / "temp"
RECORDINGS_DIR = DATA_DIR / "recordings"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
PROCESSED_DIR = DATA_DIR / "processed"

# Lazy directory creation function
def ensure_directories():
    """Create required directories only when needed"""
    directories = [DATA_DIR, MODEL_DIR, TEMP_DIR, RECORDINGS_DIR, TRANSCRIPTS_DIR, PROCESSED_DIR]
    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to create directory {directory}: {e}")

# Auto-create directories on first access
_directories_created = False

def _ensure_directories_created():
    """Ensure directories are created on first access"""
    global _directories_created
    if not _directories_created:
        ensure_directories()
        _directories_created = True

# Audio settings
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHUNK_SIZE = 4096

# Whisper Models Configuration
WHISPER_MODELS = {
    "en": "small.en",  # English-only optimized small model
    "pt": "small",     # Multilingual small model for Portuguese  
    "es": "small"      # Multilingual small model for Spanish
}

# Model directories
WHISPER_MODEL_DIR = MODEL_DIR / "whisper"
PYAUDIOANALYSIS_MODEL_DIR = MODEL_DIR / "pyaudioanalysis"

# PyAudioAnalysis Configuration - Free speaker diarization
PYAUDIOANALYSIS_CONFIG = {
    "default_speakers": 2,  # Default number of speakers
    "classifier": "svm",  # Classification method: svm, randomforest, gradientboosting
    "vad_preprocessing": True,  # Voice Activity Detection preprocessing
    "lda_dim_reduction": 0.5,  # LDA dimensionality reduction factor
    "segment_duration": 0.5,  # Duration of each audio segment in seconds
    "feature_extraction": "mfcc",  # Feature type: mfcc, chroma, spectral
    "clustering_method": "kmeans"  # Clustering: kmeans, hierarchical
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

# Whisper Configuration
WHISPER_CONFIG = {
    "device": "cpu",  # Use "cuda" if GPU available
    "compute_type": "int8",  # Quantization for performance
    "download_root": str(WHISPER_MODEL_DIR),
    "language_detection": True,
    "word_timestamps": True,
    "temperature": 0.0,  # Deterministic output
    "best_of": 1,  # Single decode for speed
    "beam_size": 5,
    "patience": 1.0,
    "length_penalty": 1.0,
    "suppress_tokens": "-1",
    "initial_prompt": None,
    "condition_on_previous_text": True,
    "fp16": False,  # Avoid GPU dependency
    "compression_ratio_threshold": 2.4,
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.6
}

# Speaker Diarization Configuration - Using PyAudioAnalysis (Free)
# PyAnnote configuration removed - replaced by PYAUDIOANALYSIS_CONFIG above

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
    "whisper_optimization": {
        "word_timestamps": True,  # Enable word-level timestamps
        "language": "auto",  # Automatic language detection
        "beam_size": 5,  # Beam search width
        "best_of": 5,  # Number of candidates when using sampling
        "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)  # Temperature fallbacks
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