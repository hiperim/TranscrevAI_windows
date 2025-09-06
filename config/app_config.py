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

# Whisper Models Configuration - Optimized per language
WHISPER_MODELS = {
    "en": "small.en",  # English-only optimized small model
    "pt": "base",      # Upgraded to base model for better Portuguese accuracy  
    "es": "base"       # Upgraded to base model for better Spanish accuracy
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

# Whisper Configuration - Optimized per Language
WHISPER_CONFIG = {
    "device": "cpu",  # Use "cuda" if GPU available
    "compute_type": "int8",  # Quantization for performance
    "download_root": str(WHISPER_MODEL_DIR),
    "language_detection": True,
    "word_timestamps": True,
    "condition_on_previous_text": True,
    "fp16": False,  # Avoid GPU dependency
    "compression_ratio_threshold": 2.4,
    "logprob_threshold": -1.0,
    
    # Language-specific optimized configurations
    "language_configs": {
        "pt": {  # Portuguese (Brazilian)
            "temperature": (0.0, 0.1),  # More conservative for better accuracy
            "best_of": 2,  # Reduced for speed while maintaining quality
            "beam_size": 3,  # Balanced beam search
            "no_speech_threshold": 0.55,  # Optimized for Portuguese speech patterns
            "initial_prompt": "Transcrição em português brasileiro com pontuação e acentuação corretas.",
            "patience": 1.2,
            "length_penalty": 1.1
        },
        "en": {  # English
            "temperature": (0.0, 0.2),  # Slightly more flexible
            "best_of": 3,  # Good balance for English
            "beam_size": 4,  # Optimal for English small.en model
            "no_speech_threshold": 0.6,  # Standard threshold
            "initial_prompt": "Professional English transcription with proper punctuation and grammar.",
            "patience": 1.0,
            "length_penalty": 1.0
        },
        "es": {  # Spanish
            "temperature": (0.0, 0.15),  # Balanced for Spanish variations
            "best_of": 2,  # Efficient for Spanish
            "beam_size": 3,  # Balanced beam search
            "no_speech_threshold": 0.55,  # Similar to Portuguese
            "initial_prompt": "Transcripción en español con acentuación y puntuación correctas.",
            "patience": 1.1,
            "length_penalty": 1.05
        }
    },
    
    # Fallback configuration for unsupported languages
    "fallback_config": {
        "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        "best_of": 5,
        "beam_size": 5,
        "no_speech_threshold": 0.6,
        "initial_prompt": "Audio transcription with proper punctuation.",
        "patience": 1.0,
        "length_penalty": 1.0
    }
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
        "target_lufs": -20.0,  # Less aggressive to avoid clipping (was -23.0)
        "fallback_peak_level": 0.7  # Reduced to prevent clipping (was 0.8)
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
        "threshold": 0.4,  # Higher threshold for less aggressive compression
        "ratio": 2.5,      # Lower ratio to prevent clipping (was 3.0)
        "final_amplitude_limit": 0.8  # Reduced limit to prevent clipping (was 0.9)
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

# PERFORMANCE OPTIMIZATIONS - Real-time Configuration Profile
# Merged from realtime_config.py for consolidated configuration
REALTIME_CONFIG = {
    "audio_preprocessing": {
        "enabled": True,
        "lufs_normalization": False,  # Disabled for speed
        "noise_reduction": False,     # Disabled for speed
        "simple_normalization_only": True,
        "high_pass_filter": {
            "enabled": True,
            "cutoff_freq": 80,
            "order": 2  # Reduced order for speed
        }
    },
    "whisper_optimization": {
        "model_size": "tiny",         # Fastest model
        "beam_size": 1,               # No beam search
        "best_of": 1,                 # Single candidate
        "temperature": 0.0,           # Deterministic
        "word_timestamps": False,     # Disabled for speed
        "fp16": False,                # CPU compatibility
        "condition_on_previous_text": False  # Disabled for speed
    },
    "diarization": {
        "method": "energy_based_fast", # Fastest method
        "max_speakers": 3,             # Limit complexity
        "min_segment_duration": 1.0,   # Larger segments
        "enable_temporal_smoothing": False,  # Disabled for speed
        "skip_feature_extraction": True     # Use simple energy only
    },
    "memory_management": {
        "max_buffer_size": 1024 * 1024,  # 1MB buffer
        "enable_garbage_collection": True,
        "cleanup_interval": 10.0,         # Clean every 10s
        "max_cached_models": 1            # Limit model cache
    },
    "performance": {
        "target_latency": 0.5,        # 500ms target
        "max_processing_time": 2.0,   # 2s timeout
        "enable_performance_logging": False,  # Disabled for speed
        "chunk_duration": 2.0         # Process in 2s chunks
    }
}

# Quality vs Speed profiles for different use cases
PROCESSING_PROFILES = {
    "realtime": {
        "transcription_model": "tiny",
        "diarization_method": "energy_based",
        "preprocessing": "minimal",
        "target_latency": 0.5,  # 500ms
        "quality_score": 60     # 60% quality for max speed
    },
    "balanced": {
        "transcription_model": "small", 
        "diarization_method": "frequency_analysis",
        "preprocessing": "standard",
        "target_latency": 2.0,  # 2s
        "quality_score": 80     # 80% quality
    },
    "quality": {
        "transcription_model": "base",
        "diarization_method": "pyaudioanalysis", 
        "preprocessing": "full",
        "target_latency": 10.0, # 10s
        "quality_score": 95     # 95% quality
    }
}

# Language-specific preprocessing optimizations
LANGUAGE_PREPROCESSING = {
    "pt": {
        "lufs_target": -20.0,  # Optimized for Portuguese speech patterns
        "high_pass_cutoff": 85,  # Portuguese phonetics
        "noise_reduction": "moderate",
        "model_preference": "base"
    },
    "en": {
        "lufs_target": -23.0,  # Standard for English
        "high_pass_cutoff": 80,
        "noise_reduction": "light",
        "model_preference": "small.en"
    },
    "es": {
        "lufs_target": -21.0,  # Optimized for Spanish
        "high_pass_cutoff": 90,  # Spanish consonants
        "noise_reduction": "moderate",
        "model_preference": "base"
    }
}

# Memory optimization settings
REALTIME_MEMORY_CONFIG = {
    "max_audio_buffer_mb": 64,      # 64MB audio buffer limit
    "max_model_cache_mb": 256,      # 256MB model cache limit
    "enable_memory_monitoring": True,
    "memory_cleanup_threshold": 0.8, # Cleanup at 80% usage
    "temp_file_cleanup_interval": 30 # Clean temp files every 30s
}

# Configuration validation function
def validate_config():
    """Validate configuration values"""
    errors = []
    
    # Validate basic audio settings
    if DEFAULT_SAMPLE_RATE <= 0:
        errors.append("DEFAULT_SAMPLE_RATE must be positive")
    
    if DEFAULT_CHUNK_SIZE <= 0:
        errors.append("DEFAULT_CHUNK_SIZE must be positive")
    
    # Validate Whisper models
    if not WHISPER_MODELS:
        errors.append("WHISPER_MODELS cannot be empty")
    
    for lang, model in WHISPER_MODELS.items():
        if not isinstance(model, str) or not model.strip():
            errors.append(f"Invalid model for language {lang}: {model}")
    
    # Validate FastAPI configuration
    if not isinstance(FASTAPI_CONFIG.get("port", 0), int) or FASTAPI_CONFIG.get("port", 0) <= 0:
        errors.append("FASTAPI_CONFIG port must be a positive integer")
    
    # Validate processing profiles
    for profile_name, profile in PROCESSING_PROFILES.items():
        if not isinstance(profile.get("target_latency", 0), (int, float)) or profile.get("target_latency", 0) <= 0:
            errors.append(f"Invalid target_latency for profile {profile_name}")
        
        if not isinstance(profile.get("quality_score", 0), int) or not (0 <= profile.get("quality_score", 0) <= 100):
            errors.append(f"Invalid quality_score for profile {profile_name} (must be 0-100)")
    
    # Validate realtime config
    perf_config = REALTIME_CONFIG.get("performance", {})
    if not isinstance(perf_config.get("chunk_duration", 0), (int, float)) or perf_config.get("chunk_duration", 0) <= 0:
        errors.append("REALTIME_CONFIG performance.chunk_duration must be positive")
    
    # Validate memory config
    mem_config = REALTIME_MEMORY_CONFIG
    if not isinstance(mem_config.get("memory_cleanup_threshold", 0), (int, float)) or not (0 < mem_config.get("memory_cleanup_threshold", 0) <= 1):
        errors.append("memory_cleanup_threshold must be between 0 and 1")
    
    if errors:
        raise ValueError("Configuration validation errors:\n" + "\n".join(f"- {error}" for error in errors))
    
    return True

# Validate configuration on module load
try:
    validate_config()
    print("[OK] Configuration validation passed")
except ValueError as e:
    print(f"[ERROR] Configuration validation failed:\n{e}")
    raise