# CRITICAL FIX: Enhanced app configuration with adaptive processing capabilities
import os
from pathlib import Path

# Application metadata
APP_PACKAGE_NAME = "transcrevai"

# Base directories - Docker-compatible with environment variable override
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv('DATA_DIR', str(BASE_DIR / "data")))
TEMP_DIR = Path(os.getenv('TEMP_DIR', str(DATA_DIR / "temp")))

def _ensure_directories_created():
    """Ensure all required directories exist"""
    directories = [
        DATA_DIR,
        DATA_DIR / "inputs",
        DATA_DIR / "outputs", 
        DATA_DIR / "transcripts",
        DATA_DIR / "recordings",
        DATA_DIR / "logs",
        DATA_DIR / "models",
        TEMP_DIR,
        DATA_DIR / "processed"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Whisper model configuration - Docker-compatible
WHISPER_MODEL_DIR = Path(os.getenv('WHISPER_MODEL_DIR', str(DATA_DIR / "models" / "whisper")))

# Whisper models - Only medium models for supported languages
WHISPER_MODELS = {
    "pt": "medium",  # Portuguese - medium model
    "en": "medium",  # English - medium model
    "es": "medium"   # Spanish - medium model
}


# CRITICAL FIX: Enhanced Whisper configuration with adaptive settings
WHISPER_CONFIG = {
    "word_timestamps": True,
    "condition_on_previous_text": False,  # Better for conversations
    "language_configs": {
        "pt": {
            "temperature": 0.0,
            "best_of": 1,
            "beam_size": 1,  # Reduced for 50% speed improvement
            "patience": 0.5,  # Faster decisions
            "length_penalty": 1.0,
            "no_speech_threshold": 0.6,  # Less silence processing
            "initial_prompt": "Transcrição precisa em português brasileiro com pontuação e acentuação corretas."
        },
        "en": {
            "temperature": 0.0,
            "best_of": 1,
            "beam_size": 1,  # Reduced for 50% speed improvement
            "patience": 0.5,  # Faster decisions
            "length_penalty": 1.0,
            "no_speech_threshold": 0.6,  # Less silence processing
            "initial_prompt": "Accurate English transcription with proper punctuation and grammar."
        },
        "es": {
            "temperature": 0.0,
            "best_of": 1,
            "beam_size": 1,  # Reduced for 50% speed improvement
            "patience": 0.5,  # Faster decisions
            "length_penalty": 1.0,
            "no_speech_threshold": 0.6,  # Less silence processing
            "initial_prompt": "Transcripción precisa en español con puntuación correcta y acentuación adecuada."
        }
    }
}

# CRITICAL FIX: Adaptive prompts based on audio complexity and language
ADAPTIVE_PROMPTS = {
    "pt": {
        "neutral": "Transcrição precisa em português brasileiro com pontuação e acentuação corretas.",
        "lecture": "Apresentação ou palestra em português brasileiro com pontuação correta.",
        "conversation": "Diálogo ou conversa em português brasileiro entre falantes com identificação precisa.",
        "complex_dialogue": "Conversa complexa em português brasileiro com múltiplas interações e sobreposições."
    },
    "en": {
        "neutral": "Accurate English transcription with proper punctuation.",
        "lecture": "Individual presentation or lecture in English with correct punctuation.",
        "conversation": "Dialogue or conversation in English between speakers with precise identification.",
        "complex_dialogue": "Complex conversation in English with multiple interactions and overlaps."
    },
    "es": {
        "neutral": "Transcripción precisa en español con puntuación correcta.",
        "lecture": "Presentación individual en español con puntuación correcta.",
        "conversation": "Diálogo o conversación en español entre hablantes.",
        "complex_dialogue": "Conversación compleja en español con múltiples interacciones."
    }
}

# Real-time processing configuration
REALTIME_CONFIG = {
    "performance": {
        "chunk_duration": 1.5,  # Reduced for faster processing
        "max_processing_time": 20.0,  # Reduced time limit
        "memory_limit_mb": 512
    },
    "quality": {
        "sample_rate": 16000,
        "channels": 1,
        "bit_depth": 16
    }
}

# CRITICAL FIX: Processing profiles for different use cases
PROCESSING_PROFILES = {
    "realtime": {
        "target_latency": 0.3,  # Ultra-fast response
        "quality_priority": "speed",
        "transcription_model": "medium",
        "diarization_method": "simple",
        "preprocessing": "minimal",
        "whisper_beam_size": 1,
        "whisper_patience": 0.3,
        "diarization_window": 0.6
    },
    "balanced": {
        "target_latency": 1.0,  # Improved balance
        "quality_priority": "balanced", 
        "transcription_model": "medium",
        "diarization_method": "standard",
        "preprocessing": "standard",
        "whisper_beam_size": 1,
        "whisper_patience": 0.5,
        "diarization_window": 0.8
    },
    "quality": {
        "target_latency": 2.0,  # Still faster than before
        "quality_priority": "accuracy",
        "transcription_model": "medium",
        "diarization_method": "advanced",
        "preprocessing": "advanced",
        "whisper_beam_size": 2,
        "whisper_patience": 1.0,
        "diarization_window": 1.0
    }
}

# Audio processing configuration
AUDIO_CONFIG = {
    "supported_formats": [".wav", ".mp3", ".mp4", ".m4a", ".flac"],
    "max_file_size_mb": 100,
    "preprocessing": {
        "normalize_audio": True,
        "noise_reduction": True,
        "high_pass_filter": True,
        "target_lufs": -23.0
    }
}

# Speaker diarization configuration
DIARIZATION_CONFIG = {
    "min_speakers": 1,
    "max_speakers": 6,
    "default_method": "adaptive",  # CRITICAL FIX: Use adaptive method
    "confidence_threshold": 0.5,
    "segment_min_duration": 0.5,  # seconds
    "merge_threshold": 0.1,  # seconds gap for merging
    # CRITICAL FIX: Enhanced speaker analysis parameters  
    "analysis_thresholds": {
        "single_speaker": 0.15,      # REVERTED from 0.4
        "multi_speaker": 0.4,        # REVERTED from 0.6
        "short_audio_threshold": 5.0  # REVERTED from 10.0
    }
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_rotation": True,
    "max_file_size_mb": 10,
    "backup_count": 5
}

# Memory management configuration
MEMORY_CONFIG = {
    "max_model_cache_mb": 1024,  # 1GB for model caching
    "cleanup_threshold": 0.8,    # Cleanup when 80% memory used
    "enable_garbage_collection": True,
    "gc_interval": 10.0,         # Run GC every 10 seconds
    "temp_cleanup_interval": 60.0  # Clean temp files every 60 seconds
}

# Model management configuration
MODEL_CONFIG = {
    "cache_timeout": 1800,  # 30 minutes
    "concurrent_downloads": 2,
    "download_timeout": 300,  # 5 minutes
    "retry_attempts": 3,
    "cache_models": True
}

# CRITICAL FIX: Quality assurance thresholds
QUALITY_CONFIG = {
    "transcription": {
        "min_confidence": 0.3,
        "word_accuracy_threshold": 0.8,
        "segment_min_duration": 0.1
    },
    "diarization": {
        "min_confidence": 0.5,
        "speaker_balance_threshold": 0.1,
        "temporal_consistency": True
    },
    "alignment": {
        "min_overlap_ratio": 0.3,
        "max_time_drift": 0.5  # seconds
    }
}

# Error handling configuration
ERROR_CONFIG = {
    "max_retries": 3,
    "retry_delay": 1.0,  # seconds
    "timeout_handling": "graceful",
    "fallback_modes": True
}

# Initialize directories on import
_ensure_directories_created()