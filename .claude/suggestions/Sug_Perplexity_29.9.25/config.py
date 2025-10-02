"""
TranscrevAI Optimized - PT-BR Configuration
Configurações otimizadas para português brasileiro com foco em performance
"""

import os
from pathlib import Path
from typing import Dict, Any, List

# ============================================================================
# PROJECT PATHS & DIRECTORIES
# ============================================================================

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
CACHE_DIR = PROJECT_ROOT / "cache"
MODELS_DIR = CACHE_DIR / "models"
TEMPLATES_DIR = PROJECT_ROOT / "templates"

# Data subdirectories
RECORDINGS_DIR = DATA_DIR / "recordings"
TEMP_DIR = DATA_DIR / "temp"
OUTPUT_DIR = DATA_DIR / "output"

# Ensure directories exist
for directory in [DATA_DIR, LOGS_DIR, CACHE_DIR, MODELS_DIR, RECORDINGS_DIR, TEMP_DIR, OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# SERVER CONFIGURATION
# ============================================================================

# Server settings
SERVER_HOST = os.getenv("TRANSCREVAI_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("TRANSCREVAI_PORT", "8001"))
DEBUG_MODE = os.getenv("TRANSCREVAI_DEBUG", "false").lower() == "true"
LOG_LEVEL = os.getenv("TRANSCREVAI_LOG_LEVEL", "INFO").upper()

# WebSocket settings
WEBSOCKET_PING_INTERVAL = 20  # seconds
WEBSOCKET_PING_TIMEOUT = 10   # seconds
WEBSOCKET_MAX_SIZE = 16 * 1024 * 1024  # 16MB

# ============================================================================
# HARDWARE & PERFORMANCE CONFIGURATION
# ============================================================================

# Hardware detection and optimization
CPU_CORES = int(os.getenv("TRANSCREVAI_CPU_CORES", str(os.cpu_count() or 4)))
MAX_MEMORY_MB = int(os.getenv("TRANSCREVAI_MAX_MEMORY_MB", "2048"))  # 2GB default
ENABLE_CACHE = os.getenv("TRANSCREVAI_ENABLE_CACHE", "true").lower() == "true"

# Performance thresholds
MEMORY_WARNING_THRESHOLD = 0.75  # 75% RAM usage warning
MEMORY_EMERGENCY_THRESHOLD = 0.85  # 85% RAM usage emergency
MEMORY_CRITICAL_THRESHOLD = 0.90  # 90% RAM usage critical
STREAMING_MODE_THRESHOLD = 0.80  # 80% RAM usage enables streaming

# Processing targets
TARGET_PROCESSING_RATIO_WARM = 0.65  # 0.6-0.75x for warm starts
TARGET_PROCESSING_RATIO_COLD = 1.25  # 1.0-1.5x for cold starts
MAX_STARTUP_TIME_SECONDS = 5.0  # Progressive loading target

# ============================================================================
# WHISPER MODEL CONFIGURATION - PT-BR OPTIMIZED
# ============================================================================

# PT-BR exclusive model configuration
WHISPER_MODEL_NAME = "medium"  # Fixed - optimized for PT-BR
WHISPER_LANGUAGE = "pt"  # Fixed - Portuguese Brazilian only
WHISPER_TASK = "transcribe"  # Fixed task

# Model paths
WHISPER_MODELS_PATH = str(MODELS_DIR)
WHISPER_CACHE_DIR = str(MODELS_DIR / "whisper_cache")

# PT-BR Optimized Whisper Parameters
WHISPER_CONFIG = {
    "model": WHISPER_MODEL_NAME,
    "language": WHISPER_LANGUAGE,
    "task": WHISPER_TASK,
    
    # Performance optimizations
    "fp16": False,  # CPU-only, use fp32
    "verbose": False,
    
    # Accuracy optimizations for PT-BR
    "beam_size": 1,  # Already optimized
    "best_of": 1,  # Already optimized  
    "temperature": 0.1,  # Slightly higher for better accuracy
    
    # Speed optimizations
    "compression_ratio_threshold": 1.6,  # Optimized for PT-BR
    "logprob_threshold": -0.8,  # More conservative for PT-BR accuracy
    "no_speech_threshold": 0.85,  # Adjusted for Portuguese
    
    # PT-BR specific settings
    "condition_on_previous_text": False,  # Better for conversation
    "word_timestamps": True,  # Enable for SRT generation
    "suppress_blank": True,
    "suppress_tokens": [-1],
    "without_timestamps": False,  # We want timestamps
    
    # Audio preprocessing
    "hallucination_silence_threshold": 1.0,
}

# PT-BR Contextual Corrections and Prompts
PT_BR_CORRECTIONS = {
    # Common PT-BR transcription fixes
    "pos": "pois",
    "ta": "está", 
    "tá": "está",
    "pa": "para",
    "pro": "para o",
    "pra": "para",
    "dum": "de um",
    "numa": "em uma",
    "numas": "em umas",
    
    # Question words with proper accents
    "que": "que",  # Context-dependent
    "como": "como",  # Context-dependent  
    "quando": "quando",  # Context-dependent
    "onde": "onde",  # Context-dependent
    
    # Common conversation markers
    "né": "né",
    "sabe": "sabe",
    "tipo assim": "tipo assim",
    "a gente": "a gente",
    "cadê": "onde está",
    "então": "então",
    "daí": "daí",
}

# PT-BR Conversation Context Prompts
PT_BR_INITIAL_PROMPTS = {
    "neutral": "Esta é uma transcrição em português brasileiro com fala natural e expressões coloquiais.",
    "formal": "Esta é uma transcrição formal em português brasileiro com linguagem técnica e acadêmica.", 
    "conversation": "Esta é uma conversa em português brasileiro entre duas ou mais pessoas com linguagem informal.",
    "interview": "Esta é uma entrevista em português brasileiro com perguntas e respostas estruturadas.",
    "meeting": "Esta é uma reunião de trabalho em português brasileiro com discussões técnicas e decisões.",
}

# ============================================================================
# DIARIZATION CONFIGURATION
# ============================================================================

# Speaker diarization settings
DIARIZATION_CONFIG = {
    "max_speakers": 6,  # Reasonable limit for most cases
    "min_speakers": 1,  # Single speaker minimum
    "default_speakers": 2,  # Assume conversation by default
    
    # PyAudioAnalysis parameters - optimized for PT-BR
    "mid_window": 0.8,  # Balance between speed and accuracy
    "short_window": 0.15,  # Fast processing window
    "step": 0.05,  # Analysis step size
    
    # Quality thresholds
    "min_segment_duration": 0.5,  # Minimum 500ms segments
    "confidence_threshold": 0.6,  # Minimum confidence for segments
    
    # Overlapping detection
    "enable_overlap_detection": True,
    "overlap_threshold": 0.3,  # Energy variance threshold
    "merge_close_segments": True,
    "segment_merge_threshold": 0.2,  # 200ms merge threshold
}

# ============================================================================
# AUDIO PROCESSING CONFIGURATION
# ============================================================================

# Audio recording settings
AUDIO_CONFIG = {
    "sample_rate": 16000,  # Whisper's preferred sample rate
    "channels": 1,  # Mono recording
    "dtype": "float32",  # Consistent with Whisper
    "blocksize": 1024,  # Low latency
    "latency": "low",
    
    # Format support
    "supported_formats": [".wav", ".mp3", ".mp4", ".m4a", ".flac", ".ogg"],
    "recording_formats": ["wav", "mp4"],  # User choice for live recording
    
    # Quality settings
    "max_duration_minutes": 120,  # 2 hours max recording
    "silence_threshold": 0.01,  # Audio activity detection
    "max_file_size_mb": 500,  # 500MB max file size
}

# FFmpeg settings for format conversion
FFMPEG_CONFIG = {
    "audio_codec": "aac",
    "audio_bitrate": "192k",
    "sample_rate": 16000,
    "channels": 1,
    "timeout_seconds": 120,
}

# ============================================================================
# SUBTITLE GENERATION (SRT) CONFIGURATION
# ============================================================================

# SRT generation settings
SRT_CONFIG = {
    "max_line_length": 84,  # Characters per line
    "max_lines_per_subtitle": 2,  # Lines per subtitle block
    "min_duration_ms": 500,  # Minimum subtitle duration
    "max_duration_ms": 7000,  # Maximum subtitle duration
    "gap_threshold_ms": 200,  # Merge threshold between segments
    
    # Timing adjustments
    "start_offset_ms": 0,  # Timing offset for synchronization
    "end_padding_ms": 100,  # Extra time at end of subtitle
    
    # Speaker labels
    "include_speaker_labels": True,
    "speaker_label_format": "[Falante {speaker_id}]",
    "unknown_speaker_label": "[Falante]",
}

# ============================================================================
# CACHE & STORAGE CONFIGURATION  
# ============================================================================

# Model cache settings
CACHE_CONFIG = {
    "enable_model_cache": ENABLE_CACHE,
    "cache_ttl_hours": 24,  # 24 hour cache TTL
    "max_cache_size_mb": 1024,  # 1GB cache limit
    "cleanup_interval_hours": 6,  # Cache cleanup frequency
    
    # LRU eviction settings
    "enable_lru_eviction": True,
    "max_cached_models": 3,  # Maximum models in memory
}

# File cleanup settings
CLEANUP_CONFIG = {
    "auto_cleanup_temp": True,
    "temp_file_ttl_hours": 2,  # 2 hour temp file TTL
    "auto_cleanup_recordings": False,  # Keep user recordings
    "max_recordings_count": 50,  # Limit recording files
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Logging settings
LOGGING_CONFIG = {
    "level": LOG_LEVEL,
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
    
    # Log files
    "main_log": str(LOGS_DIR / "transcrevai.log"),
    "performance_log": str(LOGS_DIR / "performance.log"),
    "resource_log": str(LOGS_DIR / "resources.log"), 
    "error_log": str(LOGS_DIR / "errors.log"),
    
    # Rotation settings
    "max_file_size_mb": 10,  # 10MB log rotation
    "backup_count": 5,  # Keep 5 backup files
    "rotation_interval": "midnight",
}

# ============================================================================
# COMPLIANCE & QUALITY RULES
# ============================================================================

# Quality assurance thresholds
QUALITY_CONFIG = {
    "min_transcription_confidence": 0.7,  # Minimum transcription confidence
    "min_diarization_confidence": 0.6,  # Minimum speaker detection confidence
    "max_processing_time_multiplier": 2.0,  # 2x audio length max processing
    
    # Accuracy targets
    "target_transcription_accuracy": 0.95,  # 95% transcription accuracy
    "target_diarization_accuracy": 0.90,  # 90% speaker detection accuracy
    "target_timestamp_precision_ms": 100,  # ±100ms timestamp precision
}

# Browser-safe compliance rules
BROWSER_SAFETY_CONFIG = {
    "max_continuous_processing_ms": 100,  # Yield control every 100ms
    "progressive_loading": True,  # Enable progressive loading
    "memory_pressure_monitoring": True,  # Monitor memory usage
    "emergency_cleanup": True,  # Enable emergency memory cleanup
    
    # WebSocket safety
    "max_message_size_mb": 10,  # 10MB max WebSocket message
    "ping_timeout_seconds": 30,  # WebSocket ping timeout
    "max_concurrent_connections": 10,  # Limit concurrent connections
}

# ============================================================================
# DEVELOPMENT & DEBUGGING
# ============================================================================

# Development settings
DEV_CONFIG = {
    "enable_debug_logs": DEBUG_MODE,
    "enable_performance_tracking": True,
    "enable_memory_profiling": DEBUG_MODE,
    "save_intermediate_files": DEBUG_MODE,
    
    # Testing settings
    "test_audio_file": str(DATA_DIR / "test_audio.wav"),
    "test_output_dir": str(DATA_DIR / "test_output"),
    "enable_mock_processing": False,  # For UI testing
}

# ============================================================================
# EXPORT CONFIGURATION DICT
# ============================================================================

# Main configuration export
CONFIG = {
    "project": {
        "name": "TranscrevAI Optimized",
        "version": "1.0.0",
        "description": "PT-BR Exclusive Transcription & Diarization System",
        "language": "pt-BR",
    },
    
    "paths": {
        "project_root": str(PROJECT_ROOT),
        "data_dir": str(DATA_DIR),
        "logs_dir": str(LOGS_DIR),
        "cache_dir": str(CACHE_DIR),
        "models_dir": str(MODELS_DIR),
        "recordings_dir": str(RECORDINGS_DIR),
        "temp_dir": str(TEMP_DIR),
        "output_dir": str(OUTPUT_DIR),
        "templates_dir": str(TEMPLATES_DIR),
    },
    
    "server": {
        "host": SERVER_HOST,
        "port": SERVER_PORT,
        "debug": DEBUG_MODE,
        "websocket": {
            "ping_interval": WEBSOCKET_PING_INTERVAL,
            "ping_timeout": WEBSOCKET_PING_TIMEOUT,
            "max_size": WEBSOCKET_MAX_SIZE,
        }
    },
    
    "hardware": {
        "cpu_cores": CPU_CORES,
        "max_memory_mb": MAX_MEMORY_MB,
        "enable_cache": ENABLE_CACHE,
        "memory_thresholds": {
            "warning": MEMORY_WARNING_THRESHOLD,
            "emergency": MEMORY_EMERGENCY_THRESHOLD,
            "critical": MEMORY_CRITICAL_THRESHOLD,
            "streaming": STREAMING_MODE_THRESHOLD,
        }
    },
    
    "performance": {
        "targets": {
            "processing_ratio_warm": TARGET_PROCESSING_RATIO_WARM,
            "processing_ratio_cold": TARGET_PROCESSING_RATIO_COLD,
            "max_startup_time": MAX_STARTUP_TIME_SECONDS,
        }
    },
    
    "whisper": WHISPER_CONFIG,
    "diarization": DIARIZATION_CONFIG,
    "audio": AUDIO_CONFIG,
    "srt": SRT_CONFIG,
    "cache": CACHE_CONFIG,
    "cleanup": CLEANUP_CONFIG,
    "logging": LOGGING_CONFIG,
    "quality": QUALITY_CONFIG,
    "browser_safety": BROWSER_SAFETY_CONFIG,
    "development": DEV_CONFIG,
}

# PT-BR specific exports
PT_BR_CONFIG = {
    "corrections": PT_BR_CORRECTIONS,
    "prompts": PT_BR_INITIAL_PROMPTS,
    "model_name": WHISPER_MODEL_NAME,
    "language": WHISPER_LANGUAGE,
}