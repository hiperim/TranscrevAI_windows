"""
TranscrevAI Optimized - Configuration Module
Sistema de configuração PT-BR exclusivo com otimizações para hardware mínimo
"""

import os
import platform
import psutil
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("transcrevai.config")


def detect_system_specs() -> Dict[str, Any]:
    """Detect system specifications automatically"""
    try:
        cpu_count = psutil.cpu_count(logical=True)
        cpu_cores = psutil.cpu_count(logical=False)
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        return {
            "cpu_count": cpu_count,
            "cpu_cores": cpu_cores, 
            "memory_gb": round(memory_gb, 1),
            "platform": platform.system(),
            "architecture": platform.machine()
        }
    except Exception as e:
        logger.warning(f"Failed to detect system specs: {e}")
        return {
            "cpu_count": 4,
            "cpu_cores": 4,
            "memory_gb": 8.0,
            "platform": "Unknown",
            "architecture": "Unknown"
        }


def get_optimal_settings(system_specs: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate optimal settings based on system specifications"""
    cpu_cores = system_specs["cpu_cores"]
    memory_gb = system_specs["memory_gb"]
    
    # Conservative settings for browser-safe operation
    max_workers = min(4, max(1, cpu_cores - 1))  # Leave 1 core for UI
    memory_per_worker_mb = min(512, max(256, int(memory_gb * 1024 * 0.15)))  # 15% of total memory
    
    return {
        "max_workers": max_workers,
        "memory_per_worker_mb": memory_per_worker_mb,
        "enable_multiprocessing": cpu_cores >= 4,
        "enable_model_cache": memory_gb >= 4.0
    }


# Detect system specifications
SYSTEM_SPECS = detect_system_specs()
OPTIMAL_SETTINGS = get_optimal_settings(SYSTEM_SPECS)

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
CACHE_DIR = BASE_DIR / "cache"
TEMP_DIR = BASE_DIR / "temp"

# Environment variables with defaults
def get_env_bool(key: str, default: bool) -> bool:
    """Get boolean environment variable"""
    return os.getenv(key, str(default)).lower() in ('true', '1', 'yes', 'on')

def get_env_int(key: str, default: int) -> int:
    """Get integer environment variable"""
    try:
        return int(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default

def get_env_float(key: str, default: float) -> float:
    """Get float environment variable"""
    try:
        return float(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


# ===== MAIN CONFIGURATION =====
CONFIG = {
    # ===== APPLICATION SETTINGS =====
    "app": {
        "name": "TranscrevAI Optimized",
        "version": "1.0.0",
        "description": "Sistema profissional de transcrição PT-BR",
        "host": os.getenv("TRANSCREVAI_HOST", "0.0.0.0"),
        "port": get_env_int("TRANSCREVAI_PORT", 8001),
        "debug": get_env_bool("TRANSCREVAI_DEBUG", False),
        "reload": get_env_bool("TRANSCREVAI_RELOAD", False),
        "workers": get_env_int("TRANSCREVAI_WORKERS", 1),
        "max_request_size": get_env_int("TRANSCREVAI_MAX_REQUEST_MB", 100) * 1024 * 1024,
    },
    
    # ===== HARDWARE CONFIGURATION =====
    "hardware": {
        "cpu_cores": SYSTEM_SPECS["cpu_cores"],
        "cpu_count": SYSTEM_SPECS["cpu_count"],
        "memory_total_gb": SYSTEM_SPECS["memory_gb"],
        "memory_per_worker_mb": OPTIMAL_SETTINGS["memory_per_worker_mb"],
        "max_workers": OPTIMAL_SETTINGS["max_workers"],
        "enable_multiprocessing": OPTIMAL_SETTINGS["enable_multiprocessing"],
        "platform": SYSTEM_SPECS["platform"],
        "architecture": SYSTEM_SPECS["architecture"],
    },
    
    # ===== PATHS CONFIGURATION =====
    "paths": {
        "base_dir": str(BASE_DIR),
        "data_dir": str(DATA_DIR),
        "temp_dir": str(TEMP_DIR),
        "output_dir": str(DATA_DIR / "output"),
        "recordings_dir": str(DATA_DIR / "recordings"),
        "cache_dir": str(CACHE_DIR),
        "models_cache_dir": str(CACHE_DIR / "models"),
        "logs_dir": str(LOGS_DIR),
    },
    
    # ===== AUDIO PROCESSING =====
    "audio": {
        "sample_rate": get_env_int("TRANSCREVAI_SAMPLE_RATE", 16000),  # Optimal for Whisper
        "channels": 1,  # Mono for transcription
        "chunk_duration": get_env_float("TRANSCREVAI_CHUNK_DURATION", 30.0),  # 30 second chunks
        "overlap_duration": get_env_float("TRANSCREVAI_OVERLAP_DURATION", 1.0),  # 1 second overlap
        "max_file_size_mb": get_env_int("TRANSCREVAI_MAX_FILE_MB", 500),
        "max_duration_minutes": get_env_int("TRANSCREVAI_MAX_DURATION_MIN", 120),  # 2 hours max
        "supported_formats": [".wav", ".mp3", ".mp4", ".m4a", ".flac", ".ogg"],
        "recording_format": "wav",
        "bitrate": get_env_int("TRANSCREVAI_BITRATE", 192000),  # 192 kbps
    },
    
    # ===== WHISPER MODEL CONFIGURATION (PT-BR OPTIMIZED) =====
    "whisper": {
        "model_size": "medium",  # Fixed for PT-BR optimization
        "language": "pt",        # Fixed for Portuguese
        "task": "transcribe",    # Always transcribe (not translate)
        "temperature": [0.0, 0.2, 0.4],  # Temperature fallback strategy
        "best_of": 3,           # Multiple attempts for best result
        "beam_size": 5,         # Beam search size
        "patience": 1.0,        # Patience for beam search
        "length_penalty": 1.0,  # Length penalty
        "compression_ratio_threshold": 2.4,
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": True,  # Context awareness
        "initial_prompt": "Este é um áudio em português brasileiro. Por favor, transcreva com precisão, incluindo pontuação adequada.",  # PT-BR prompt
        "suppress_tokens": [-1],  # Don't suppress any tokens
        "fp16": True,           # Use FP16 for faster processing
    },
    
    # ===== DIARIZATION CONFIGURATION =====
    "diarization": {
        "enable_diarization": get_env_bool("TRANSCREVAI_ENABLE_DIARIZATION", True),
        "min_speakers": get_env_int("TRANSCREVAI_MIN_SPEAKERS", 1),
        "max_speakers": get_env_int("TRANSCREVAI_MAX_SPEAKERS", 8),
        "default_speakers": 2,
        "min_segment_duration": get_env_float("TRANSCREVAI_MIN_SEGMENT_DURATION", 0.5),  # 0.5 seconds
        "segment_merge_threshold": get_env_float("TRANSCREVAI_MERGE_THRESHOLD", 1.0),  # 1 second
        "overlap_threshold": get_env_float("TRANSCREVAI_OVERLAP_THRESHOLD", 0.3),
        "enable_overlap_detection": get_env_bool("TRANSCREVAI_ENABLE_OVERLAP", True),
        "merge_close_segments": get_env_bool("TRANSCREVAI_MERGE_SEGMENTS", True),
        "mid_window": 2.0,      # Window size for feature extraction
        "step": 0.5,            # Step size for sliding window
    },
    
    # ===== SUBTITLE GENERATION =====
    "subtitles": {
        "format": "srt",        # Default format
        "include_speaker_labels": get_env_bool("TRANSCREVAI_INCLUDE_SPEAKERS", True),
        "max_chars_per_line": get_env_int("TRANSCREVAI_MAX_CHARS_LINE", 60),
        "max_lines_per_subtitle": get_env_int("TRANSCREVAI_MAX_LINES", 2),
        "min_subtitle_duration": get_env_float("TRANSCREVAI_MIN_SUBTITLE_DURATION", 0.8),
        "max_subtitle_duration": get_env_float("TRANSCREVAI_MAX_SUBTITLE_DURATION", 7.0),
        "subtitle_gap": get_env_float("TRANSCREVAI_SUBTITLE_GAP", 0.2),  # Gap between subtitles
        "overlap_threshold": get_env_float("TRANSCREVAI_SUBTITLE_OVERLAP_THRESHOLD", 0.3),
        "proximity_threshold": get_env_float("TRANSCREVAI_SUBTITLE_PROXIMITY_THRESHOLD", 0.5),
        "merge_consecutive_speakers": get_env_bool("TRANSCREVAI_MERGE_CONSECUTIVE", True),
        "max_merge_gap": get_env_float("TRANSCREVAI_MAX_MERGE_GAP", 2.0),
        "min_segment_duration": get_env_float("TRANSCREVAI_MIN_SEGMENT_DURATION", 0.3),
        "min_text_length": get_env_int("TRANSCREVAI_MIN_TEXT_LENGTH", 3),
    },
    
    # ===== RESOURCE MANAGEMENT =====
    "resource_management": {
        "memory_limit_mb": get_env_int("TRANSCREVAI_MAX_MEMORY_MB", int(SYSTEM_SPECS["memory_gb"] * 1024 * 0.8)),  # 80% of total
        "memory_warning_threshold": get_env_float("TRANSCREVAI_MEMORY_WARNING_THRESHOLD", 0.75),  # 75%
        "memory_emergency_threshold": get_env_float("TRANSCREVAI_MEMORY_EMERGENCY_THRESHOLD", 0.85),  # 85%
        "cpu_limit_percent": get_env_float("TRANSCREVAI_CPU_LIMIT", 80.0),  # 80% CPU max
        "enable_memory_monitoring": get_env_bool("TRANSCREVAI_ENABLE_MEMORY_MONITORING", True),
        "monitoring_interval": get_env_float("TRANSCREVAI_MONITORING_INTERVAL", 5.0),  # 5 seconds
        "cleanup_interval": get_env_float("TRANSCREVAI_CLEANUP_INTERVAL", 30.0),  # 30 seconds
        "enable_gc_optimization": get_env_bool("TRANSCREVAI_ENABLE_GC_OPTIMIZATION", True),
        "max_cache_size_mb": get_env_int("TRANSCREVAI_MAX_CACHE_MB", min(2048, int(SYSTEM_SPECS["memory_gb"] * 1024 * 0.3))),  # 30% of memory
    },
    
    # ===== MODEL CACHE CONFIGURATION =====
    "model_cache": {
        "enable_cache": OPTIMAL_SETTINGS["enable_model_cache"],
        "cache_ttl_hours": get_env_int("TRANSCREVAI_CACHE_TTL_HOURS", 24),  # 24 hours
        "max_cached_models": get_env_int("TRANSCREVAI_MAX_CACHED_MODELS", 3),
        "preload_model": get_env_bool("TRANSCREVAI_PRELOAD_MODEL", True),
        "lazy_loading": get_env_bool("TRANSCREVAI_LAZY_LOADING", True),
        "memory_pressure_unload": get_env_bool("TRANSCREVAI_MEMORY_PRESSURE_UNLOAD", True),
        "warmup_on_startup": get_env_bool("TRANSCREVAI_WARMUP_ON_STARTUP", False),  # Disabled for faster startup
    },
    
    # ===== PERFORMANCE TARGETS =====
    "performance": {
        "targets": {
            "processing_ratio_warm": get_env_float("TRANSCREVAI_PROCESSING_RATIO_WARM", 0.65),  # 0.65x for warm starts
            "processing_ratio_cold": get_env_float("TRANSCREVAI_PROCESSING_RATIO_COLD", 1.2),   # 1.2x for cold starts
            "startup_time_target": get_env_float("TRANSCREVAI_STARTUP_TIME_TARGET", 5.0),       # 5 seconds
            "memory_usage_target_mb": get_env_int("TRANSCREVAI_MEMORY_TARGET_MB", 1536),        # 1.5GB target
        },
        "browser_safe": {
            "max_blocking_time_ms": get_env_int("TRANSCREVAI_MAX_BLOCKING_MS", 50),  # 50ms max blocking
            "yield_interval_ms": get_env_int("TRANSCREVAI_YIELD_INTERVAL_MS", 10),   # 10ms yield time
            "progress_update_interval": get_env_float("TRANSCREVAI_PROGRESS_INTERVAL", 0.5),    # 0.5 seconds
            "chunk_processing_timeout": get_env_int("TRANSCREVAI_CHUNK_TIMEOUT", 60),           # 60 seconds per chunk
        },
    },
    
    # ===== LOGGING CONFIGURATION =====
    "logging": {
        "level": os.getenv("TRANSCREVAI_LOG_LEVEL", "INFO"),
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file_max_bytes": get_env_int("TRANSCREVAI_LOG_MAX_BYTES", 10 * 1024 * 1024),  # 10MB
        "file_backup_count": get_env_int("TRANSCREVAI_LOG_BACKUP_COUNT", 5),
        "enable_console_logging": get_env_bool("TRANSCREVAI_CONSOLE_LOGGING", True),
        "enable_file_logging": get_env_bool("TRANSCREVAI_FILE_LOGGING", True),
        "enable_performance_logging": get_env_bool("TRANSCREVAI_PERFORMANCE_LOGGING", True),
        "enable_resource_logging": get_env_bool("TRANSCREVAI_RESOURCE_LOGGING", True),
    },
    
    # ===== PROGRESSIVE LOADING =====
    "progressive_loading": {
        "enable": get_env_bool("TRANSCREVAI_PROGRESSIVE_LOADING", True),
        "phases": ["essential", "models", "processing", "optimization"],
        "essential_timeout": get_env_float("TRANSCREVAI_ESSENTIAL_TIMEOUT", 10.0),
        "models_timeout": get_env_float("TRANSCREVAI_MODELS_TIMEOUT", 60.0),
        "processing_timeout": get_env_float("TRANSCREVAI_PROCESSING_TIMEOUT", 30.0),
        "optimization_timeout": get_env_float("TRANSCREVAI_OPTIMIZATION_TIMEOUT", 20.0),
    },
    
    # ===== WEBSOCKET CONFIGURATION =====
    "websocket": {
        "ping_interval": get_env_int("TRANSCREVAI_WS_PING_INTERVAL", 20),       # 20 seconds
        "ping_timeout": get_env_int("TRANSCREVAI_WS_PING_TIMEOUT", 10),         # 10 seconds  
        "close_timeout": get_env_int("TRANSCREVAI_WS_CLOSE_TIMEOUT", 5),        # 5 seconds
        "max_size": get_env_int("TRANSCREVAI_WS_MAX_SIZE", 16 * 1024 * 1024),   # 16MB
        "max_queue": get_env_int("TRANSCREVAI_WS_MAX_QUEUE", 32),               # 32 messages
    },
    
    # ===== DEVELOPMENT & DEBUG =====
    "development": {
        "enable_debug_endpoints": get_env_bool("TRANSCREVAI_DEBUG_ENDPOINTS", False),
        "enable_metrics": get_env_bool("TRANSCREVAI_ENABLE_METRICS", True),
        "enable_health_checks": get_env_bool("TRANSCREVAI_ENABLE_HEALTH_CHECKS", True),
        "cors_allow_origins": ["http://localhost:3000", "http://localhost:8080", "http://127.0.0.1:8001"],
        "cors_allow_credentials": True,
        "cors_allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "cors_allow_headers": ["*"],
    }
}

# ===== PT-BR SPECIFIC OPTIMIZATIONS =====
PT_BR_CONFIG = {
    "corrections": {
        # Common transcription corrections for Portuguese
        "voce": "você",
        "nao": "não", 
        "tambem": "também",
        "so": "só",
        "ja": "já",
        "la": "lá",
        "ca": "cá",
        "pos": "pós",
        "pre": "pré",
        "pro": "pró",
        "anti": "anti",
        
        # Contractions
        "pra": "para",
        "pro": "para o",
        "pras": "para as", 
        "pros": "para os",
        "numa": "em uma",
        "numas": "em umas",
        "nuns": "em uns",
        "dum": "de um",
        "duma": "de uma",
        "dumas": "de umas",
        
        # Question patterns
        "onde que": "onde",
        "como que": "como", 
        "quando que": "quando",
        "porque que": "por que",
        "pra que": "para que",
        
        # Common expressions
        "ta bom": "tá bom",
        "ta certo": "tá certo",
        "beleza entao": "beleza, então",
        "ai meu deus": "ai, meu Deus",
        "nossa senhora": "nossa Senhora",
        "valeu cara": "valeu, cara",
    },
    
    "contextual_patterns": {
        # Regex patterns for contextual corrections
        "contractions": [
            (r'\bpra\b', 'para'),
            (r'\bpro\b', 'para o'),
            (r'\bvoce\b', 'você'),
            (r'\bnao\b', 'não'),
        ],
        
        "questions": [
            (r'\bonde esta\b', 'onde está'),
            (r'\bcomo esta\b', 'como está'), 
            (r'\bquem e\b', 'quem é'),
            (r'\bo que e\b', 'o que é'),
        ],
        
        "capitalization": [
            (r'\bbrasil\b', 'Brasil'),
            (r'\bsao paulo\b', 'São Paulo'),
            (r'\brio de janeiro\b', 'Rio de Janeiro'),
        ],
    },
    
    "whisper_optimizations": {
        "initial_prompt": (
            "Este é um áudio em português brasileiro. "
            "Por favor, transcreva com precisão, incluindo pontuação adequada, "
            "acentos corretos e tratamento formal/informal apropriado."
        ),
        "temperature_strategy": "fallback",  # Use fallback temperatures
        "suppress_hallucinations": True,
        "enhance_punctuation": True,
        "preserve_brazilian_expressions": True,
    }
}

# ===== VALIDATION AND INITIALIZATION =====
def validate_config() -> bool:
    """Validate configuration settings"""
    try:
        # Validate paths
        for path_key, path_value in CONFIG["paths"].items():
            path_obj = Path(path_value)
            if not path_obj.exists():
                path_obj.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {path_value}")
        
        # Validate hardware limits
        if CONFIG["hardware"]["memory_per_worker_mb"] > CONFIG["hardware"]["memory_total_gb"] * 1024 * 0.5:
            logger.warning("Memory per worker is high, reducing to safe level")
            CONFIG["hardware"]["memory_per_worker_mb"] = int(CONFIG["hardware"]["memory_total_gb"] * 1024 * 0.3)
        
        # Validate audio settings
        if CONFIG["audio"]["sample_rate"] not in [8000, 16000, 22050, 44100, 48000]:
            logger.warning(f"Unusual sample rate: {CONFIG['audio']['sample_rate']}")
        
        # Validate temperature settings
        temperatures = CONFIG["whisper"]["temperature"]
        if not isinstance(temperatures, list) or len(temperatures) < 1:
            logger.error("Invalid temperature configuration")
            return False
        
        logger.info("Configuration validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False


def get_config_summary() -> Dict[str, Any]:
    """Get configuration summary for debugging"""
    return {
        "system": SYSTEM_SPECS,
        "optimal_settings": OPTIMAL_SETTINGS,
        "audio_settings": CONFIG["audio"],
        "whisper_model": CONFIG["whisper"]["model_size"],
        "language": CONFIG["whisper"]["language"],
        "diarization_enabled": CONFIG["diarization"]["enable_diarization"],
        "cache_enabled": CONFIG["model_cache"]["enable_cache"],
        "progressive_loading": CONFIG["progressive_loading"]["enable"],
        "memory_limit_mb": CONFIG["resource_management"]["memory_limit_mb"],
        "max_workers": CONFIG["hardware"]["max_workers"],
    }


# Initialize and validate configuration on import
if not validate_config():
    logger.error("Configuration validation failed!")
    raise ValueError("Invalid configuration detected")

# Export main configuration objects
__all__ = ["CONFIG", "PT_BR_CONFIG", "SYSTEM_SPECS", "OPTIMAL_SETTINGS", "get_config_summary"]

# Log configuration summary
logger.info("TranscrevAI Optimized Configuration Loaded")
logger.info(f"System: {SYSTEM_SPECS['cpu_cores']} cores, {SYSTEM_SPECS['memory_gb']:.1f}GB RAM")
logger.info(f"Settings: {OPTIMAL_SETTINGS['max_workers']} workers, {OPTIMAL_SETTINGS['memory_per_worker_mb']}MB/worker")
logger.info(f"Model: {CONFIG['whisper']['model_size']} ({CONFIG['whisper']['language']})")
logger.info(f"Cache: {'enabled' if CONFIG['model_cache']['enable_cache'] else 'disabled'}")