"""
Enhanced App Configuration - Fixed Historical Comments and Dynamic Settings
Production-ready configuration with proper PT-BR optimization and compliance

Fixes applied:
- Cleaned up historical comments (FASE 10, SPRINT 3)
- Made whisper_beam_size and whisper_patience configurable instead of fixed
- Fixed DIARIZATION_CONFIG.min_speakers validation
- Removed outdated WHISPER_MODEL_PATH comments
- Cleaned up REVERTED comments in analysis_thresholds
- Added compliance-focused configuration options
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Production-ready logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# ====== CORE PATHS AND DIRECTORIES ======

# Determine base application directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories (cross-platform)
DATA_DIR = BASE_DIR / "data"
TEMP_DIR = DATA_DIR / "temp"
MODELS_DIR = DATA_DIR / "models"
INPUTS_DIR = DATA_DIR / "inputs"
OUTPUTS_DIR = DATA_DIR / "outputs"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
RECORDINGS_DIR = DATA_DIR / "recordings"

# Configuration directory
CONFIG_DIR = BASE_DIR / "config"

def _ensure_directories_created():
    """Ensure all required directories exist"""
    directories = [
        DATA_DIR, TEMP_DIR, MODELS_DIR, INPUTS_DIR, 
        OUTPUTS_DIR, TRANSCRIPTS_DIR, RECORDINGS_DIR, CONFIG_DIR
    ]
    
    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create directory {directory}: {e}")

# Create directories on import
_ensure_directories_created()

# ====== MODEL CONFIGURATION ======

# ENHANCED: Dynamic model configuration (was fixed values)
MODEL_CONFIG = {
    # Primary model settings
    "model_name": "medium",  # PT-BR optimized medium model
    "device": "cpu",         # CPU-only for compliance
    "compute_type": "int8",  # Optimized for CPU performance
    
    # FIXED: Dynamic whisper parameters instead of fixed values
    "whisper_beam_size": 1,      # Default, can be dynamically adjusted
    "whisper_patience": None,    # None for maximum speed (correct configuration)
    "whisper_temperature": 0.0,  # Deterministic for stability
    
    # Model management
    "cache_timeout": 1800,       # 30 minutes model cache
    "preload_model": True,       # Preload for faster startup
    "model_download_timeout": 600, # 10 minutes download timeout
    
    # Memory management for compliance
    "max_memory_usage_mb": 2048, # 2GB memory limit
    "memory_cleanup_threshold": 0.8, # Cleanup at 80% usage
    
    # Performance targets (compliance requirements)
    "target_processing_ratio": 0.5, # 0.5s processing per 1s audio
    "target_accuracy": 0.90,         # 90%+ accuracy for PT-BR
}

# Language-specific optimization
LANGUAGE_CONFIG = {
    "primary_language": "pt",    # Portuguese Brazilian only
    "supported_languages": ["pt"], # PT-BR exclusive compliance
    "language_detection": False,  # Disabled for performance
    "auto_language": False,       # Fixed to PT-BR
}

# ====== PROCESSING CONFIGURATION ======

# ENHANCED: Improved processing configuration with dynamic settings
PROCESSING_CONFIG = {
    # Audio processing
    "sample_rate": 16000,
    "chunk_size": 1024,
    "audio_format": "wav",
    "max_audio_duration": 3600,  # 1 hour max
    
    # Transcription settings (ENHANCED from fixed values)
    "whisper_beam_size": lambda complexity: min(3, max(1, {
        "low": 1, "medium": 1, "high": 2
    }.get(complexity, 1))),  # Dynamic based on audio complexity
    
    "whisper_patience": lambda speed_priority: None if speed_priority else 1.0,
    
    "whisper_temperature": lambda accuracy_priority: 0.0 if accuracy_priority else 0.1,
    
    # Quality settings
    "confidence_threshold": 0.7,
    "quality_filter": True,
    "noise_reduction": True,
    
    # Performance settings
    "parallel_processing": True,
    "max_concurrent_sessions": 4,
    "processing_timeout": 600,  # 10 minutes timeout
}

# ====== DIARIZATION CONFIGURATION ======

# ENHANCED: Fixed diarization configuration with proper validation
DIARIZATION_CONFIG = {
    # Speaker detection
    "min_speakers": 1,           # FIXED: Proper minimum (was problematic)
    "max_speakers": 6,           # Reasonable maximum for most use cases
    "auto_detect_speakers": True, # Dynamic speaker detection
    
    # Method selection (CPU-only MFCC + Prosodic)
    "method": "mfcc_prosodic",   # Production CPU-only method
    "fallback_method": "clustering",
    "simple_method_threshold": 10.0, # Use simple method for <10s audio
    
    # Confidence and quality
    "confidence_threshold": 0.5,
    "min_segment_duration": 0.5, # Minimum 500ms segments
    "merge_threshold": 0.3,       # Merge segments <300ms apart
    
    # FIXED: Cleaned up analysis thresholds (removed REVERTED comments)
    "analysis_thresholds": {
        "energy_threshold": 0.01,    # Voice activity detection
        "spectral_threshold": 500,   # Speaker complexity detection
        "silence_threshold": 0.2,    # Silence detection
        "change_threshold": 0.05     # Speaker change detection
    },
    
    # Performance optimization
    "chunk_processing": True,
    "adaptive_processing": True,
    "memory_efficient": True,
}

# ====== WEBSOCKET CONFIGURATION ======

# Enhanced WebSocket configuration for browser stability
WEBSOCKET_CONFIG = {
    # Connection settings
    "host": "127.0.0.1",
    "port": 8000,
    "max_connections": 10,
    
    # Message handling
    "max_message_size": 10 * 1024 * 1024,  # 10MB max message
    "heartbeat_interval": 30,                # 30s heartbeat
    "connection_timeout": 300,               # 5 minutes timeout
    
    # Progress updates for compliance (freeze prevention)
    "progress_update_interval": 0.5,  # Update every 500ms
    "detailed_progress": True,        # Detailed progress info
    "status_messages": True,          # User-friendly status messages
    
    # Browser compatibility
    "cors_enabled": True,
    "browser_buffer_size": 1024 * 1024,  # 1MB browser buffer
    "memory_safety_mode": True,           # Extra memory safety for browsers
}

# ====== COMPLIANCE CONFIGURATION ======

# TranscrevAI compliance requirements configuration
COMPLIANCE_CONFIG = {
    # Performance requirements
    "target_processing_speed": 0.5,      # 0.5s per 1s audio
    "max_memory_usage_mb": 3500,         # 3.5GB hard limit
    "target_memory_mb": 2048,            # 2GB target usage
    "min_accuracy_percent": 90,          # 90% accuracy minimum
    
    # Hardware requirements
    "min_cpu_cores": 4,
    "min_memory_gb": 8,
    "min_storage_gb": 5,
    
    # System requirements
    "windows_compatible": True,
    "cpu_only_mode": True,               # No GPU dependencies
    "websocket_stability": True,         # Must not freeze browsers
    "real_time_capable": True,           # Support real-time processing
    
    # Validation testing
    "reference_audio_path": RECORDINGS_DIR,
    "expected_results_pattern": "expected_results_{filename}.txt",
    "test_validation_required": True,
}

# ====== PRODUCTION CONFIGURATION ======

# Production environment settings
PRODUCTION_CONFIG = {
    # Environment
    "debug_mode": False,
    "log_level": "INFO",
    "performance_monitoring": True,
    "metrics_collection": True,
    
    # Security
    "secure_file_handling": True,
    "input_validation": True,
    "path_sanitization": True,
    
    # Resource management
    "automatic_cleanup": True,
    "memory_monitoring": True,
    "cpu_monitoring": True,
    "disk_space_monitoring": True,
    
    # Error handling
    "graceful_degradation": True,
    "fallback_methods": True,
    "error_recovery": True,
    "crash_prevention": True,
}

# ====== TESTING CONFIGURATION ======

# Testing and validation configuration
TESTING_CONFIG = {
    "test_audio_directory": str(RECORDINGS_DIR),
    "test_timeout_seconds": 300,        # 5 minutes per test
    "validation_accuracy_threshold": 0.85,
    "performance_test_enabled": True,
    
    # Reference files for validation
    "reference_files": {
        "t.speakers.wav": "expected_results_t.speakers.txt",
        "t2.speakers.wav": "expected_results_t2.speakers.txt", 
        "d.speakers.wav": "expected_results_d.speakers.txt",
        "q.speakers.wav": "expected_results_q.speakers.txt",
    },
    
    # Test categories
    "test_transcription": True,
    "test_diarization": True,
    "test_full_pipeline": True,
    "test_websocket": True,
    "test_memory_usage": True,
    "test_performance": True,
}

# ====== LOGGING CONFIGURATION ======

LOGGING_CONFIG = {
    "level": logging.INFO,
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_logging": True,
    "console_logging": True,
    "log_rotation": True,
    "max_log_size_mb": 50,
    "backup_count": 5,
    "log_directory": str(DATA_DIR / "logs")
}

# ====== UTILITY FUNCTIONS ======

def get_model_path(model_name: str = "medium") -> str:
    """
    Get path for model storage
    ENHANCED: Better path handling without outdated comments
    """
    model_path = MODELS_DIR / f"whisper-{model_name}"
    model_path.mkdir(parents=True, exist_ok=True)
    return str(model_path)

def get_temp_path() -> str:
    """Get temporary directory path"""
    return str(TEMP_DIR)

def get_processing_config(audio_duration: float = 30.0, complexity: str = "medium") -> Dict[str, Any]:
    """
    Get dynamic processing configuration based on audio characteristics
    ENHANCED: Dynamic configuration instead of fixed values
    """
    config = PROCESSING_CONFIG.copy()
    
    # Dynamic beam size based on complexity
    if callable(config["whisper_beam_size"]):
        config["whisper_beam_size"] = config["whisper_beam_size"](complexity)
    
    # Dynamic patience based on speed priority
    speed_priority = audio_duration < 30.0  # Prioritize speed for short audio
    if callable(config["whisper_patience"]):
        config["whisper_patience"] = config["whisper_patience"](speed_priority)
    
    # Dynamic temperature based on accuracy priority
    accuracy_priority = audio_duration > 60.0  # Prioritize accuracy for long audio
    if callable(config["whisper_temperature"]):
        config["whisper_temperature"] = config["whisper_temperature"](accuracy_priority)
    
    return config

def get_compliance_status() -> Dict[str, Any]:
    """Get current compliance status"""
    import psutil
    
    memory_usage = psutil.virtual_memory()
    available_memory_gb = memory_usage.available / (1024**3)
    cpu_count = psutil.cpu_count()
    
    return {
        "memory_compliant": available_memory_gb >= COMPLIANCE_CONFIG["min_memory_gb"],
        "cpu_compliant": cpu_count >= COMPLIANCE_CONFIG["min_cpu_cores"],
        "memory_available_gb": available_memory_gb,
        "cpu_cores_available": cpu_count,
        "system_ready": (
            available_memory_gb >= COMPLIANCE_CONFIG["min_memory_gb"] and
            cpu_count >= COMPLIANCE_CONFIG["min_cpu_cores"]
        )
    }

def validate_configuration() -> Dict[str, Any]:
    """Validate all configuration settings"""
    issues = []
    
    # Validate paths
    critical_dirs = [DATA_DIR, TEMP_DIR, MODELS_DIR]
    for directory in critical_dirs:
        if not directory.exists():
            issues.append(f"Critical directory missing: {directory}")
    
    # Validate model configuration
    if MODEL_CONFIG["model_name"] not in ["tiny", "base", "small", "medium", "large"]:
        issues.append(f"Invalid model name: {MODEL_CONFIG['model_name']}")
    
    # Validate memory limits
    if MODEL_CONFIG["max_memory_usage_mb"] > COMPLIANCE_CONFIG["max_memory_usage_mb"]:
        issues.append("Model memory limit exceeds compliance limit")
    
    # Validate processing configuration
    if DIARIZATION_CONFIG["min_speakers"] < 1:
        issues.append("min_speakers must be at least 1")
    
    if DIARIZATION_CONFIG["max_speakers"] < DIARIZATION_CONFIG["min_speakers"]:
        issues.append("max_speakers must be >= min_speakers")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "directories_ready": all(d.exists() for d in critical_dirs),
        "compliance_ready": get_compliance_status()["system_ready"]
    }

# ====== EXPORT CONFIGURATION ======

# Main configuration dictionary
CONFIG = {
    "model": MODEL_CONFIG,
    "language": LANGUAGE_CONFIG,
    "processing": PROCESSING_CONFIG,
    "diarization": DIARIZATION_CONFIG,
    "websocket": WEBSOCKET_CONFIG,
    "compliance": COMPLIANCE_CONFIG,
    "production": PRODUCTION_CONFIG,
    "testing": TESTING_CONFIG,
    "logging": LOGGING_CONFIG,
}

# Export commonly used configurations
__all__ = [
    "CONFIG",
    "MODEL_CONFIG",
    "PROCESSING_CONFIG", 
    "DIARIZATION_CONFIG",
    "WEBSOCKET_CONFIG",
    "COMPLIANCE_CONFIG",
    "DATA_DIR",
    "TEMP_DIR",
    "MODELS_DIR",
    "get_model_path",
    "get_temp_path",
    "get_processing_config",
    "get_compliance_status",
    "validate_configuration",
    "_ensure_directories_created"
]

# Initialize configuration validation on startup
if __name__ == "__main__":
    validation_result = validate_configuration()
    if validation_result["valid"]:
        print("✅ Configuration validation passed")
    else:
        print("❌ Configuration validation failed:")
        for issue in validation_result["issues"]:
            print(f"  - {issue}")
        sys.exit(1)