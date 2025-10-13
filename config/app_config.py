# config/app_config.py
"""
Enhanced Application Configuration for TranscrevAI
Production-ready configuration management with environment support
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)

@dataclass
class AppConfig:
    """Application configuration with validation and defaults - portable paths"""

    # === CORE DIRECTORIES (portable, relative to project root) ===
    base_dir: Path = field(init=False)
    src_dir: Path = field(init=False)
    data_dir: Path = field(init=False)
    recordings_dir: Path = field(init=False)
    temp_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    changes_dir: Path = field(init=False)
    
    # === MODEL CONFIGURATION ===
    model_name: str = "medium"
    model_language: str = "pt"
    device: str = "cpu"
    compute_type: str = "int8"
    
    # === PERFORMANCE SETTINGS ===
    max_memory_gb: float = 2.0
    max_workers: int = 4
    processing_timeout: float = 300.0  # 5 minutes
    
    # === WEBSOCKET SETTINGS ===
    websocket_timeout: float = 30.0
    max_message_size: int = 16 * 1024 * 1024  # 16MB
    heartbeat_interval: float = 10.0
    
    # === AUDIO SETTINGS ===
    sample_rate: int = 16000
    chunk_size: int = 1024
    max_audio_duration: float = 3600.0  # 1 hour
    
    # === LOGGING SETTINGS ===
    log_level: str = "INFO"
    log_max_bytes: int = 10 * 1024 * 1024  # 10MB
    log_backup_count: int = 5
    
    # === DEVELOPMENT SETTINGS ===
    debug_mode: bool = False
    force_cpu: bool = False
    enable_performance_monitoring: bool = True
    enable_memory_profiling: bool = False
    
    def __post_init__(self):
        # Initialize portable paths (relative to project root)
        self.base_dir = Path(__file__).parent.parent.resolve()
        self.src_dir = self.base_dir / "src"
        self.data_dir = self.base_dir / "data"
        self.recordings_dir = self.data_dir / "recordings"
        self.temp_dir = self.base_dir / "temp"
        self.logs_dir = self.base_dir / "logs"
        self.models_dir = self.base_dir / "models"
        self.changes_dir = self.base_dir / ".claude" / "CHANGES_MADE"

        self._validate_and_create_directories()
        self._load_environment_overrides()
        self._validate_settings()
        
    def _validate_and_create_directories(self):
        directories = [self.data_dir, self.recordings_dir, self.temp_dir, self.logs_dir, self.models_dir, self.changes_dir]
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"Failed to create directory {directory}: {e}")
                raise
                
    def _load_environment_overrides(self):
        env_mappings = {
            'TRANSCREVAI_MODEL_NAME': 'model_name',
            'TRANSCREVAI_DEVICE': 'device',
            'TRANSCREVAI_MAX_MEMORY': 'max_memory_gb',
            'TRANSCREVAI_LOG_LEVEL': 'log_level',
            'TRANSCREVAI_DEBUG': 'debug_mode'
        }
        for env_key, attr_name in env_mappings.items():
            env_value = os.getenv(env_key)
            if env_value:
                try:
                    if attr_name in ['max_memory_gb']:
                        setattr(self, attr_name, float(env_value))
                    elif attr_name in ['debug_mode']:
                        setattr(self, attr_name, env_value.lower() in ['true', '1', 'yes'])
                    else:
                        setattr(self, attr_name, env_value)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid environment value for {env_key}: {env_value} ({e})")
                    
    def _validate_settings(self):
        valid_models = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
        if self.model_name not in valid_models:
            logger.warning(f"Model {self.model_name} not in validated list")
        if self.device not in ["cpu", "cuda", "auto"]:
            logger.warning(f"Device {self.device} not in supported list")

app_config = AppConfig()

def get_config() -> AppConfig:
    return app_config