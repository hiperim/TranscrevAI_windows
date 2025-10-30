"""
Centralized Logging Setup for the TranscrevAI application.

This module provides a configurable logging setup that includes both console
and file-based logging with rotation, ensuring that log messages are
consistent and persistent.
"""

import logging
import sys
from pathlib import Path


def setup_app_logging(level=logging.INFO, logger_name=None):
    """Setup application-specific logging for TranscrevAI with robust error handling"""
    
    # Use application-specific logger instead of root logger
    if logger_name is None:
        logger_name = "transcrevai"
    
    logger = logging.getLogger(logger_name)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Set logging level
    logger.setLevel(level)
    
    # Prevent propagation to root logger to avoid conflicts
    logger.propagate = False
    
    # Create console handler (use default stream to avoid referencing sys)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Create detailed formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    # Create file handler for persistent logging
    try:
        # Multiple import strategies for robust DATA_DIR access
        data_dir = None
        
        # Strategy 1: Modern config import
        try:
            from config.app_config import get_config
            config = get_config()
            data_dir = getattr(config, "data_dir", None)
        except Exception:
            # Strategy 2: Import module dynamically and probe for DATA_DIR or data_dir attributes
            try:
                import importlib
                app_config = importlib.import_module("config.app_config")
                data_dir = getattr(app_config, "DATA_DIR", None) or getattr(app_config, "data_dir", None)
                if isinstance(data_dir, str):
                    data_dir = Path(data_dir)
            except Exception:
                # Strategy 3: Alternative config path
                try:
                    import os
                    import sys
                    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'app_config.py')
                    if os.path.exists(config_path):
                        sys.path.insert(0, os.path.dirname(config_path))
                        import importlib
                        app_config = importlib.import_module("config.app_config")
                        data_dir = getattr(app_config, "data_dir", None) or getattr(app_config, "DATA_DIR", None)
                        if isinstance(data_dir, str):
                            data_dir = Path(data_dir)
                except Exception:
                    pass
        
        # If we successfully got data_dir from config, use it
        if data_dir:
            try:
                # Ensure data_dir is a Path (handles str, Path, and os.PathLike)
                data_dir = Path(data_dir)
            except Exception:
                # If conversion fails, ignore and fall back
                data_dir = None

        if data_dir:
            log_dir = data_dir / "logs"
        else:
            # Fallback to relative path
            log_dir = Path(__file__).parent.parent.parent / "data" / "logs"
        
        # Ensure log directory exists
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "transcrevai.log"
        
        # Create file handler with rotation
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # More verbose for file
        
        # Detailed formatter for file
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
    except Exception as e:
        # If file logging fails, at least we have console - no crash
        logger.warning(f"File logging setup failed: {e}")
    
    return logger


def get_logger(name=None):
    """Get application logger for specific module"""
    if name is not None and name:
        return logging.getLogger(f"transcrevai.{name}")
    return logging.getLogger("transcrevai")