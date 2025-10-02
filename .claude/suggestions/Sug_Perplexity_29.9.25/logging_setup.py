"""
TranscrevAI Optimized - Logging Setup
Configuração centralizada de logging com múltiplos níveis e arquivos
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Dict, Optional

# Import configuration
try:
    from config import LOGGING_CONFIG, LOGS_DIR
except ImportError:
    # Fallback configuration if config.py not available
    LOGS_DIR = Path("logs")
    LOGS_DIR.mkdir(exist_ok=True)
    
    LOGGING_CONFIG = {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "date_format": "%Y-%m-%d %H:%M:%S",
        "main_log": str(LOGS_DIR / "transcrevai.log"),
        "performance_log": str(LOGS_DIR / "performance.log"),
        "resource_log": str(LOGS_DIR / "resources.log"),
        "error_log": str(LOGS_DIR / "errors.log"),
        "max_file_size_mb": 10,
        "backup_count": 5,
        "rotation_interval": "midnight",
    }


class TranscrevAILogger:
    """
    Centralized logging system for TranscrevAI
    Supports multiple log files with different purposes and rotation
    """
    
    _loggers: Dict[str, logging.Logger] = {}
    _handlers_setup: bool = False
    
    @classmethod
    def setup_logging(cls) -> None:
        """Setup all logging handlers and formatters"""
        if cls._handlers_setup:
            return
            
        # Ensure logs directory exists
        os.makedirs(LOGS_DIR, exist_ok=True)
        
        # Setup main logger
        cls._setup_main_logger()
        
        # Setup specialized loggers
        cls._setup_performance_logger()
        cls._setup_resource_logger()
        cls._setup_error_logger()
        
        # Configure third-party loggers
        cls._configure_third_party_loggers()
        
        cls._handlers_setup = True
        
        # Log initialization
        main_logger = cls.get_logger("transcrevai.logging")
        main_logger.info("Logging system initialized successfully")
        main_logger.info(f"Log files location: {LOGS_DIR}")
    
    @classmethod
    def _setup_main_logger(cls) -> None:
        """Setup main application logger"""
        logger = logging.getLogger("transcrevai")
        logger.setLevel(getattr(logging, LOGGING_CONFIG["level"]))
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            LOGGING_CONFIG["main_log"],
            maxBytes=LOGGING_CONFIG["max_file_size_mb"] * 1024 * 1024,
            backupCount=LOGGING_CONFIG["backup_count"],
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, LOGGING_CONFIG["level"]))
        file_formatter = logging.Formatter(
            LOGGING_CONFIG["format"],
            datefmt=LOGGING_CONFIG["date_format"]
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        cls._loggers["main"] = logger
    
    @classmethod
    def _setup_performance_logger(cls) -> None:
        """Setup performance monitoring logger"""
        logger = logging.getLogger("transcrevai.performance")
        logger.setLevel(logging.INFO)
        logger.propagate = False  # Don't propagate to parent logger
        
        # Performance file handler
        file_handler = logging.handlers.RotatingFileHandler(
            LOGGING_CONFIG["performance_log"],
            maxBytes=LOGGING_CONFIG["max_file_size_mb"] * 1024 * 1024,
            backupCount=LOGGING_CONFIG["backup_count"],
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        
        # Custom formatter for performance metrics
        performance_formatter = logging.Formatter(
            "%(asctime)s - PERF - %(message)s",
            datefmt=LOGGING_CONFIG["date_format"]
        )
        file_handler.setFormatter(performance_formatter)
        logger.addHandler(file_handler)
        
        cls._loggers["performance"] = logger
    
    @classmethod
    def _setup_resource_logger(cls) -> None:
        """Setup resource monitoring logger"""
        logger = logging.getLogger("transcrevai.resources")
        logger.setLevel(logging.INFO)
        logger.propagate = False  # Don't propagate to parent logger
        
        # Resource file handler
        file_handler = logging.handlers.RotatingFileHandler(
            LOGGING_CONFIG["resource_log"],
            maxBytes=LOGGING_CONFIG["max_file_size_mb"] * 1024 * 1024,
            backupCount=LOGGING_CONFIG["backup_count"],
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        
        # Custom formatter for resource metrics
        resource_formatter = logging.Formatter(
            "%(asctime)s - RES - %(message)s",
            datefmt=LOGGING_CONFIG["date_format"]
        )
        file_handler.setFormatter(resource_formatter)
        logger.addHandler(file_handler)
        
        cls._loggers["resources"] = logger
    
    @classmethod
    def _setup_error_logger(cls) -> None:
        """Setup error-only logger for critical issues"""
        logger = logging.getLogger("transcrevai.errors")
        logger.setLevel(logging.ERROR)
        logger.propagate = False  # Don't propagate to parent logger
        
        # Error file handler
        file_handler = logging.handlers.RotatingFileHandler(
            LOGGING_CONFIG["error_log"],
            maxBytes=LOGGING_CONFIG["max_file_size_mb"] * 1024 * 1024,
            backupCount=LOGGING_CONFIG["backup_count"],
            encoding='utf-8'
        )
        file_handler.setLevel(logging.ERROR)
        
        # Detailed formatter for errors
        error_formatter = logging.Formatter(
            "%(asctime)s - ERROR - %(name)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s",
            datefmt=LOGGING_CONFIG["date_format"]
        )
        file_handler.setFormatter(error_formatter)
        logger.addHandler(file_handler)
        
        cls._loggers["errors"] = logger
    
    @classmethod
    def _configure_third_party_loggers(cls) -> None:
        """Configure third-party library loggers to reduce noise"""
        # Whisper logging
        logging.getLogger("whisper").setLevel(logging.WARNING)
        logging.getLogger("whisper.transcribe").setLevel(logging.WARNING)
        
        # PyAudio and audio libraries
        logging.getLogger("pyaudio").setLevel(logging.WARNING)
        logging.getLogger("sounddevice").setLevel(logging.WARNING)
        logging.getLogger("librosa").setLevel(logging.WARNING)
        
        # ML libraries
        logging.getLogger("sklearn").setLevel(logging.WARNING)
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("numba").setLevel(logging.WARNING)
        
        # Web framework
        logging.getLogger("uvicorn").setLevel(logging.INFO)
        logging.getLogger("fastapi").setLevel(logging.INFO)
        logging.getLogger("websockets").setLevel(logging.WARNING)
        
        # HTTP libraries
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        
        # Suppress specific noisy warnings
        logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
        
        # PyTorch warnings (CPU-only mode)
        logging.getLogger("torch").setLevel(logging.WARNING)
        logging.getLogger("torchaudio").setLevel(logging.WARNING)
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get a logger with the specified name
        
        Args:
            name: Logger name (e.g., 'transcrevai.audio_processing')
            
        Returns:
            Configured logger instance
        """
        if not cls._handlers_setup:
            cls.setup_logging()
            
        return logging.getLogger(name)
    
    @classmethod
    def log_performance(cls, message: str, **kwargs) -> None:
        """Log performance metrics"""
        if not cls._handlers_setup:
            cls.setup_logging()
            
        performance_logger = cls._loggers.get("performance", logging.getLogger("transcrevai.performance"))
        
        # Format message with metrics
        if kwargs:
            metrics_str = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
            full_message = f"{message} | {metrics_str}"
        else:
            full_message = message
            
        performance_logger.info(full_message)
    
    @classmethod
    def log_resource_usage(cls, component: str, memory_mb: float, cpu_percent: float, **kwargs) -> None:
        """Log resource usage metrics"""
        if not cls._handlers_setup:
            cls.setup_logging()
            
        resource_logger = cls._loggers.get("resources", logging.getLogger("transcrevai.resources"))
        
        # Format resource message
        message = f"{component} | MEM={memory_mb:.1f}MB | CPU={cpu_percent:.1f}%"
        
        if kwargs:
            metrics_str = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
            message += f" | {metrics_str}"
            
        resource_logger.info(message)
    
    @classmethod
    def log_error(cls, logger_name: str, error: Exception, context: Optional[str] = None) -> None:
        """Log error with full context"""
        if not cls._handlers_setup:
            cls.setup_logging()
            
        error_logger = cls._loggers.get("errors", logging.getLogger("transcrevai.errors"))
        
        # Build error message
        message_parts = [f"[{logger_name}]"]
        
        if context:
            message_parts.append(f"Context: {context}")
            
        message_parts.extend([
            f"Error Type: {type(error).__name__}",
            f"Error Message: {str(error)}"
        ])
        
        full_message = " | ".join(message_parts)
        error_logger.error(full_message, exc_info=True)
    
    @classmethod
    def shutdown(cls) -> None:
        """Shutdown all loggers and handlers"""
        if cls._handlers_setup:
            # Flush all handlers
            for logger in cls._loggers.values():
                for handler in logger.handlers:
                    handler.flush()
                    handler.close()
            
            # Shutdown logging
            logging.shutdown()
            cls._handlers_setup = False


# Convenience functions for easy usage
def setup_app_logging(logger_name: str = "transcrevai") -> logging.Logger:
    """
    Setup and return a logger for the application
    
    Args:
        logger_name: Name for the logger
        
    Returns:
        Configured logger instance
    """
    TranscrevAILogger.setup_logging()
    return TranscrevAILogger.get_logger(logger_name)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance"""
    return TranscrevAILogger.get_logger(name)


def log_performance(message: str, **metrics) -> None:
    """Log performance metrics"""
    TranscrevAILogger.log_performance(message, **metrics)


def log_resource_usage(component: str, memory_mb: float, cpu_percent: float, **metrics) -> None:
    """Log resource usage"""
    TranscrevAILogger.log_resource_usage(component, memory_mb, cpu_percent, **metrics)


def log_error(logger_name: str, error: Exception, context: Optional[str] = None) -> None:
    """Log error with context"""
    TranscrevAILogger.log_error(logger_name, error, context)


# Module-level setup - automatically initialize when imported
if __name__ != "__main__":
    TranscrevAILogger.setup_logging()


# Test function for development
def test_logging():
    """Test all logging functions"""
    logger = setup_app_logging("test")
    
    logger.info("Testing main logger")
    logger.warning("Testing warning level")
    
    log_performance("Test performance", duration=1.23, accuracy=0.95)
    log_resource_usage("test_component", 512.5, 45.2, threads=4)
    
    try:
        raise ValueError("Test error for logging")
    except ValueError as e:
        log_error("test_logger", e, "Testing error logging")
    
    logger.info("Logging test completed")


if __name__ == "__main__":
    test_logging()