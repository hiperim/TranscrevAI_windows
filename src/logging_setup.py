import logging
import sys
from pathlib import Path

def setup_app_logging(level=logging.INFO, logger_name=None):
    """Setup application-specific logging for TranscrevAI"""
    
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
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
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
        # Use fixed path for consistency with other files
        log_dir = Path(r"c:\\TranscrevAI_windows\\data\\logs")
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
        # If file logging fails, at least we have console
        logger.warning(f"File logging setup failed: {e}")
    
    return logger

def get_logger(name=None):
    """Get application logger for specific module"""
    if name:
        return logging.getLogger(f"transcrevai.{name}")
    return logging.getLogger("transcrevai")