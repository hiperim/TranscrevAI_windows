"""
TranscrevAI Optimized - Professional Logging Setup
Sistema de logging robusto e performático com múltiplos handlers
"""

import logging
import logging.handlers
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union
import asyncio
import threading
from queue import Queue
import traceback

try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.traceback import install
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    from loguru import logger as loguru_logger
    LOGURU_AVAILABLE = True
except ImportError:
    LOGURU_AVAILABLE = False


class PerformanceTracker:
    """Track performance metrics for logging"""
    
    def __init__(self):
        self.start_times: Dict[str, float] = {}
        self.metrics: Dict[str, list] = {}
        self._lock = threading.Lock()
    
    def start(self, operation: str) -> None:
        """Start timing an operation"""
        with self._lock:
            self.start_times[operation] = time.time()
    
    def end(self, operation: str) -> Optional[float]:
        """End timing an operation and return duration"""
        with self._lock:
            start_time = self.start_times.get(operation)
            if start_time is None:
                return None
            
            duration = time.time() - start_time
            del self.start_times[operation]
            
            if operation not in self.metrics:
                self.metrics[operation] = []
            
            self.metrics[operation].append(duration)
            
            # Keep only last 100 measurements
            if len(self.metrics[operation]) > 100:
                self.metrics[operation] = self.metrics[operation][-100:]
            
            return duration
    
    def get_stats(self, operation: str) -> Dict[str, float]:
        """Get statistics for an operation"""
        with self._lock:
            durations = self.metrics.get(operation, [])
            if not durations:
                return {}
            
            return {
                "count": len(durations),
                "total": sum(durations),
                "average": sum(durations) / len(durations),
                "min": min(durations),
                "max": max(durations),
                "last": durations[-1]
            }


class ResourceTracker:
    """Track resource usage for logging"""
    
    def __init__(self):
        self.resource_logs: list = []
        self._lock = threading.Lock()
    
    def log_usage(self, component: str, memory_mb: float, cpu_percent: float, **kwargs):
        """Log resource usage"""
        with self._lock:
            entry = {
                "timestamp": time.time(),
                "component": component,
                "memory_mb": memory_mb,
                "cpu_percent": cpu_percent,
                **kwargs
            }
            
            self.resource_logs.append(entry)
            
            # Keep only last 1000 entries
            if len(self.resource_logs) > 1000:
                self.resource_logs = self.resource_logs[-1000:]
    
    def get_recent_usage(self, component: Optional[str] = None, last_n: int = 10) -> list:
        """Get recent resource usage"""
        with self._lock:
            logs = self.resource_logs
            
            if component:
                logs = [log for log in logs if log["component"] == component]
            
            return logs[-last_n:] if logs else []


class AsyncLogHandler(logging.Handler):
    """Asynchronous log handler that doesn't block"""
    
    def __init__(self, target_handler: logging.Handler, maxsize: int = 1000):
        super().__init__()
        self.target_handler = target_handler
        self.queue = Queue(maxsize=maxsize)
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        self._shutdown = False
    
    def _worker(self):
        """Worker thread that processes log messages"""
        while not self._shutdown:
            try:
                record = self.queue.get(timeout=1.0)
                if record is None:  # Shutdown signal
                    break
                
                self.target_handler.emit(record)
                self.queue.task_done()
                
            except Exception:
                pass  # Ignore exceptions in logging
    
    def emit(self, record: logging.LogRecord):
        """Emit a log record asynchronously"""
        try:
            if not self.queue.full():
                self.queue.put_nowait(record)
        except:
            pass  # Don't let logging errors crash the app
    
    def close(self):
        """Close the async handler"""
        self._shutdown = True
        try:
            self.queue.put_nowait(None)  # Shutdown signal
        except:
            pass
        
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        
        self.target_handler.close()
        super().close()


class TranscrevAIFormatter(logging.Formatter):
    """Custom formatter for TranscrevAI with enhanced information"""
    
    def __init__(self):
        self.start_time = time.time()
        super().__init__()
    
    def format(self, record: logging.LogRecord) -> str:
        # Add relative timestamp
        record.relative_time = time.time() - self.start_time
        
        # Add memory usage if available
        try:
            import psutil
            process = psutil.Process()
            record.memory_mb = process.memory_info().rss / (1024 * 1024)
        except:
            record.memory_mb = 0.0
        
        # Format based on log level
        if record.levelno >= logging.ERROR:
            # Enhanced error formatting
            template = (
                "[ERROR] {asctime} | {name} | {memory_mb:.1f}MB | {relative_time:.2f}s\n"
                "Message: {message}\n"
                "File: {pathname}:{lineno}\n"
                "Function: {funcName}\n"
            )
            
            if record.exc_info:
                template += "Exception: {exc_text}\n"
                record.exc_text = ''.join(traceback.format_exception(*record.exc_info))
            
        elif record.levelno >= logging.WARNING:
            # Warning formatting
            template = "[WARN] {asctime} | {name} | {memory_mb:.1f}MB | {message}"
            
        elif record.levelno >= logging.INFO:
            # Info formatting
            template = "[INFO] {asctime} | {name} | {message}"
            
        else:
            # Debug formatting with more details
            template = "[DEBUG] {asctime} | {name} | {pathname}:{lineno} | {message}"
        
        # Apply formatting
        formatter = logging.Formatter(template, style='{')
        return formatter.format(record)


class LoggingConfig:
    """Configuration for logging system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logs_dir = Path(config.get("logs_dir", "logs"))
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Global trackers
        self.performance_tracker = PerformanceTracker()
        self.resource_tracker = ResourceTracker()
        
        # Logging levels
        self.level = getattr(logging, config.get("level", "INFO").upper())
        self.console_logging = config.get("enable_console_logging", True)
        self.file_logging = config.get("enable_file_logging", True)
        self.performance_logging = config.get("enable_performance_logging", True)
        self.resource_logging = config.get("enable_resource_logging", True)
        
        # File settings
        self.max_bytes = config.get("file_max_bytes", 10 * 1024 * 1024)  # 10MB
        self.backup_count = config.get("file_backup_count", 5)
    
    def setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging system"""
        
        # Install rich traceback if available
        if RICH_AVAILABLE:
            install(show_locals=True)
        
        # Root logger configuration
        root_logger = logging.getLogger()
        root_logger.setLevel(self.level)
        
        # Clear any existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        if self.console_logging:
            if RICH_AVAILABLE:
                console_handler = RichHandler(
                    console=Console(stderr=True),
                    show_path=False,
                    rich_tracebacks=True,
                    tracebacks_show_locals=True
                )
                console_handler.setFormatter(logging.Formatter("%(message)s"))
            else:
                console_handler = logging.StreamHandler(sys.stderr)
                console_handler.setFormatter(TranscrevAIFormatter())
            
            console_handler.setLevel(self.level)
            root_logger.addHandler(AsyncLogHandler(console_handler))
        
        # File handlers
        if self.file_logging:
            # Main log file
            main_log_file = self.logs_dir / "transcrevai.log"
            main_handler = logging.handlers.RotatingFileHandler(
                main_log_file,
                maxBytes=self.max_bytes,
                backupCount=self.backup_count,
                encoding='utf-8'
            )
            main_handler.setFormatter(TranscrevAIFormatter())
            main_handler.setLevel(self.level)
            root_logger.addHandler(AsyncLogHandler(main_handler))
            
            # Error log file
            error_log_file = self.logs_dir / "transcrevai_errors.log"
            error_handler = logging.handlers.RotatingFileHandler(
                error_log_file,
                maxBytes=self.max_bytes,
                backupCount=self.backup_count,
                encoding='utf-8'
            )
            error_handler.setFormatter(TranscrevAIFormatter())
            error_handler.setLevel(logging.ERROR)
            root_logger.addHandler(AsyncLogHandler(error_handler))
            
            # Performance log file
            if self.performance_logging:
                perf_log_file = self.logs_dir / "transcrevai_performance.log"
                perf_handler = logging.handlers.RotatingFileHandler(
                    perf_log_file,
                    maxBytes=self.max_bytes // 2,  # Smaller files for performance logs
                    backupCount=3,
                    encoding='utf-8'
                )
                perf_formatter = logging.Formatter(
                    "%(asctime)s | PERF | %(message)s"
                )
                perf_handler.setFormatter(perf_formatter)
                perf_handler.setLevel(logging.INFO)
                
                # Create performance logger
                perf_logger = logging.getLogger("transcrevai.performance")
                perf_logger.addHandler(AsyncLogHandler(perf_handler))
                perf_logger.setLevel(logging.INFO)
                perf_logger.propagate = False
        
        # Setup Loguru integration if available
        if LOGURU_AVAILABLE:
            self._setup_loguru_integration()
        
        return root_logger
    
    def _setup_loguru_integration(self):
        """Setup Loguru integration for enhanced logging"""
        try:
            # Configure loguru
            loguru_logger.remove()  # Remove default handler
            
            if self.console_logging:
                loguru_logger.add(
                    sys.stderr,
                    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                           "<level>{level: <8}</level> | "
                           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                           "<level>{message}</level>",
                    level=logging.getLevelName(self.level),
                    colorize=True,
                    backtrace=True,
                    diagnose=True
                )
            
            if self.file_logging:
                loguru_logger.add(
                    self.logs_dir / "transcrevai_loguru.log",
                    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
                    level=logging.getLevelName(self.level),
                    rotation="10 MB",
                    retention="5 files",
                    compression="gz",
                    backtrace=True,
                    diagnose=True
                )
        
        except Exception as e:
            logging.warning(f"Failed to setup Loguru integration: {e}")


# Global instances
_logging_config: Optional[LoggingConfig] = None
_main_logger: Optional[logging.Logger] = None


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Setup logging system with configuration"""
    global _logging_config, _main_logger
    
    if _logging_config is None:
        _logging_config = LoggingConfig(config)
        _main_logger = _logging_config.setup_logging()
        
        # Log startup message
        _main_logger.info("=== TranscrevAI Optimized Logging System Started ===")
        _main_logger.info(f"Log level: {logging.getLevelName(_logging_config.level)}")
        _main_logger.info(f"Logs directory: {_logging_config.logs_dir}")
        _main_logger.info(f"Console logging: {_logging_config.console_logging}")
        _main_logger.info(f"File logging: {_logging_config.file_logging}")
        _main_logger.info(f"Rich available: {RICH_AVAILABLE}")
        _main_logger.info(f"Loguru available: {LOGURU_AVAILABLE}")
    
    return _main_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name"""
    return logging.getLogger(name)


def log_performance(operation: str, **kwargs) -> None:
    """Log performance information"""
    global _logging_config
    
    if _logging_config and _logging_config.performance_logging:
        duration = kwargs.get('duration', 0.0)
        
        # Track in performance tracker
        if 'duration' in kwargs:
            _logging_config.performance_tracker.metrics.setdefault(operation, []).append(duration)
        
        # Format performance message
        perf_data = []
        for key, value in kwargs.items():
            if isinstance(value, float):
                if key.endswith('_mb'):
                    perf_data.append(f"{key}={value:.1f}")
                elif key.endswith('_percent'):
                    perf_data.append(f"{key}={value:.1f}%")
                else:
                    perf_data.append(f"{key}={value:.3f}")
            else:
                perf_data.append(f"{key}={value}")
        
        perf_msg = f"{operation} | {' | '.join(perf_data)}"
        
        perf_logger = logging.getLogger("transcrevai.performance")
        perf_logger.info(perf_msg)


def log_resource_usage(component: str, memory_mb: float, cpu_percent: float, **kwargs) -> None:
    """Log resource usage information"""
    global _logging_config
    
    if _logging_config and _logging_config.resource_logging:
        # Track in resource tracker
        _logging_config.resource_tracker.log_usage(component, memory_mb, cpu_percent, **kwargs)
        
        # Format resource message
        extra_data = []
        for key, value in kwargs.items():
            extra_data.append(f"{key}={value}")
        
        extra_str = f" | {' | '.join(extra_data)}" if extra_data else ""
        resource_msg = f"{component} | memory={memory_mb:.1f}MB | cpu={cpu_percent:.1f}%{extra_str}"
        
        resource_logger = logging.getLogger("transcrevai.resource")
        resource_logger.info(resource_msg)


def get_performance_stats() -> Dict[str, Any]:
    """Get performance statistics"""
    global _logging_config
    
    if _logging_config:
        stats = {}
        for operation, durations in _logging_config.performance_tracker.metrics.items():
            stats[operation] = _logging_config.performance_tracker.get_stats(operation)
        return stats
    
    return {}


def get_resource_stats() -> Dict[str, Any]:
    """Get resource usage statistics"""
    global _logging_config
    
    if _logging_config:
        return {
            "recent_logs_count": len(_logging_config.resource_tracker.resource_logs),
            "components": list(set(
                log["component"] for log in _logging_config.resource_tracker.resource_logs
            )),
        }
    
    return {}


class PerformanceLogger:
    """Context manager for performance logging"""
    
    def __init__(self, operation: str, logger: Optional[logging.Logger] = None):
        self.operation = operation
        self.logger = logger or get_logger("transcrevai.performance")
        self.start_time = None
        self.extra_data = {}
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.extra_data["duration"] = duration
            
            if exc_type:
                self.extra_data["error"] = str(exc_val)
                self.logger.error(f"PERFORMANCE | {self.operation} | FAILED | {self.extra_data}")
            else:
                log_performance(self.operation, **self.extra_data)
    
    def add_data(self, **kwargs):
        """Add extra data to performance log"""
        self.extra_data.update(kwargs)


# Convenience function for performance measurement
def measure_performance(operation: str):
    """Decorator for measuring function performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with PerformanceLogger(operation) as perf:
                result = func(*args, **kwargs)
                perf.add_data(
                    function=func.__name__,
                    args_count=len(args),
                    kwargs_count=len(kwargs)
                )
                return result
        return wrapper
    return decorator


# Export main functions
__all__ = [
    "setup_logging", 
    "get_logger", 
    "log_performance", 
    "log_resource_usage",
    "get_performance_stats",
    "get_resource_stats", 
    "PerformanceLogger",
    "measure_performance"
]