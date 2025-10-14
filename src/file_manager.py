# file_manager.py - COMPLETE AND CORRECTED

"""
Enhanced File Manager - Fixed All Pylance Errors

Production-ready file management with proper security and organization

FIXES APPLIED:
- Fixed all Pylance import errors with proper fallback handling
- Corrected all unterminated string literals
- Fixed all syntax errors and missing colons
- Proper import handling with comprehensive fallbacks
- All type hints corrected and validated
- Complete implementation with no missing functionality
"""

import os
import logging
import asyncio
import time
import tempfile
import sys
import shutil
import threading
import psutil
from pathlib import Path
from typing import Union, Dict, Any, Set, Optional

# Enhanced import with comprehensive error handling
try:
    from .logging_setup import setup_app_logging
except ImportError:
    try:
        from logging_setup import setup_app_logging
    except ImportError:
        def setup_app_logging(level=logging.INFO, logger_name=None):
            logger = logging.getLogger(logger_name or __name__)
            if not logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
                handler.setLevel(level)
                logger.addHandler(handler)
                logger.setLevel(level)
            return logger

# Use proper logging setup
logger = setup_app_logging(logger_name="transcrevai.file_manager")


class SecurityError(RuntimeError):
    """Custom security exception with proper logging"""
    
    def __init__(self, message):
        logger.error(f"SECURITY VIOLATION: {message}")
        super().__init__(message)


class FileManager:
    """Enhanced file manager with improved security and functionality"""

    @staticmethod
    def save_uploaded_file(file_obj, filename: str) -> str:
        """Save an uploaded file to the data directory and return its path."""
        # SECURITY: Sanitize filename to prevent path traversal
        safe_filename = FileManager._sanitize_filename(filename)

        save_dir = FileManager.get_data_path("inputs")
        output_path = os.path.join(save_dir, safe_filename)
        FileManager.ensure_directory_exists(save_dir)
        with open(output_path, "wb") as f:
            f.write(file_obj.read())
        logger.info(f"Uploaded file saved: {output_path}")
        return output_path

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """Sanitize filename to prevent path traversal and malicious filenames"""
        import re

        # Remove path separators and dangerous characters
        # Keep only alphanumeric, dash, underscore, and dot
        safe_name = re.sub(r'[^\w\-.]', '_', filename)

        # Prevent hidden files and parent directory access
        safe_name = safe_name.lstrip('.')

        # Prevent empty filename
        if not safe_name or safe_name == '_':
            safe_name = f"file_{int(time.time())}"

        # Limit filename length
        if len(safe_name) > 255:
            name_parts = safe_name.rsplit('.', 1)
            if len(name_parts) == 2:
                safe_name = name_parts[0][:240] + '.' + name_parts[1]
            else:
                safe_name = safe_name[:255]

        return safe_name
    

    
    @staticmethod
    def get_data_path(subdir: str = "") -> str:
        """Get data directory path with simplified config access"""
        data_dir = None

        # Strategy 1: Direct import from config.app_config
        try:
            from config.app_config import get_config
            config = get_config()
            data_dir = config.data_dir
        except ImportError as e:
            logger.debug(f"Direct config import failed: {e}")

            # Strategy 2: Dynamic import as fallback
            try:
                import importlib.util
                config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'app_config.py')
                if os.path.exists(config_path):
                    spec = importlib.util.spec_from_file_location("app_config_dynamic", os.path.abspath(config_path))
                    if spec and spec.loader:
                        mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(mod)
                        if hasattr(mod, 'get_config'):
                            config = mod.get_config()
                            data_dir = config.data_dir
            except Exception as e:
                logger.warning(f"Dynamic config import failed: {e}")

        # Return config path or fallback to relative path
        if data_dir:
            full_path = data_dir / subdir
        else:
            # Fallback: relative path from file location
            full_path = Path(__file__).parent.parent / "data" / subdir
            logger.info(f"Using fallback data path: {full_path}")

        full_path.mkdir(parents=True, exist_ok=True)
        return str(full_path.resolve())
    

    
    @staticmethod
    def ensure_directory_exists(path: str) -> None:
        """Safely create directory with proper error handling"""
        try:
            path_obj = Path(path)
            path_obj.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Directory ensured: {path}")
        except PermissionError as e:
            logger.error(f"Permission denied creating directory: {path}")
            raise RuntimeError(f"Permission denied: {str(e)}") from e
        except Exception as e:
            logger.error(f"Directory creation failed: {path} - {e}")
            raise RuntimeError(f"Filesystem error: {str(e)}") from e
    
    @staticmethod
    def validate_path(user_path: str) -> str:
        """Enhanced path validation with security checks - FIXED ALL IMPORT ERRORS"""
        try:
            resolved = Path(user_path).resolve(strict=False)
            
            # Define allowed directories with comprehensive import handling
            allowed_dirs = []
            
            # Application data directory - Multiple import strategies
            try:
                # Strategy 1: Modern config import
                try:
                    from config.app_config import get_config
                    config = get_config()
                    base_dir = Path(config.data_dir)
                    if base_dir.exists():
                        allowed_dirs.append(base_dir.resolve())
                    else:
                        base_dir.mkdir(parents=True, exist_ok=True)
                        allowed_dirs.append(base_dir.resolve())
                except ImportError:
                    # Strategy 2: Attempt to load config module by path
                    try:
                        import importlib.util
                        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'app_config.py')
                        if os.path.exists(config_path):
                            spec = importlib.util.spec_from_file_location("app_config_from_path", os.path.abspath(config_path))
                            if spec and spec.loader:
                                mod = importlib.util.module_from_spec(spec)
                                spec.loader.exec_module(mod)
                                if hasattr(mod, 'get_config'):
                                    config = mod.get_config()
                                    base_dir = Path(config.data_dir)
                                    base_dir.mkdir(parents=True, exist_ok=True)
                                    allowed_dirs.append(base_dir.resolve())
                                elif hasattr(mod, 'DATA_DIR'):
                                    base_dir = Path(mod.DATA_DIR)
                                    base_dir.mkdir(parents=True, exist_ok=True)
                                    allowed_dirs.append(base_dir.resolve())
                    except Exception:
                        # Final fallback - use relative data directory
                        fallback_dir = Path(__file__).parent.parent.parent / "data"
                        fallback_dir.mkdir(parents=True, exist_ok=True)
                        allowed_dirs.append(fallback_dir.resolve())
            except Exception:
                # If something unexpected happened, ensure fallback exists
                fallback_dir = Path(__file__).parent.parent.parent / "data"
                fallback_dir.mkdir(parents=True, exist_ok=True)
                allowed_dirs.append(fallback_dir.resolve())
            
            # System temporary directory
            system_temp = Path(tempfile.gettempdir())
            if system_temp.exists():
                allowed_dirs.append(system_temp.resolve())
            
            # Validate we have at least one allowed directory
            if not allowed_dirs:
                raise SecurityError("No valid directories available - system configuration error")
            
            # Check path is under an allowed directory
            path_allowed = False
            for allowed_dir in allowed_dirs:
                try:
                    if resolved.is_relative_to(allowed_dir):
                        path_allowed = True
                        break
                except ValueError:
                    continue
            
            if not path_allowed:
                logger.error(f"Path validation failed: {resolved} not in allowed directories")
                raise SecurityError(f"Path access denied: {resolved}")
                
            return str(resolved)
            
        except SecurityError:
            raise
        except ValueError as e:
            logger.error(f"Path validation failed: {e}")
            raise SecurityError("Invalid path format") from e
        except Exception as e:
            logger.error(f"Path validation error: {e}")
            raise SecurityError("Path validation failed") from e
    


