"""
Centralized File Management for TranscrevAI.

Provides a secure and robust interface for file operations, including
sanitizing filenames, validating paths against an allow-list, and managing
the application's data directory structure.
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

import aiofiles
from fastapi import UploadFile

# Use proper logging setup
logger = setup_app_logging(logger_name="transcrevai.file_manager")


class SecurityError(RuntimeError):
    """Custom security exception with proper logging"""
    
    def __init__(self, message):
        logger.error(f"SECURITY VIOLATION: {message}")
        super().__init__(message)


class FileManager:
    """File management with a configurable data directory.

    Accepts a data directory path via the constructor for easier testing
    and deployment flexibility. Defaults to environment variable or a
    standard ./data location.
    """

    DEFAULT_DATA_DIR = Path("./data").resolve()

    def __init__(self, data_dir: Optional[Union[str, Path]] = None):
        """Initialize FileManager with a specific data directory.

        Args:
            data_dir: Optional custom data directory path.
                     If None, it's loaded from the environment or a default.
        """
        if data_dir is None:
            self.data_dir = self._load_data_dir_from_config()
        else:
            self.data_dir = Path(data_dir).resolve()
        
        self._ensure_directory_structure()
        logger.info(f"FileManager initialized with data_dir: {self.data_dir}")

    def _load_data_dir_from_config(self) -> Path:
        """Load the data directory from the environment or use the default.

        Priority order:
        1. Environment variable DATA_DIR
        2. Default: ./data

        Returns:
            Path object pointing to the data directory.
        """
        env_data_dir = os.getenv("DATA_DIR")
        if env_data_dir:
            logger.info(f"Using DATA_DIR from environment: {env_data_dir}")
            return Path(env_data_dir).resolve()
        
        logger.info(f"DATA_DIR environment variable not set, using default: {self.DEFAULT_DATA_DIR}")
        return self.DEFAULT_DATA_DIR

    def _ensure_directory_structure(self) -> None:
        """Create required subdirectories if they don't exist."""
        subdirs = ["uploads", "transcripts", "srt", "subtitles", "recordings", "temp", "inputs"]
        for subdir in subdirs:
            path = self.data_dir / subdir
            path.mkdir(parents=True, exist_ok=True)

    def get_data_path(self, subdir: str = "") -> Path:
        """Get a path to a subdirectory within the main data directory."""
        return self.data_dir / subdir

    async def save_uploaded_file(self, file: UploadFile, filename: str) -> str:
        """Asynchronously save an uploaded file to the data directory and return its path."""
        safe_filename = self._sanitize_filename(filename)
        save_dir = self.get_data_path("inputs")
        output_path = save_dir / safe_filename
        
        try:
            async with aiofiles.open(output_path, "wb") as f:
                while content := await file.read(1024 * 1024):  # Read in 1MB chunks
                    await f.write(content)
            logger.info(f"Uploaded file saved asynchronously: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Failed to save uploaded file asynchronously: {e}", exc_info=True)
            raise RuntimeError(f"Failed to save file: {e}") from e

    async def read_file_async(self, filepath: Union[str, Path]) -> str:
        """Read file contents asynchronously."""
        async with aiofiles.open(filepath, "r", encoding="utf-8") as f:
            return await f.read()

    async def write_file_async(self, filepath: Union[str, Path], content: str) -> None:
        """Write content to a file asynchronously."""
        async with aiofiles.open(filepath, "w", encoding="utf-8") as f:
            await f.write(content)

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to prevent path traversal and malicious filenames."""
        import re
        safe_name = re.sub(r'[^\w\-.]', '_', filename)
        safe_name = safe_name.lstrip('.')
        if not safe_name or safe_name == '_':
            safe_name = f"file_{int(time.time())}"
        if len(safe_name) > 255:
            name_parts = safe_name.rsplit('.', 1)
            if len(name_parts) == 2:
                safe_name = name_parts[0][:240] + '.' + name_parts[1]
            else:
                safe_name = safe_name[:255]
        return safe_name

    def ensure_directory_exists(self, path: Union[str, Path]) -> None:
        """Safely create a directory."""
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            logger.error(f"Permission denied creating directory: {path}")
            raise SecurityError(f"Permission denied: {str(e)}") from e
        except Exception as e:
            logger.error(f"Directory creation failed: {path} - {e}")
            raise RuntimeError(f"Filesystem error: {str(e)}") from e

    def validate_path(self, user_path: str) -> str:
        """Validate that a given path is within the allowed data directory."""
        try:
            resolved_path = Path(user_path).resolve()
            allowed_dirs = [self.data_dir, Path(tempfile.gettempdir()).resolve()]

            is_allowed = any(resolved_path.is_relative_to(allowed) for allowed in allowed_dirs)

            if not is_allowed:
                logger.error(f"Path validation failed: {resolved_path} is not in any allowed directory.")
                raise SecurityError(f"Path access denied: {resolved_path}")
                
            return str(resolved_path)
        except SecurityError:
            raise
        except Exception as e:
            logger.error(f"Path validation error: {e}", exc_info=True)
            raise SecurityError("Path validation failed due to an unexpected error.") from e
    


