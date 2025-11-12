import os
import logging
import time
import tempfile
from pathlib import Path
from typing import Union, Optional

import aiofiles
from fastapi import UploadFile

# Simple logging setup
logger = logging.getLogger("transcrevai.file_manager")


class SecurityError(RuntimeError):
    """Custom security exception with proper logging"""
    
    def __init__(self, message):
        logger.error(f"SECURITY VIOLATION: {message}")
        super().__init__(message)


class FileManager:
    """File management with configurable data directory - defaults to environment variable or a
    standard ./data location"""

    DEFAULT_DATA_DIR = Path("./data").resolve()

    def __init__(self, data_dir: Optional[Union[str, Path]] = None):
        
        if data_dir is None:
            self.data_dir = self._load_data_dir_from_config()
        else:
            self.data_dir = Path(data_dir).resolve()
        
        self._ensure_directory_structure()
        logger.info(f"FileManager initialized with data_dir: {self.data_dir}")

    def _load_data_dir_from_config(self) -> Path:
        
        env_data_dir = os.getenv("DATA_DIR")
        if env_data_dir:
            logger.info(f"Using DATA_DIR from environment: {env_data_dir}")
            return Path(env_data_dir).resolve()
        
        logger.info(f"DATA_DIR environment variable not set, using default: {self.DEFAULT_DATA_DIR}")
        return self.DEFAULT_DATA_DIR

    def _ensure_directory_structure(self) -> None:
        
        subdirs = ["uploads", "transcripts", "srt", "subtitles", "recordings", "temp", "inputs"]
        for subdir in subdirs:
            path = self.data_dir / subdir
            path.mkdir(parents=True, exist_ok=True)

    def get_data_path(self, subdir: str = "") -> Path:
        
        return self.data_dir / subdir

    async def save_uploaded_file(self, file: UploadFile, filename: str) -> str:
        """Async save uploaded file to data directory and return its path"""
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
        """Read file contents async"""
        async with aiofiles.open(filepath, "r", encoding="utf-8") as f:
            return await f.read()

    async def write_file_async(self, filepath: Union[str, Path], content: str) -> None:
        """Write content to a file async"""
        async with aiofiles.open(filepath, "w", encoding="utf-8") as f:
            await f.write(content)

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to prevent path traversal and malicious filenames"""
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
        """Safely create a directory"""
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            logger.error(f"Permission denied creating directory: {path}")
            raise SecurityError(f"Permission denied: {str(e)}") from e
        except Exception as e:
            logger.error(f"Directory creation failed: {path} - {e}")
            raise RuntimeError(f"Filesystem error: {str(e)}") from e

    def validate_path(self, user_path: str) -> str:
        """Validate that a given path is within allowed data directory"""
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
    


