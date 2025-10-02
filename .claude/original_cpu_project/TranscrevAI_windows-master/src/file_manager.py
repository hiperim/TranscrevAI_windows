import os
import logging
import asyncio
import time
import tempfile
import sys
import shutil
from pathlib import Path
from typing import Union
from src.logging_setup import setup_app_logging
from config.app_config import APP_PACKAGE_NAME

# Use proper logging setup
logger = setup_app_logging(logger_name="transcrevai.file_manager")

def sanitize_path(user_input, base_dir):
    # Securely sanitize user input paths
    try:
        base_path = Path(base_dir).resolve()
        resolved_path = base_path.joinpath(user_input).resolve()
        
        # Ensure base directory exists
        if not base_path.exists():
            raise SecurityError(f"Base directory does not exist: {base_dir}")
        
        if not resolved_path.is_relative_to(base_path):
            raise SecurityError("Attempted directory traversal detected")
        
        return str(resolved_path)
    except Exception as e:
        logger.error(f"Path sanitization failed: {e}")
        raise SecurityError(f"Invalid path operation: {str(e)}")

class SecurityError(RuntimeError):
    # Custom security exception with proper logging
    def __init__(self, message):
        logger.error(f"SECURITY VIOLATION: {message}")
        super().__init__(message)

class FileManager:    
    @staticmethod
    def get_base_directory(subdir=""):
        # Get base application directory
        base = Path(__file__).resolve().parent.parent.parent
        return str(base / subdir) if subdir else str(base)
    
    @staticmethod
    def get_data_path(subdir="") -> str:
        # Get data directory path with validation
        # Use cross-platform base directory
        from config.app_config import DATA_DIR, _ensure_directories_created
        
        # Lazy directory creation
        _ensure_directories_created()
        
        full_path = DATA_DIR / subdir

        # Ensure specific subdirectory exists
        if subdir:
            try:
                full_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"Failed to create data directory: {full_path}")
                raise RuntimeError(f"Data directory creation failed: {str(e)}")

        return str(full_path.resolve())
    
    @staticmethod
    def get_unified_temp_dir() -> str:
        """Create secure temporary directory with race-condition prevention"""
        try:
            base_temp = FileManager.get_data_path("temp")
            FileManager.ensure_directory_exists(base_temp)
            
            # Use tempfile.mkdtemp which is atomic and handles race conditions
            temp_dir = tempfile.mkdtemp(
                dir=base_temp,
                prefix=f"temp_{os.getpid()}_",
                suffix=f"_{int(time.time() * 1000000)}"  # microseconds for better uniqueness
            )
            
            return FileManager.validate_path(temp_dir)
        except Exception as e:
            logger.error(f"Temporary directory creation failed: {e}")
            raise RuntimeError(f"Cannot create temporary directory: {str(e)}")
    
    @staticmethod
    def ensure_directory_exists(path):
        """Safely create directory with proper error handling"""
        try:
            path_obj = Path(path)
            path_obj.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Directory ensured: {path}")
        except PermissionError as e:
            logger.error(f"Permission denied creating directory: {path}")
            raise RuntimeError(f"Permission denied: {str(e)}")
        except Exception as e:
            logger.error(f"Directory creation failed: {path} - {e}")
            raise RuntimeError(f"Filesystem error: {str(e)}")
    
    @staticmethod
    def validate_path(user_path: str) -> str:
        """Enhanced path validation with security checks"""
        try:
            resolved = Path(user_path).resolve(strict=False)
            
            # Define allowed directories with existence checks
            allowed_dirs = []
            
            # Application data directory - use cross-platform path
            from config.app_config import DATA_DIR
            base_dir = DATA_DIR
            if base_dir.exists():
                allowed_dirs.append(base_dir.resolve())
            else:
                # Create data directory if it doesn't exist
                try:
                    base_dir.mkdir(parents=True, exist_ok=True)
                    allowed_dirs.append(base_dir.resolve())
                except Exception as e:
                    logger.error(f"Cannot create base data directory: {e}")
            
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
                    # Handle paths that can't be compared
                    continue
            
            if not path_allowed:
                logger.error(f"Path validation failed: {resolved} not in allowed directories: {allowed_dirs}")
                raise SecurityError(f"Path access denied: {resolved}")
            
            return str(resolved)
            
        except SecurityError:
            raise  # Re-raise security errors
        except ValueError as e:
            logger.error(f"Path validation failed: {e}")
            raise SecurityError("Invalid path format") from e
        except Exception as e:
            logger.error(f"Path validation error: {e}")
            raise SecurityError("Path validation failed") from e
    
    @staticmethod
    def save_audio(data, filename="output.wav") -> str:
        """Save audio data with validation and error handling"""
        try:
            # Use data path instead of validate_path for inputs
            safe_dir = FileManager.get_data_path("inputs")
            output_path = Path(safe_dir) / filename
            
            # Validate filename
            if ".." in filename or "/" in filename or "\\" in filename:
                raise SecurityError("Invalid filename detected")
            
            FileManager.ensure_directory_exists(str(output_path.parent))
            
            # Write with proper error handling
            try:
                with open(str(output_path), 'wb') as f:
                    f.write(data)
                    f.flush()  # Ensure data is written
                    os.fsync(f.fileno())  # Force filesystem sync
            except IOError as e:
                logger.error(f"File write failed: {e}")
                raise RuntimeError(f"Cannot write audio file: {str(e)}")
            
            logger.info(f"Audio file saved: {output_path}")
            return str(output_path)
            
        except SecurityError:
            raise  # Re-raise security errors
        except Exception as e:
            logger.error(f"Audio save failed: {e}")
            raise RuntimeError(f"Audio save error: {str(e)}")
    
    @staticmethod
    def save_transcript(data: Union[str, list], filename="output.txt") -> None:
        """Save transcript with proper validation"""
        try:
            output_dir = FileManager.get_data_path("transcripts")
            
            # Validate filename
            if ".." in filename or "/" in filename or "\\" in filename:
                raise SecurityError("Invalid filename detected")
            
            output_path = Path(output_dir) / filename
            FileManager.ensure_directory_exists(str(output_path.parent))
            
            with open(str(output_path), 'w', encoding='utf-8') as f:
                f.write(str(data))
                f.flush()
                os.fsync(f.fileno())
            
            logger.info(f"Transcript saved: {output_path}")
            
        except SecurityError:
            raise  # Re-raise security errors
        except Exception as e:
            logger.error(f"Transcript save failed: {e}")
            raise RuntimeError(f"Transcript save error: {str(e)}")
    
    @staticmethod
    def cleanup_temp_dirs():
        """Secure atomic cleanup of temporary directories with path validation"""
        try:
            base_temp = FileManager.get_data_path("temp")
            
            if not Path(base_temp).exists():
                logger.info("No temp directory to clean")
                return
            
            # Validate base_temp is actually our temp directory
            from config.app_config import TEMP_DIR
            if not Path(base_temp).resolve().is_relative_to(TEMP_DIR.parent.resolve()):
                raise SecurityError("Invalid temp directory for cleanup")
            
            # Get list of items to clean atomically
            try:
                temp_items = list(Path(base_temp).iterdir())
            except OSError as e:
                logger.error(f"Failed to list temp directory: {e}")
                return
            
            # Filter items to clean
            items_to_clean = []
            for temp_item in temp_items:
                if temp_item.name.startswith("temp_") or temp_item.name.startswith("atomic_"):
                    items_to_clean.append(temp_item)
                else:
                    logger.warning(f"Skipping non-temp item: {temp_item.name}")
            
            # Perform cleanup with atomic operations
            cleaned_count = 0
            error_count = 0
            
            for item_path in items_to_clean:
                try:
                    # Use atomic operations for each item
                    if item_path.is_dir():
                        # For directories, use a temporary rename + delete pattern
                        temp_name = item_path.with_name(f"{item_path.name}_deleting_{int(time.time() * 1000000)}")
                        try:
                            item_path.rename(temp_name)
                            shutil.rmtree(str(temp_name))
                            cleaned_count += 1
                        except OSError:
                            # If rename fails, try direct deletion
                            shutil.rmtree(str(item_path))
                            cleaned_count += 1
                    elif item_path.is_file():
                        item_path.unlink()
                        cleaned_count += 1
                except PermissionError as e:
                    logger.warning(f"Permission denied cleaning: {item_path}")
                    error_count += 1
                except Exception as e:
                    logger.warning(f"Temp cleanup failed for {item_path}: {e}")
                    error_count += 1
            
            logger.info(f"Temp cleanup completed: {cleaned_count} items cleaned, {error_count} errors")
            
        except SecurityError:
            raise  # Re-raise security errors
        except Exception as e:
            logger.error(f"Temp directory cleanup failed: {e}")
            raise RuntimeError(f"Cleanup operation failed: {str(e)}")