import os
import logging
import zipfile
import requests
import asyncio
import time
import tempfile
import sys
import shutil
from pathlib import Path
from typing import Union
from src.logging_setup import setup_app_logging
from config.app_config import APP_PACKAGE_NAME

# Use application-specific logger
logger = logging.getLogger(__name__)

def sanitize_path(user_input, base_dir):
    """Securely sanitize user input paths"""
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
    """Custom security exception with proper logging"""
    def __init__(self, message):
        logger.error(f"SECURITY VIOLATION: {message}")
        super().__init__(message)

class FileManager:
    """Enhanced file manager with robust security and error handling"""
    
    @staticmethod
    def get_base_directory(subdir=""):
        """Get base application directory"""
        base = Path(__file__).resolve().parent.parent.parent
        return str(base / subdir) if subdir else str(base)

    @staticmethod
    def get_data_path(subdir="") -> str:
        """Get data directory path with validation"""
        base = Path(__file__).resolve().parent.parent.parent / "data"
        full_path = base / subdir
        
        # Ensure data directory exists
        try:
            full_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create data directory: {full_path}")
            raise RuntimeError(f"Data directory creation failed: {str(e)}")
            
        return os.path.normpath(str(full_path))

    @staticmethod
    def get_unified_temp_dir() -> str:
        """Create secure temporary directory"""
        try:
            base_temp = FileManager.get_data_path("temp")
            FileManager.ensure_directory_exists(base_temp)
            temp_dir = tempfile.mkdtemp(
                dir=base_temp, 
                prefix=f"temp_{os.getpid()}_",
                suffix=f"_{int(time.time())}"
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
            
            # Application data directory
            base_dir = Path(__file__).parent.parent.parent / "data"
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
            output_path = os.path.join(safe_dir, filename)
            
            # Validate filename
            if ".." in filename or "/" in filename or "\\" in filename:
                raise SecurityError("Invalid filename detected")
            
            FileManager.ensure_directory_exists(os.path.dirname(output_path))
            
            # Write with proper error handling
            try:
                with open(output_path, 'wb') as f:
                    f.write(data)
                    f.flush()  # Ensure data is written
                    os.fsync(f.fileno())  # Force filesystem sync
            except IOError as e:
                logger.error(f"File write failed: {e}")
                raise RuntimeError(f"Cannot write audio file: {str(e)}")
            
            logger.info(f"Audio file saved: {output_path}")
            return output_path
            
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
                
            output_path = os.path.join(output_dir, filename)
            FileManager.ensure_directory_exists(os.path.dirname(output_path))
            
            with open(output_path, 'w', encoding='utf-8') as f:
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
    def _sync_download_and_extract(url, language_code, output_dir):
        """Synchronous download and extract with enhanced security"""
        try:
            # Validate language code to prevent path injection
            if not language_code.isalnum() or len(language_code) > 10:
                raise SecurityError("Invalid language code format")
                
            model_path = os.path.join(output_dir, language_code)
            
            if os.path.exists(model_path) and any(os.listdir(model_path)):
                logger.info(f"Existing model found: {language_code}")
                return model_path

            zip_path = os.path.join(output_dir, f"{language_code}.zip")

            for attempt in range(3):
                try:
                    logger.info(f"Downloading model: {language_code} (attempt {attempt + 1})")
                    
                    # Validate URL
                    from urllib.parse import urlparse
                    parsed = urlparse(url)
                    if not parsed.scheme.startswith('http'):
                        raise SecurityError("Invalid URL scheme")
                    
                    response = requests.get(url, stream=True, timeout=30)
                    response.raise_for_status()

                    with open(zip_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=1024):
                            f.write(chunk)

                    # Create secure temporary directory for extraction
                    temp_extract_dir = os.path.join(output_dir, f"temp_{language_code}_{int(time.time())}")
                    if os.path.exists(temp_extract_dir):
                        shutil.rmtree(temp_extract_dir)
                    os.makedirs(temp_extract_dir, exist_ok=True)

                    # Validate ZIP file before extraction
                    if not zipfile.is_zipfile(zip_path):
                        raise RuntimeError("Downloaded file is not a valid ZIP")

                    with zipfile.ZipFile(zip_path, "r") as zip_ref:
                        # Security check: validate all paths in ZIP
                        for member in zip_ref.namelist():
                            if os.path.isabs(member) or ".." in member:
                                raise SecurityError(f"Unsafe ZIP entry: {member}")
                        zip_ref.extractall(temp_extract_dir)

                    # Process extracted files
                    required_folders = ["am", "conf", "graph", "ivector"]
                    
                    # Prepare clean model directory
                    if os.path.exists(model_path):
                        shutil.rmtree(model_path)
                    os.makedirs(model_path, exist_ok=True)

                    # Find and copy required directories
                    found = {folder: False for folder in required_folders}
                    for root, dirs, files in os.walk(temp_extract_dir):
                        for folder in required_folders:
                            if not found[folder] and folder in dirs:
                                src_path = os.path.join(root, folder)
                                dst_path = os.path.join(model_path, folder)
                                shutil.copytree(src_path, dst_path)
                                found[folder] = True
                        
                        if all(found.values()):
                            break

                    # Verify all directories were found
                    missing = [f for f, ok in found.items() if not ok]
                    if missing:
                        raise RuntimeError(f"Missing model folders: {missing}")

                    # Clean up
                    if os.path.exists(temp_extract_dir):
                        shutil.rmtree(temp_extract_dir)
                    if os.path.exists(zip_path):
                        os.remove(zip_path)

                    logger.info(f"Model extracted successfully: {model_path}")
                    return model_path

                except Exception as e:
                    logger.error(f"Download attempt {attempt + 1} failed: {e}")
                    
                    # Clean up on error
                    for cleanup_path in [temp_extract_dir if 'temp_extract_dir' in locals() else None, zip_path]:
                        if cleanup_path and os.path.exists(cleanup_path):
                            try:
                                if os.path.isdir(cleanup_path):
                                    shutil.rmtree(cleanup_path)
                                else:
                                    os.remove(cleanup_path)
                            except Exception as cleanup_error:
                                logger.warning(f"Cleanup failed: {cleanup_error}")
                    
                    if attempt < 2:  # Not the last attempt
                        time.sleep(2 * (attempt + 1))
                        continue
                    else:
                        raise

            raise RuntimeError("All download attempts failed")
            
        except SecurityError:
            raise  # Re-raise security errors
        except Exception as e:
            logger.error(f"Model download/extraction failed: {e}")
            raise RuntimeError(f"Model setup failed: {str(e)}")

    @staticmethod
    async def download_and_extract_model(url, language_code, output_dir):
        """Async wrapper for model download with validation"""
        try:
            # Validate inputs
            from urllib.parse import urlparse
            parsed = urlparse(url)
            if not parsed.scheme.startswith('http'):
                raise ValueError("Invalid model URL scheme")
            
            if not language_code.isalnum() or len(language_code) > 10:
                raise ValueError("Invalid language code")

            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, 
                FileManager._sync_download_and_extract, 
                url, 
                language_code, 
                output_dir
            )
        except Exception as e:
            logger.error(f"Async model download failed: {e}")
            raise

    @staticmethod
    def cleanup_temp_dirs():
        """Secure cleanup of temporary directories with path validation"""
        try:
            base_temp = FileManager.get_data_path("temp")
            
            if not os.path.exists(base_temp):
                logger.info("No temp directory to clean")
                return
                
            # Validate base_temp is actually our temp directory
            expected_base = Path(__file__).parent.parent.parent / "data" / "temp"
            if not Path(base_temp).resolve().is_relative_to(expected_base.parent.resolve()):
                raise SecurityError("Invalid temp directory for cleanup")
            
            cleaned_count = 0
            error_count = 0
            
            for temp_item in os.listdir(base_temp):
                item_path = os.path.join(base_temp, temp_item)
                
                # Additional security check: only clean items that look like our temp dirs
                if not (temp_item.startswith("temp_") and temp_item.replace("temp_", "").replace("_", "").isdigit()):
                    logger.warning(f"Skipping non-temp item: {temp_item}")
                    continue
                
                try:
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                        cleaned_count += 1
                    elif os.path.isfile(item_path):
                        os.remove(item_path)
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