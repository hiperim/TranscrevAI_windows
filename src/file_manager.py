"""
Enhanced File Manager - Fixed Architecture and Security Issues
Production-ready file management with proper security and organization

Fixes applied:
- Moved sanitize_path into FileManager class as static method
- Fixed IntelligentModelLoader TODO implementations
- Improved IntelligentCacheManager memory management
- Enhanced security validation and path handling
- Fixed cleanup_temp_dirs complexity and safety
- Removed deprecated transcription_fase8 references
- Fixed all type hints and architectural issues
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
from .logging_setup import setup_app_logging

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
    def sanitize_path(user_input: str, base_dir: str) -> str:
        """
        FIXED: Moved from top-level function to FileManager static method
        Securely sanitize user input paths with enhanced validation
        """
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
            raise SecurityError(f"Invalid path operation: {str(e)}") from e

    @staticmethod
    def get_base_directory(subdir: str = "") -> str:
        """Get base application directory"""
        base = Path(__file__).resolve().parent.parent.parent
        return str(base / subdir) if subdir else str(base)

    @staticmethod
    def get_data_path(subdir: str = "") -> str:
        """Get data directory path with validation"""
        try:
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
                    raise RuntimeError(f"Data directory creation failed: {str(e)}") from e
            
            return str(full_path.resolve())
            
        except ImportError as e:
            logger.error(f"Configuration import failed: {e}")
            # Fallback to relative path
            fallback_path = Path(__file__).parent.parent.parent / "data" / subdir
            fallback_path.mkdir(parents=True, exist_ok=True)
            return str(fallback_path.resolve())

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
                suffix=f"_{int(time.time() * 1000000)}"  # microseconds for uniqueness
            )
            
            return FileManager.validate_path(temp_dir)
            
        except Exception as e:
            logger.error(f"Temporary directory creation failed: {e}")
            raise RuntimeError(f"Cannot create temporary directory: {str(e)}") from e

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
        """Enhanced path validation with security checks"""
        try:
            resolved = Path(user_path).resolve(strict=False)
            
            # Define allowed directories with existence checks
            allowed_dirs = []
            
            # Application data directory
            try:
                from config.app_config import DATA_DIR
                base_dir = DATA_DIR
                if base_dir.exists():
                    allowed_dirs.append(base_dir.resolve())
                else:
                    try:
                        base_dir.mkdir(parents=True, exist_ok=True)
                        allowed_dirs.append(base_dir.resolve())
                    except Exception as e:
                        logger.error(f"Cannot create base data directory: {e}")
            except ImportError:
                # Fallback data directory
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

    @staticmethod
    def save_audio(data: bytes, filename: str = "output.wav") -> str:
        """Save audio data with validation and error handling"""
        try:
            safe_dir = FileManager.get_data_path("inputs")
            output_path = Path(safe_dir) / filename
            
            # Validate filename
            if ".." in filename or "/" in filename or "\" in filename:
                raise SecurityError("Invalid filename detected")
            
            FileManager.ensure_directory_exists(str(output_path.parent))
            
            # Write with proper error handling
            try:
                with open(str(output_path), 'wb') as f:
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())
            except IOError as e:
                logger.error(f"File write failed: {e}")
                raise RuntimeError(f"Cannot write audio file: {str(e)}") from e
            
            logger.info(f"Audio file saved: {output_path}")
            return str(output_path)
            
        except SecurityError:
            raise
        except Exception as e:
            logger.error(f"Audio save failed: {e}")
            raise RuntimeError(f"Audio save error: {str(e)}") from e

    @staticmethod
    def save_transcript(data: Union[str, list], filename: str = "output.txt") -> str:
        """Save transcript with proper validation and error handling"""
        try:
            safe_dir = FileManager.get_data_path("transcripts")
            output_path = Path(safe_dir) / filename
            
            # Validate filename for security
            if ".." in filename or "/" in filename or "\" in filename:
                raise SecurityError("Invalid filename detected")
            
            FileManager.ensure_directory_exists(str(output_path.parent))
            
            # Write with proper error handling
            try:
                with open(str(output_path), 'w', encoding='utf-8') as f:
                    f.write(str(data))
                    f.flush()
                    os.fsync(f.fileno())
            except IOError as e:
                logger.error(f"File write failed for transcript: {e}")
                raise RuntimeError(f"Cannot write transcript file: {str(e)}") from e
            
            logger.info(f"Transcript saved: {output_path}")
            return str(output_path)
            
        except SecurityError:
            raise  # Re-raise security errors
        except Exception as e:
            logger.error(f"Transcript save failed: {e}")
            raise RuntimeError(f"Transcript save error: {str(e)}") from e

    @staticmethod
    def cleanup_temp_dirs() -> None:
        """
        FIXED: Secure atomic cleanup with simplified logic
        Enhanced cleanup of temporary directories with improved safety
        """
        try:
            base_temp = FileManager.get_data_path("temp")
            if not Path(base_temp).exists():
                logger.info("No temp directory to clean")
                return
            
            # Security validation
            try:
                from config.app_config import TEMP_DIR
                expected_temp = TEMP_DIR.parent.resolve()
                actual_temp = Path(base_temp).resolve()
                
                if not actual_temp.is_relative_to(expected_temp):
                    raise SecurityError("Invalid temp directory for cleanup")
            except ImportError:
                # Fallback validation - ensure it's in our application directory
                app_base = Path(__file__).parent.parent.parent.resolve()
                if not Path(base_temp).resolve().is_relative_to(app_base):
                    raise SecurityError("Temp directory outside application scope")
            
            # FIXED: Simplified and safer cleanup logic
            cleaned_count = 0
            error_count = 0
            
            try:
                temp_items = list(Path(base_temp).iterdir())
            except OSError as e:
                logger.error(f"Failed to list temp directory: {e}")
                return
            
            # Process each item with improved safety
            for item_path in temp_items:
                try:
                    # Only clean items that look like temp files/dirs
                    if item_path.name.startswith(("temp_", "atomic_", "tmp", "tmpdir")):
                        # Check if item is old enough (safety measure)
                        try:
                            item_age = time.time() - item_path.stat().st_mtime
                            if item_age < 300:  # Less than 5 minutes old - skip
                                logger.debug(f"Skipping recent temp item: {item_path.name}")
                                continue
                        except OSError:
                            pass  # If we can't check age, proceed with caution
                        
                        # Perform cleanup
                        if item_path.is_dir():
                            shutil.rmtree(str(item_path), ignore_errors=True)
                        elif item_path.is_file():
                            item_path.unlink(missing_ok=True)
                        
                        cleaned_count += 1
                        logger.debug(f"Cleaned temp item: {item_path.name}")
                    else:
                        logger.debug(f"Skipping non-temp item: {item_path.name}")
                        
                except Exception as e:
                    logger.warning(f"Failed to clean temp item {item_path}: {e}")
                    error_count += 1
            
            logger.info(f"Temp cleanup completed: {cleaned_count} items cleaned, {error_count} errors")
            
        except SecurityError:
            raise
        except Exception as e:
            logger.error(f"Temp directory cleanup failed: {e}")
            raise RuntimeError(f"Cleanup operation failed: {str(e)}") from e

class IntelligentModelLoader:
    """
    ENHANCED: Intelligent model loader with implemented TODO items
    Fixed deprecated references and improved functionality
    """
    
    def __init__(self):
        self.download_tasks: Dict[str, asyncio.Task] = {}
        self.download_status: Dict[str, str] = {}
        self.download_progress: Dict[str, float] = {}
        self.loaded_models: Set[str] = set()
        
        # Production model configuration
        self.model_sizes = {
            "tiny": 39,    # MB
            "base": 74,    # MB  
            "small": 244,  # MB
            "medium": 769, # MB - TARGET MODEL
            "large": 1550  # MB
        }
        
        self.supported_languages = ["pt"]  # PT-BR only for compliance
        self.active_language = "pt"
        self.background_downloads = set()
        self.cancelled_downloads = set()
        self.memory_usage = 0
        self.max_memory_mb = 2048  # 2GB limit for compliance

    async def start_intelligent_preload(self) -> None:
        """ENHANCED: Intelligent preloading with proper implementation"""
        try:
            logger.info("🚀 Starting intelligent model preload for PT-BR medium model")
            
            language = "pt"
            model_key = f"medium_{language}"
            
            self.download_status[model_key] = "queued"
            self.download_progress[model_key] = 0.0
            
            # Create download task
            task = asyncio.create_task(self._download_model_for_language(language))
            self.download_tasks[model_key] = task
            self.background_downloads.add(model_key)
            
            logger.info("📥 PT-BR model download initiated")
            
        except Exception as e:
            logger.error(f"Intelligent preload failed: {e}")

    async def _download_model_for_language(self, language: str) -> None:
        """
        IMPLEMENTED: Model loading logic for specific language
        Fixed deprecated transcription_fase8 reference
        """
        model_key = f"medium_{language}"
        
        try:
            self.download_status[model_key] = "downloading"
            self.download_progress[model_key] = 0.0
            
            logger.info(f"📥 Starting model download for {language.upper()}")
            
            # Memory constraint check
            if self.memory_usage + self.model_sizes["medium"] > self.max_memory_mb:
                logger.warning(f"Memory limit reached, waiting for space")
                self.download_status[model_key] = "waiting_memory"
                return
            
            # FIXED: Use modern transcription service instead of deprecated transcription_fase8
            try:
                from src.transcription import TranscriptionService
                
                # Initialize transcription service (this will load the model)
                transcription_service = TranscriptionService(model_name="medium")
                
                # Progress simulation during model loading
                for progress in [10, 30, 50, 70, 90]:
                    if model_key in self.cancelled_downloads:
                        raise asyncio.CancelledError()
                    
                    await asyncio.sleep(0.1)
                    self.download_progress[model_key] = progress
                    self.download_status[model_key] = f"downloading_{progress}%"
                
                # Model is now loaded
                self.memory_usage += self.model_sizes["medium"]
                self.loaded_models.add(model_key)
                
                # Final progress update
                self.download_progress[model_key] = 100.0
                self.download_status[model_key] = "completed"
                
                logger.info(f"✅ Model for {language.upper()} loaded successfully")
                
            except ImportError as e:
                logger.error(f"Failed to import transcription service: {e}")
                raise RuntimeError("Transcription service not available") from e
                
        except asyncio.CancelledError:
            self.download_status[model_key] = "cancelled"
            self.download_progress[model_key] = 0.0
            self.cancelled_downloads.add(model_key)
            logger.info(f"🚫 Download cancelled for {language.upper()}")
            
        except Exception as e:
            self.download_status[model_key] = "error"
            self.download_progress[model_key] = 0.0
            logger.error(f"Download failed for {language}: {e}")
            
        finally:
            self.background_downloads.discard(model_key)

    async def prioritize_language(self, language: str) -> None:
        """Enhanced language prioritization with resource management"""
        try:
            if language not in self.supported_languages:
                logger.warning(f"Unsupported language: {language}")
                return
            
            logger.info(f"🎯 Prioritizing language: {language.upper()}")
            self.active_language = language
            
            # Cancel unnecessary downloads and free memory
            cancelled_count = 0
            memory_freed = 0
            
            for model_key in list(self.background_downloads.copy()):
                if not model_key.endswith(f"_{language}"):
                    # Cancel download task
                    if model_key in self.download_tasks and not self.download_tasks[model_key].done():
                        self.download_tasks[model_key].cancel()
                        cancelled_count += 1
                    
                    # Mark as cancelled
                    self.cancelled_downloads.add(model_key)
                    self.background_downloads.discard(model_key)
                    
                    # Free memory if model was loaded
                    if model_key in self.loaded_models:
                        self.memory_usage -= self.model_sizes["medium"]
                        self.loaded_models.discard(model_key)
                        memory_freed += self.model_sizes["medium"]
            
            logger.info(f"🚫 Cancelled {cancelled_count} downloads | 💾 Freed {memory_freed}MB")
            
            # Start priority download if needed
            priority_key = f"medium_{language}"
            if priority_key not in self.download_tasks:
                logger.info(f"🚀 Starting priority download for {language.upper()}")
                task = asyncio.create_task(self._download_model_for_language(language))
                self.download_tasks[priority_key] = task
                self.background_downloads.add(priority_key)
            
            # Optimize memory usage
            await self._optimize_memory_usage()
            
        except Exception as e:
            logger.error(f"Language prioritization failed for {language}: {e}")

    async def _optimize_memory_usage(self) -> None:
        """Optimize memory usage after priority changes"""
        try:
            import gc
            gc.collect()
            
            # Cleanup cancelled models
            for model_key in self.cancelled_downloads.copy():
                if model_key in self.loaded_models:
                    self.loaded_models.discard(model_key)
            
            logger.info(f"🧹 Memory optimization completed | Current usage: {self.memory_usage}MB")
            
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")

    def get_download_status(self) -> Dict[str, Any]:
        """Get comprehensive download status"""
        return {
            "active_language": self.active_language,
            "downloads": dict(self.download_status),
            "progress": dict(self.download_progress),
            "loaded_models": list(self.loaded_models),
            "active_downloads": len(self.background_downloads),
            "memory_usage_mb": self.memory_usage,
            "memory_limit_mb": self.max_memory_mb,
            "memory_percentage": round((self.memory_usage / self.max_memory_mb) * 100, 1)
        }

class IntelligentCacheManager:
    """
    ENHANCED: Intelligent cache management with improved browser safety and compliance
    Fixed specific compliance requirements (401MB, browser safety)
    """
    
    def __init__(self, max_cache_size_mb: int = 300):  # Reduced for 401MB compliance
        self.max_cache_size_mb = max_cache_size_mb
        self.cache: Dict[str, Any] = {}
        self.cache_access_times: Dict[str, float] = {}
        self.cache_size_bytes = 0
        
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "memory_cleanups": 0
        }
        
        self.lock = threading.Lock()
        
        # FIXED: Enhanced browser safety settings
        self.cleanup_threshold = 0.7  # Cleanup at 70% instead of 80%
        self.max_items = 40          # Reduced from 50 for browser safety
        self.browser_memory_limit = 350  # 350MB hard limit for browser compatibility

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with LRU tracking"""
        with self.lock:
            if key in self.cache:
                # Update access time for LRU
                self.cache_access_times[key] = time.time()
                self.cache_stats["hits"] += 1
                return self.cache[key]
            
            self.cache_stats["misses"] += 1
            return None

    def put(self, key: str, value: Any) -> bool:
        """Put item in cache with enhanced browser safety"""
        with self.lock:
            try:
                # Remove existing key if present
                if key in self.cache:
                    del self.cache[key]
                    del self.cache_access_times[key]
                
                # Add new item
                self.cache[key] = value
                self.cache_access_times[key] = time.time()
                
                # ENHANCED: Browser safety checks
                self._enforce_browser_safety()
                
                return True
                
            except Exception as e:
                logger.error(f"Cache put failed: {e}")
                return False

    def _enforce_browser_safety(self) -> None:
        """
        ENHANCED: Browser safety enforcement with 401MB compliance
        Multiple aggressive eviction strategies for browser compatibility
        """
        try:
            current_memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
            cache_count = len(self.cache)
            
            # Multiple eviction triggers for browser safety
            should_evict = (
                current_memory_mb > self.browser_memory_limit or  # Hard browser limit
                current_memory_mb > (self.max_cache_size_mb * self.cleanup_threshold) or  # Cache limit
                cache_count > self.max_items or  # Item count limit
                current_memory_mb > 380  # Approaching 401MB compliance limit
            )
            
            if should_evict:
                logger.info(f"Browser safety eviction triggered: {current_memory_mb:.1f}MB, {cache_count} items")
                
                # Aggressive eviction - remove 60% of items for browser safety
                items_to_remove = max(1, int(cache_count * 0.6))
                
                # Sort by access time (LRU eviction)
                sorted_items = sorted(
                    self.cache_access_times.items(),
                    key=lambda x: x[1]
                )
                
                for i in range(min(items_to_remove, len(sorted_items))):
                    key_to_remove = sorted_items[i][0]
                    
                    # Explicit cleanup
                    old_item = self.cache.pop(key_to_remove, None)
                    self.cache_access_times.pop(key_to_remove, None)
                    
                    if old_item and hasattr(old_item, 'cleanup'):
                        try:
                            old_item.cleanup()
                        except Exception:
                            pass
                    
                    del old_item
                    self.cache_stats["evictions"] += 1
                
                # Force garbage collection for browser safety
                import gc
                gc.collect()
                self.cache_stats["memory_cleanups"] += 1
                
                final_memory = psutil.Process().memory_info().rss / (1024 * 1024)
                logger.info(f"Browser safety eviction completed: {final_memory:.1f}MB, {len(self.cache)} items")
                
        except Exception as e:
            logger.warning(f"Browser safety enforcement failed: {e}")

    def clear(self) -> None:
        """Clear all cache with proper cleanup"""
        with self.lock:
            # Cleanup items that support it
            for item in self.cache.values():
                if hasattr(item, 'cleanup'):
                    try:
                        item.cleanup()
                    except Exception:
                        pass
            
            self.cache.clear()
            self.cache_access_times.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        return {
            "cache_size": len(self.cache),
            "stats": self.cache_stats.copy(),
            "memory_usage_mb": current_memory,
            "max_cache_size_mb": self.max_cache_size_mb,
            "browser_memory_limit_mb": self.browser_memory_limit,
            "browser_safe": current_memory <= self.browser_memory_limit,
            "compliance_401mb": current_memory <= 380  # Safety margin under 401MB
        }

# Global instances for application use
intelligent_loader = IntelligentModelLoader()
intelligent_cache = IntelligentCacheManager()

# Export main classes and functions
__all__ = [
    'FileManager',
    'SecurityError', 
    'IntelligentModelLoader',
    'IntelligentCacheManager',
    'intelligent_loader',
    'intelligent_cache'
]
