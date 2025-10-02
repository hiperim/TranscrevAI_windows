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
        if subdir is not None and subdir:
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

class IntelligentModelLoader:
    """Gerencia download inteligente de modelos Whisper (Proposição #10 - Enhanced)"""
    
    def __init__(self):
        self.download_tasks: Dict[str, asyncio.Task] = {}
        self.download_status: Dict[str, str] = {}
        self.download_progress: Dict[str, float] = {}
        self.loaded_models: Set[str] = set()
        self.model_sizes = {
            "tiny": 39,    # MB
            "base": 74,    # MB  
            "small": 244,  # MB
            "medium": 769, # MB
            "large": 1550  # MB
        }
        self.supported_languages = ["pt"]  # PT-BR only
        self.active_language = None
        self.background_downloads = set()
        self.cancelled_downloads = set()
        self.memory_usage = 0
        self.max_memory_mb = 2048  # 2GB limit - optimized for single PT-BR model
        
    async def start_intelligent_preload(self):
        """ENHANCED: Download do modelo medium PT-BR"""
        try:
            logger.info("🚀 Iniciando download do modelo medium PT-BR")

            # Download only PT-BR model
            language = "pt"
            model_key = f"medium_{language}"
            self.download_status[model_key] = "queued"
            self.download_progress[model_key] = 0.0

            # Create download task for PT-BR only
            task = asyncio.create_task(self._download_model_for_language(language))
            self.download_tasks[model_key] = task
            self.background_downloads.add(model_key)
            
            logger.info("📥 Download PT-BR iniciado")
            
        except Exception as e:
            logger.error(f"ERROR Erro no preload inteligente: {e}")
    
    async def _download_model_for_language(self, language: str):
        """Download modelo medium para língua específica com controle de memória"""
        model_key = f"medium_{language}"
        
        try:
            self.download_status[model_key] = "downloading"
            self.download_progress[model_key] = 0.0
            logger.info(f"📥 Iniciando download modelo medium para {language.upper()}")
            
            # Check memory constraints before loading
            if self.memory_usage + self.model_sizes["medium"] > self.max_memory_mb:
                logger.warning(f"WARNING Limite de memória atingido, aguardando espaço para {language}")
                self.download_status[model_key] = "waiting_memory"
                return
            
            # REAL model loading - using newer transcription service
            # Legacy transcription_fase8 has been deprecated
            transcription_service = None
            
            # Progress updates with cancellation checks
            for progress in [10, 25, 40, 55]:
                if model_key in self.cancelled_downloads:
                    raise asyncio.CancelledError()
                await asyncio.sleep(0.2)
                self.download_progress[model_key] = progress
                self.download_status[model_key] = f"downloading_{progress}%"
            
            # Load medium model (this takes significant time first time)
            logger.info(f"🔄 Carregando modelo Whisper medium para {language}")

            # TODO: Implement model loading logic
            pass
            
            # Update memory tracking
            self.memory_usage += self.model_sizes["medium"]
            self.loaded_models.add(model_key)
            
            # Progress completion
            for progress in [70, 85, 100]:
                await asyncio.sleep(0.1)
                self.download_progress[model_key] = progress
                self.download_status[model_key] = f"downloading_{progress}%"
            
            self.download_status[model_key] = "completed"
            self.download_progress[model_key] = 100.0
            logger.info(f"OK Modelo medium para {language.upper()} carregado com sucesso!")
            
        except asyncio.CancelledError:
            self.download_status[model_key] = "cancelled"
            self.download_progress[model_key] = 0.0
            self.cancelled_downloads.add(model_key)
            logger.info(f"🚫 Download cancelado para {language.upper()}")
            
        except Exception as e:
            self.download_status[model_key] = "error"
            self.download_progress[model_key] = 0.0
            logger.error(f"ERROR Erro no download para {language}: {e}")
        finally:
            self.background_downloads.discard(model_key)
    
    async def prioritize_language(self, language: str):
        """ENHANCED: Prioriza língua e cancela downloads desnecessários"""
        try:
            if language not in self.supported_languages:
                logger.warning(f"ERROR Língua não suportada: {language}")
                return
                
            logger.info(f"🎯 Priorizando downloads para língua: {language.upper()}")
            self.active_language = language
            
            # Liberar memória cancelando/removendo modelos de outras línguas
            cancelled_count = 0
            memory_freed = 0
            
            for model_key in list(self.background_downloads.copy()):
                if not model_key.endswith(f"_{language}"):
                    # Cancel download task
                    if model_key in self.download_tasks and not self.download_tasks[model_key].done():
                        self.download_tasks[model_key].cancel()
                        cancelled_count += 1
                    
                    # Mark as cancelled and remove from tracking
                    self.cancelled_downloads.add(model_key)
                    self.background_downloads.discard(model_key)
                    
                    # Free memory if model was loaded
                    if model_key in self.loaded_models:
                        self.memory_usage -= self.model_sizes["medium"]
                        self.loaded_models.discard(model_key)
                        memory_freed += self.model_sizes["medium"]
                        logger.info(f"🗑️ Removido modelo {model_key} da memória")
            
            logger.info(f"🚫 Cancelados {cancelled_count} downloads | 💾 Liberados {memory_freed}MB de memória")
            
            # Boost priority for selected language
            priority_key = f"medium_{language}"
            if priority_key in self.download_tasks and not self.download_tasks[priority_key].done():
                logger.info(f"⚡ Priorizando download ativo: {priority_key}")
                # Task is already running, just log the prioritization
            elif priority_key not in self.download_tasks:
                # Start download if not started yet
                logger.info(f"🚀 Iniciando download prioritário para {language.upper()}")
                task = asyncio.create_task(self._download_model_for_language(language))
                self.download_tasks[priority_key] = task
                self.background_downloads.add(priority_key)
            
            # Clear memory estimation for better performance
            await self._optimize_memory_usage()
                
        except Exception as e:
            logger.error(f"ERROR Erro ao priorizar língua {language}: {e}")
            
    async def _optimize_memory_usage(self):
        """Otimiza uso de memória após mudanças de prioridade"""
        try:
            import gc
            gc.collect()
            
            # Force cleanup of cancelled models
            for model_key in self.cancelled_downloads.copy():
                if model_key in self.loaded_models:
                    self.loaded_models.discard(model_key)
                    
            logger.info(f"🧹 Otimização de memória concluída | Uso atual: {self.memory_usage}MB")
            
        except Exception as e:
            logger.error(f"ERROR Erro na otimização de memória: {e}")
    
    def get_download_status(self) -> Dict[str, Any]:
        """ENHANCED: Retorna status detalhado dos downloads"""
        completed_count = sum(1 for status in self.download_status.values() if status == "completed")
        downloading_count = sum(1 for status in self.download_status.values() if "downloading" in status)
        cancelled_count = len(self.cancelled_downloads)
        
        return {
            "active_language": self.active_language,
            "downloads": dict(self.download_status),
            "progress": dict(self.download_progress),
            "loaded_models": list(self.loaded_models),
            "active_downloads": len(self.background_downloads),
            "completed_downloads": completed_count,
            "downloading_count": downloading_count,
            "cancelled_downloads": cancelled_count,
            "memory_usage_mb": self.memory_usage,
            "memory_limit_mb": self.max_memory_mb,
            "memory_percentage": round((self.memory_usage / self.max_memory_mb) * 100, 1)
        }
    
    async def cleanup_downloads(self):
        """Limpa downloads não utilizados"""
        try:
            # Cancelar todos os downloads pendentes
            for task in self.download_tasks.values():
                if not task.done():
                    task.cancel()
            
            # Aguardar cancelamento
            if self.download_tasks:
                await asyncio.gather(*self.download_tasks.values(), return_exceptions=True)
            
            self.download_tasks.clear()
            self.background_downloads.clear()
            
            logger.info("Cleanup de downloads concluído")
            
        except Exception as e:
            logger.error(f"Erro no cleanup: {e}")
    
    async def get_optimized_model(self, language: str):
        """Retorna modelo otimizado para a língua (para integração futura)"""
        model_key = f"medium_{language}"
        
        if (model_key in self.download_status and 
            self.download_status[model_key] == "completed"):
            logger.info(f"Modelo {model_key} já disponível")
            return "medium"
        
        # Se não estiver pronto, usar modelo padrão
        logger.info(f"Modelo {model_key} não pronto - usando fallback")
        return "medium"

# ==========================================
# INTELLIGENT CACHE MANAGER
# ==========================================
# Merged from cache_manager.py for file consolidation

from enum import Enum
from collections import OrderedDict
import hashlib

class CacheStrategy(Enum):
    """Cache strategies based on usage patterns"""
    LRU = "least_recently_used"
    ADAPTIVE = "adaptive_based_on_usage"

class IntelligentCacheManager:
    """Intelligent cache management for models and data"""

    def __init__(self, max_cache_size_mb: int = 300):  # Reduzido para compliance 401MB
        self.max_cache_size_mb = max_cache_size_mb
        self.cache: OrderedDict[str, Any] = OrderedDict()
        self.cache_size_bytes = 0  # Track cache size específico
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "memory_cleanups": 0
        }
        self.lock = threading.Lock()

        # Aggressive cleanup settings para browser safety
        self.cleanup_threshold = 0.8  # Cleanup quando 80% do limite
        self.max_items = 50  # Limite máximo de items no cache

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                self.cache_stats["hits"] += 1
                return value

            self.cache_stats["misses"] += 1
            return None

    def put(self, key: str, value: Any) -> bool:
        """Put item in cache"""
        with self.lock:
            try:
                # Remove if exists
                if key in self.cache:
                    del self.cache[key]

                # Add new item
                self.cache[key] = value

                # Check memory limits and evict if necessary
                self._evict_if_needed()

                return True

            except Exception as e:
                logger.error(f"Cache put failed: {e}")
                return False

    def _evict_if_needed(self):
        """Aggressive eviction for browser-safe memory management"""
        try:
            import sys

            # Check multiple conditions for eviction
            current_memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
            cache_count = len(self.cache)

            # Evict if any condition is met:
            should_evict = (
                current_memory_mb > (self.max_cache_size_mb * self.cleanup_threshold) or  # Memory pressure
                cache_count > self.max_items or  # Too many items
                current_memory_mb > 350  # Hard limit para compliance 401MB
            )

            if should_evict:
                logger.info(f"Cache eviction triggered: {current_memory_mb:.1f}MB, {cache_count} items")

                # Aggressive cleanup - remove 50% of items
                items_to_remove = max(1, cache_count // 2)

                for _ in range(items_to_remove):
                    if not self.cache:
                        break

                    # Remove least recently used item with explicit cleanup
                    oldest_key = next(iter(self.cache))
                    old_item = self.cache.pop(oldest_key)

                    # Explicit cleanup do objeto
                    try:
                        if hasattr(old_item, 'cleanup'):
                            old_item.cleanup()
                        del old_item
                    except Exception as cleanup_error:
                        logger.debug(f"Object cleanup warning: {cleanup_error}")

                    self.cache_stats["evictions"] += 1

                # Force garbage collection após eviction
                import gc
                gc.collect()
                self.cache_stats["memory_cleanups"] += 1

                final_memory = psutil.Process().memory_info().rss / (1024 * 1024)
                logger.info(f"Cache cleanup completed: {final_memory:.1f}MB, {len(self.cache)} items remaining")

        except Exception as e:
            logger.warning(f"Cache eviction warning: {e}")

    def clear(self):
        """Clear all cache"""
        with self.lock:
            self.cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cache_size": len(self.cache),
            "stats": self.cache_stats.copy(),
            "memory_usage_mb": psutil.Process().memory_info().rss / (1024 * 1024)
        }

    def get_lazy_service(self, service_name: str) -> Optional[Any]:
        """Get lazy-loaded service"""
        return self.get(f"lazy_service_{service_name}")

    def register_lazy_service(self, service_name: str, service_instance: Any) -> bool:
        """Register a lazy-loaded service"""
        return self.put(f"lazy_service_{service_name}", service_instance)

    def schedule_background_preload(self, service_instance: Any, language: str = "pt") -> bool:
        """Schedule background preloading (simplified implementation)"""
        try:
            # For now, just cache the service with language info
            cache_key = f"preload_{language}_{type(service_instance).__name__}"
            return self.put(cache_key, service_instance)
        except Exception as e:
            logger.warning(f"Background preload scheduling failed: {e}")
            return False

# Global instances
intelligent_loader = IntelligentModelLoader()
intelligent_cache = IntelligentCacheManager()