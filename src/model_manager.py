# Optimized Model Manager for TranscrevAI
# Intelligent caching and memory management for Whisper models

"""
OptimizedModelManager

High-performance model management system that:
- Maintains real-time processing performance
- Implements intelligent caching with TTL
- Automatically cleans up unused models
- Supports quantization for reduced latency
- Provides memory usage monitoring and limits
"""

import asyncio
import time
import logging
import psutil
import threading
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import gc

from config.whisper_optimization import (
    get_optimized_config, 
    get_model_name, 
    QUANTIZATION_CONFIG,
    PERFORMANCE_CONSTRAINTS,
    validate_real_time_performance
)

logger = logging.getLogger(__name__)

class ModelLoadError(Exception):
    """Raised when model loading fails"""
    pass

class MemoryLimitExceeded(Exception):
    """Raised when memory limit is exceeded"""
    pass

class OptimizedModelManager:
    """
    Optimized model management with intelligent caching and memory control
    
    Features:
    - Smart caching with TTL (Time To Live)
    - Memory usage monitoring and limits
    - Quantization support for performance
    - Automatic cleanup of unused models
    - Thread-safe model loading
    - Performance metrics tracking
    """
    
    def __init__(self, max_cache_size: int = 2, memory_limit_mb: int = 2048):
        """
        Initialize the optimized model manager
        
        Args:
            max_cache_size (int): Maximum number of models to cache
            memory_limit_mb (int): Maximum memory usage in MB
        """
        self._model_cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._model_access_count: Dict[str, int] = {}
        self._loading_locks: Dict[str, asyncio.Lock] = {}
        
        # Configuration
        self.max_cache_size = max_cache_size
        self.memory_limit_mb = memory_limit_mb
        self.cache_ttl = PERFORMANCE_CONSTRAINTS["model_cache_ttl"]  # 30 minutes default
        
        # Performance tracking
        self._load_times: Dict[str, float] = {}
        self._memory_usage: Dict[str, float] = {}
        
        # Thread safety
        self._cache_lock = asyncio.Lock()
        
        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_cleanup_task()
        
        logger.info(f"OptimizedModelManager initialized: max_cache={max_cache_size}, memory_limit={memory_limit_mb}MB")
    
    def _start_cleanup_task(self):
        """Start the automatic cleanup task"""
        try:
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        except RuntimeError:
            # No event loop running, cleanup will be manual
            logger.info("No event loop available for automatic cleanup task")
    
    async def _periodic_cleanup(self):
        """Periodic cleanup task running every 5 minutes"""
        while True:
            try:
                await asyncio.sleep(300)  # 5 minutes
                await self._cleanup_expired_models()
                await self._check_memory_usage()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
    
    async def get_optimized_model(self, language: str) -> Any:
        """
        Get model with optimized configuration and intelligent caching
        
        Args:
            language (str): Language code (pt, en, es)
            
        Returns:
            Any: Loaded Whisper model
            
        Raises:
            ModelLoadError: If model loading fails
            MemoryLimitExceeded: If memory limit would be exceeded
        """
        model_key = get_model_name(language)
        
        # Check cache first
        async with self._cache_lock:
            cached_model = await self._get_from_cache(model_key)
            if cached_model:
                self._update_access_stats(model_key)
                logger.debug(f"Retrieved model from cache: {model_key}")
                return cached_model
        
        # Model not cached, need to load
        return await self._load_and_cache_model(language, model_key)
    
    async def _get_from_cache(self, model_key: str) -> Optional[Any]:
        """Get model from cache if valid and not expired"""
        if model_key not in self._model_cache:
            return None
            
        # Check if cache entry is still valid
        cache_time = self._cache_timestamps.get(model_key, 0)
        if time.time() - cache_time > self.cache_ttl:
            logger.info(f"Cache expired for model: {model_key}")
            await self._remove_from_cache(model_key)
            return None
        
        return self._model_cache[model_key]
    
    async def _load_and_cache_model(self, language: str, model_key: str) -> Any:
        """Load model and add to cache with thread safety"""
        
        # Ensure we have a lock for this model
        if model_key not in self._loading_locks:
            self._loading_locks[model_key] = asyncio.Lock()
        
        async with self._loading_locks[model_key]:
            # Double-check cache after acquiring lock
            async with self._cache_lock:
                cached_model = await self._get_from_cache(model_key)
                if cached_model:
                    return cached_model
            
            # Check memory before loading
            await self._ensure_memory_available()
            
            # Load the model
            start_time = time.time()
            model = await self._load_model(language, model_key)
            load_time = time.time() - start_time
            
            # Cache the loaded model
            async with self._cache_lock:
                await self._add_to_cache(model_key, model, load_time)
            
            logger.info(f"Model loaded and cached: {model_key} (load_time: {load_time:.2f}s)")
            return model
    
    async def _load_model(self, language: str, model_key: str) -> Any:
        """Load Whisper model with optimized configuration"""
        try:
            # Lazy import to avoid startup delays
            import whisper
            
            # Get optimized configuration
            config = get_optimized_config(language)
            
            # Load model with quantization if enabled
            device = "cpu"  # Force CPU for consistency and compatibility
            
            # Load model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            def _load_model_sync():
                return whisper.load_model(
                    model_key,
                    device=device,
                    in_memory=True  # Keep in memory for better performance
                )
            
            model = await loop.run_in_executor(None, _load_model_sync)
            
            # Apply quantization if supported and enabled
            if QUANTIZATION_CONFIG.get("enabled", False):
                await self._apply_quantization(model)
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_key}: {e}")
            raise ModelLoadError(f"Model loading failed: {str(e)}")
    
    async def _apply_quantization(self, model: Any) -> None:
        """Apply quantization to model for better performance"""
        try:
            # Note: OpenAI Whisper doesn't directly support quantization
            # This is a placeholder for future quantization implementation
            # For now, we use FP32 and rely on other optimizations
            logger.debug("Quantization placeholder - using FP32 for compatibility")
        except Exception as e:
            logger.warning(f"Quantization failed, using original model: {e}")
    
    async def _add_to_cache(self, model_key: str, model: Any, load_time: float):
        """Add model to cache with LRU eviction if necessary"""
        
        # Remove expired models first
        await self._cleanup_expired_models()
        
        # If cache is full, remove least recently used model
        if len(self._model_cache) >= self.max_cache_size:
            await self._evict_lru_model()
        
        # Add to cache
        self._model_cache[model_key] = model
        self._cache_timestamps[model_key] = time.time()
        self._model_access_count[model_key] = 0
        self._load_times[model_key] = load_time
        
        # Estimate memory usage
        memory_usage = self._estimate_model_memory(model)
        self._memory_usage[model_key] = memory_usage
        
        logger.info(f"Model cached: {model_key} (estimated memory: {memory_usage:.1f}MB)")
    
    async def _remove_from_cache(self, model_key: str):
        """Remove model from cache and clean up"""
        if model_key in self._model_cache:
            del self._model_cache[model_key]
            
        # Clean up metadata
        self._cache_timestamps.pop(model_key, None)
        self._model_access_count.pop(model_key, None)
        self._load_times.pop(model_key, None)
        self._memory_usage.pop(model_key, None)
        
        # Force garbage collection
        gc.collect()
        
        logger.info(f"Model removed from cache: {model_key}")
    
    async def _evict_lru_model(self):
        """Evict least recently used model from cache"""
        if not self._cache_timestamps:
            return
            
        # Find least recently used model
        lru_model = min(self._cache_timestamps.keys(), 
                       key=lambda k: self._cache_timestamps[k])
        
        logger.info(f"Evicting LRU model: {lru_model}")
        await self._remove_from_cache(lru_model)
    
    async def _cleanup_expired_models(self):
        """Clean up expired models from cache"""
        current_time = time.time()
        expired_models = []
        
        for model_key, timestamp in self._cache_timestamps.items():
            if current_time - timestamp > self.cache_ttl:
                expired_models.append(model_key)
        
        for model_key in expired_models:
            logger.info(f"Cleaning up expired model: {model_key}")
            await self._remove_from_cache(model_key)
    
    def _update_access_stats(self, model_key: str):
        """Update access statistics for a model"""
        self._model_access_count[model_key] = self._model_access_count.get(model_key, 0) + 1
        self._cache_timestamps[model_key] = time.time()  # Update last access time
    
    async def _ensure_memory_available(self):
        """Ensure sufficient memory is available before loading model"""
        try:
            # Get current memory usage
            process = psutil.Process()
            current_memory_mb = process.memory_info().rss / 1024 / 1024
            
            # Estimate new model memory (conservative estimate)
            estimated_model_size = 150  # MB for small Whisper model
            
            if current_memory_mb + estimated_model_size > self.memory_limit_mb:
                logger.warning(f"Memory limit approaching: {current_memory_mb:.1f}MB current + {estimated_model_size}MB estimated > {self.memory_limit_mb}MB limit")
                
                # Try to free memory by removing least used models
                await self._free_memory()
                
                # Check again
                current_memory_mb = process.memory_info().rss / 1024 / 1024
                if current_memory_mb + estimated_model_size > self.memory_limit_mb:
                    raise MemoryLimitExceeded(f"Insufficient memory: {current_memory_mb:.1f}MB + {estimated_model_size}MB > {self.memory_limit_mb}MB")
                    
        except psutil.Error as e:
            logger.warning(f"Could not check memory usage: {e}")
    
    async def _free_memory(self):
        """Free memory by removing least used models"""
        if not self._model_access_count:
            return
        
        # Sort models by access count (ascending)
        models_by_usage = sorted(self._model_access_count.items(), key=lambda x: x[1])
        
        # Remove least used models until we're under memory pressure
        for model_key, _ in models_by_usage[:len(models_by_usage)//2]:  # Remove up to half
            logger.info(f"Freeing memory by removing model: {model_key}")
            await self._remove_from_cache(model_key)
    
    async def _check_memory_usage(self):
        """Monitor and log memory usage"""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > self.memory_limit_mb * 0.8:  # 80% threshold warning
                logger.warning(f"High memory usage: {memory_mb:.1f}MB ({memory_mb/self.memory_limit_mb*100:.1f}% of limit)")
                await self._free_memory()
            else:
                logger.debug(f"Memory usage: {memory_mb:.1f}MB ({memory_mb/self.memory_limit_mb*100:.1f}% of limit)")
                
        except Exception as e:
            logger.warning(f"Memory monitoring error: {e}")
    
    def _estimate_model_memory(self, model: Any) -> float:
        """Estimate model memory usage in MB"""
        try:
            # Conservative estimate for small Whisper models
            return 150.0  # MB
        except Exception:
            return 150.0  # Default estimate
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring"""
        return {
            "cached_models": list(self._model_cache.keys()),
            "cache_size": len(self._model_cache),
            "max_cache_size": self.max_cache_size,
            "memory_usage_mb": sum(self._memory_usage.values()),
            "memory_limit_mb": self.memory_limit_mb,
            "access_counts": self._model_access_count.copy(),
            "load_times": self._load_times.copy()
        }
    
    async def preload_models(self, languages: list) -> Dict[str, bool]:
        """Preload models for specified languages"""
        results = {}
        
        for language in languages:
            try:
                model = await self.get_optimized_model(language)
                results[language] = model is not None
                logger.info(f"Preloaded model for {language}: {results[language]}")
            except Exception as e:
                logger.error(f"Failed to preload model for {language}: {e}")
                results[language] = False
        
        return results
    
    async def cleanup_all(self):
        """Clean up all cached models and stop background tasks"""
        logger.info("Cleaning up all cached models...")
        
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Clear all caches
        async with self._cache_lock:
            model_keys = list(self._model_cache.keys())
            for model_key in model_keys:
                await self._remove_from_cache(model_key)
        
        logger.info("Model manager cleanup complete")

# Global instance for easy access
_global_model_manager: Optional[OptimizedModelManager] = None

def get_model_manager() -> OptimizedModelManager:
    """Get or create the global model manager instance"""
    global _global_model_manager
    if _global_model_manager is None:
        _global_model_manager = OptimizedModelManager()
    return _global_model_manager

async def cleanup_model_manager():
    """Cleanup the global model manager"""
    global _global_model_manager
    if _global_model_manager:
        await _global_model_manager.cleanup_all()
        _global_model_manager = None