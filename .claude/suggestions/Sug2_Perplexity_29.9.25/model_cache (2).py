"""
TranscrevAI Optimized - Intelligent Model Cache System
Sistema avançado de cache com lazy loading, TTL e memory pressure coordination
"""

import asyncio
import gc
import hashlib
import os
import pickle
import threading
import time
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
import logging

logger = logging.getLogger("transcrevai.model_cache")

# Lazy imports for heavy dependencies
_whisper = None
_torch = None


def get_whisper():
    """Lazy import whisper"""
    global _whisper
    if _whisper is None:
        try:
            import whisper
            _whisper = whisper
            logger.info("OpenAI Whisper loaded for caching")
        except ImportError as e:
            logger.error(f"Failed to import whisper: {e}")
            _whisper = None
    return _whisper


def get_torch():
    """Lazy import torch"""
    global _torch
    if _torch is None:
        try:
            import torch
            _torch = torch
            logger.info("PyTorch loaded for model caching")
        except ImportError as e:
            logger.error(f"Failed to import torch: {e}")
            _torch = None
    return _torch


@dataclass
class CachedModel:
    """Cached model information"""
    model_id: str
    model_name: str
    model_type: str  # 'whisper', 'encoder', 'decoder'
    model_object: Any
    memory_mb: float
    load_time: float
    last_access: float
    access_count: int
    ttl_hours: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def age_hours(self) -> float:
        """Age of cached model in hours"""
        return (time.time() - self.load_time) / 3600
    
    @property
    def idle_hours(self) -> float:
        """Hours since last access"""
        return (time.time() - self.last_access) / 3600
    
    @property
    def is_expired(self) -> bool:
        """Check if model is expired based on TTL"""
        return self.age_hours > self.ttl_hours
    
    def touch(self):
        """Update last access time"""
        self.last_access = time.time()
        self.access_count += 1


@dataclass
class CacheStats:
    """Cache statistics"""
    cached_models_count: int
    total_memory_mb: float
    hit_count: int
    miss_count: int
    eviction_count: int
    load_count: int
    cache_hit_ratio: float
    average_load_time: float
    oldest_model_hours: float
    memory_pressure_cleanups: int
    

class IntelligentModelCache:
    """
    Advanced model caching system with:
    - Lazy loading and unloading
    - TTL-based expiration
    - LRU eviction policy  
    - Memory pressure coordination
    - Browser-safe loading patterns
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache_dir = Path(config.get("cache_dir", "cache/models"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache configuration
        self.enable_cache = config.get("enable_cache", True)
        self.max_cached_models = config.get("max_cached_models", 3)
        self.default_ttl_hours = config.get("cache_ttl_hours", 24)
        self.lazy_loading = config.get("lazy_loading", True)
        self.memory_pressure_unload = config.get("memory_pressure_unload", True)
        
        # Cache storage
        self.cached_models: Dict[str, CachedModel] = {}
        self.load_futures: Dict[str, asyncio.Future] = {}  # Prevent duplicate loads
        
        # Statistics
        self.stats = CacheStats(
            cached_models_count=0,
            total_memory_mb=0.0,
            hit_count=0,
            miss_count=0,
            eviction_count=0,
            load_count=0,
            cache_hit_ratio=0.0,
            average_load_time=0.0,
            oldest_model_hours=0.0,
            memory_pressure_cleanups=0
        )
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Resource manager integration
        self.resource_manager = None
        try:
            from resource_manager import get_resource_manager
            self.resource_manager = get_resource_manager()
        except ImportError:
            logger.warning("Resource manager not available for model cache")
        
        logger.info(f"IntelligentModelCache initialized (enabled: {self.enable_cache})")
    
    async def get_model(self, 
                       model_name: str, 
                       model_type: str = "whisper",
                       **load_kwargs) -> Optional[Any]:
        """
        Get model from cache or load it
        
        Args:
            model_name: Name/identifier of the model
            model_type: Type of model ('whisper', 'encoder', 'decoder')
            **load_kwargs: Additional arguments for model loading
            
        Returns:
            Model object or None if loading failed
        """
        if not self.enable_cache:
            return await self._load_model_direct(model_name, model_type, **load_kwargs)
        
        model_id = self._generate_model_id(model_name, model_type, load_kwargs)
        
        with self._lock:
            # Check if model is already cached
            if model_id in self.cached_models:
                cached_model = self.cached_models[model_id]
                
                # Check if expired
                if cached_model.is_expired:
                    logger.info(f"Cached model expired: {model_name}")
                    await self._evict_model(model_id)
                else:
                    # Cache hit
                    cached_model.touch()
                    self.stats.hit_count += 1
                    self._update_cache_stats()
                    
                    logger.debug(f"Cache hit: {model_name}")
                    return cached_model.model_object
            
            # Check if model is currently being loaded
            if model_id in self.load_futures:
                logger.debug(f"Waiting for ongoing load: {model_name}")
                try:
                    return await self.load_futures[model_id]
                except Exception as e:
                    logger.error(f"Failed to wait for model load: {e}")
                    return None
        
        # Cache miss - need to load model
        self.stats.miss_count += 1
        logger.info(f"Cache miss: {model_name}, loading...")
        
        return await self._load_and_cache_model(model_id, model_name, model_type, load_kwargs)
    
    async def _load_and_cache_model(self,
                                   model_id: str,
                                   model_name: str, 
                                   model_type: str,
                                   load_kwargs: Dict[str, Any]) -> Optional[Any]:
        """Load model and add to cache"""
        
        # Create future to prevent duplicate loads
        future = asyncio.Future()
        
        with self._lock:
            self.load_futures[model_id] = future
        
        try:
            # Check memory pressure before loading
            if self.resource_manager and self.resource_manager.is_memory_pressure_high():
                logger.warning("High memory pressure, cleaning cache before model load")
                await self.cleanup_for_memory_pressure()
            
            # Reserve memory for model loading
            estimated_memory = self._estimate_model_memory(model_name, model_type)
            memory_reserved = False
            
            if self.resource_manager:
                memory_reserved = self.resource_manager.reserve_memory(
                    f"model_load_{model_id}",
                    estimated_memory,
                    f"Loading {model_name} ({model_type})"
                )
                
                if not memory_reserved:
                    logger.warning(f"Could not reserve {estimated_memory}MB for model loading")
            
            # Load the model
            load_start = time.time()
            model_object = await self._load_model_direct(model_name, model_type, **load_kwargs)
            load_duration = time.time() - load_start
            
            if model_object is None:
                future.set_result(None)
                return None
            
            # Calculate actual memory usage
            actual_memory = self._calculate_model_memory(model_object)
            
            # Create cached model entry
            cached_model = CachedModel(
                model_id=model_id,
                model_name=model_name,
                model_type=model_type,
                model_object=model_object,
                memory_mb=actual_memory,
                load_time=time.time(),
                last_access=time.time(),
                access_count=1,
                ttl_hours=self.default_ttl_hours,
                metadata={
                    "load_kwargs": load_kwargs,
                    "estimated_memory_mb": estimated_memory,
                    "load_duration": load_duration
                }
            )
            
            # Add to cache with eviction if necessary
            with self._lock:
                await self._ensure_cache_space(actual_memory)
                self.cached_models[model_id] = cached_model
                
                # Update statistics
                self.stats.load_count += 1
                self._update_cache_stats()
            
            # Release memory reservation
            if memory_reserved and self.resource_manager:
                self.resource_manager.release_memory_reservation(f"model_load_{model_id}")
            
            logger.info(f"Model cached: {model_name} ({actual_memory:.1f}MB, {load_duration:.2f}s)")
            
            # Set future result
            future.set_result(model_object)
            return model_object
            
        except Exception as e:
            logger.error(f"Failed to load and cache model {model_name}: {e}")
            future.set_exception(e)
            return None
            
        finally:
            # Clean up loading future
            with self._lock:
                self.load_futures.pop(model_id, None)
    
    async def _load_model_direct(self, 
                               model_name: str, 
                               model_type: str,
                               **load_kwargs) -> Optional[Any]:
        """Load model directly without caching"""
        try:
            if model_type == "whisper":
                return await self._load_whisper_model(model_name, **load_kwargs)
            else:
                logger.error(f"Unsupported model type: {model_type}")
                return None
                
        except Exception as e:
            logger.error(f"Direct model loading failed: {e}")
            return None
    
    async def _load_whisper_model(self, model_name: str, **kwargs) -> Optional[Any]:
        """Load Whisper model with browser-safe patterns"""
        whisper = get_whisper()
        if whisper is None:
            raise ImportError("Whisper not available")
        
        try:
            # Load model in executor to prevent blocking
            loop = asyncio.get_event_loop()
            
            def load_whisper():
                # Force CPU-only loading
                device = "cpu"
                
                # Load model with specified device
                model = whisper.load_model(model_name, device=device)
                
                # Apply any additional optimizations
                if hasattr(model, 'eval'):
                    model.eval()
                
                return model
            
            model = await loop.run_in_executor(None, load_whisper)
            logger.info(f"Whisper model '{model_name}' loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load Whisper model '{model_name}': {e}")
            return None
    
    async def _ensure_cache_space(self, required_memory_mb: float):
        """Ensure cache has space for new model"""
        with self._lock:
            current_memory = sum(model.memory_mb for model in self.cached_models.values())
            
            # Simple heuristic: keep total cache under reasonable limit
            max_cache_memory = 2048  # 2GB max cache
            
            while (current_memory + required_memory_mb > max_cache_memory or
                   len(self.cached_models) >= self.max_cached_models):
                
                # Find model to evict (LRU policy)
                oldest_model_id = None
                oldest_access = float('inf')
                
                for model_id, cached_model in self.cached_models.items():
                    if cached_model.last_access < oldest_access:
                        oldest_access = cached_model.last_access
                        oldest_model_id = model_id
                
                if oldest_model_id:
                    logger.info(f"Evicting model for space: {self.cached_models[oldest_model_id].model_name}")
                    await self._evict_model(oldest_model_id)
                    current_memory = sum(model.memory_mb for model in self.cached_models.values())
                else:
                    break  # No models to evict
    
    async def _evict_model(self, model_id: str):
        """Evict a model from cache"""
        with self._lock:
            if model_id in self.cached_models:
                cached_model = self.cached_models.pop(model_id)
                
                # Clean up model object
                try:
                    if hasattr(cached_model.model_object, 'cpu'):
                        cached_model.model_object.cpu()
                    
                    del cached_model.model_object
                    gc.collect()  # Force garbage collection
                    
                except Exception as e:
                    logger.warning(f"Error during model cleanup: {e}")
                
                self.stats.eviction_count += 1
                logger.debug(f"Model evicted: {cached_model.model_name}")
    
    def _generate_model_id(self, 
                          model_name: str, 
                          model_type: str, 
                          load_kwargs: Dict[str, Any]) -> str:
        """Generate unique model ID based on parameters"""
        # Create hash from model parameters
        param_str = f"{model_name}_{model_type}_{sorted(load_kwargs.items())}"
        model_hash = hashlib.md5(param_str.encode()).hexdigest()[:12]
        return f"{model_type}_{model_name}_{model_hash}"
    
    def _estimate_model_memory(self, model_name: str, model_type: str) -> float:
        """Estimate memory required for model"""
        if model_type == "whisper":
            # Conservative estimates for Whisper models
            whisper_memory_estimates = {
                "tiny": 150,      # ~150MB
                "base": 250,      # ~250MB  
                "small": 500,     # ~500MB
                "medium": 800,    # ~800MB
                "large": 1200,    # ~1.2GB
                "large-v2": 1200,
                "large-v3": 1200,
            }
            return whisper_memory_estimates.get(model_name, 800)  # Default to medium
        
        return 500  # Default estimate
    
    def _calculate_model_memory(self, model_object: Any) -> float:
        """Calculate actual memory usage of model"""
        try:
            torch = get_torch()
            if torch and hasattr(model_object, 'parameters'):
                # Calculate PyTorch model memory
                total_params = sum(p.numel() for p in model_object.parameters())
                # Rough estimate: 4 bytes per parameter (float32)
                memory_bytes = total_params * 4
                return memory_bytes / (1024 * 1024)  # Convert to MB
                
        except Exception as e:
            logger.warning(f"Could not calculate model memory: {e}")
        
        # Fallback to process memory increase (simplified)
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except:
            pass
        
        return 500.0  # Fallback estimate
    
    async def cleanup_expired_models(self) -> int:
        """Clean up expired models from cache"""
        expired_models = []
        
        with self._lock:
            for model_id, cached_model in self.cached_models.items():
                if cached_model.is_expired:
                    expired_models.append(model_id)
        
        # Evict expired models
        for model_id in expired_models:
            await self._evict_model(model_id)
        
        if expired_models:
            logger.info(f"Cleaned up {len(expired_models)} expired models")
        
        return len(expired_models)
    
    async def cleanup_for_memory_pressure(self) -> int:
        """Aggressive cleanup for memory pressure situations"""
        cleaned_count = 0
        
        with self._lock:
            # Sort models by priority (access count and recency)
            models_by_priority = sorted(
                self.cached_models.items(),
                key=lambda x: (x[1].access_count, x[1].last_access)
            )
            
            # Remove half of the cached models
            models_to_remove = models_by_priority[:len(models_by_priority) // 2]
        
        for model_id, _ in models_to_remove:
            await self._evict_model(model_id)
            cleaned_count += 1
        
        if cleaned_count > 0:
            self.stats.memory_pressure_cleanups += 1
            logger.info(f"Memory pressure cleanup: removed {cleaned_count} models")
        
        return cleaned_count
    
    async def emergency_cleanup(self) -> int:
        """Emergency cleanup - remove all cached models"""
        model_count = len(self.cached_models)
        
        model_ids = list(self.cached_models.keys())
        for model_id in model_ids:
            await self._evict_model(model_id)
        
        if model_count > 0:
            logger.warning(f"Emergency cleanup: removed all {model_count} cached models")
        
        return model_count
    
    def _update_cache_stats(self):
        """Update cache statistics"""
        with self._lock:
            self.stats.cached_models_count = len(self.cached_models)
            self.stats.total_memory_mb = sum(model.memory_mb for model in self.cached_models.values())
            
            # Calculate hit ratio
            total_requests = self.stats.hit_count + self.stats.miss_count
            if total_requests > 0:
                self.stats.cache_hit_ratio = self.stats.hit_count / total_requests
            
            # Calculate average load time
            if self.stats.load_count > 0:
                total_load_time = sum(
                    model.metadata.get("load_duration", 0) 
                    for model in self.cached_models.values()
                )
                self.stats.average_load_time = total_load_time / self.stats.load_count
            
            # Find oldest model
            if self.cached_models:
                oldest_age = max(model.age_hours for model in self.cached_models.values())
                self.stats.oldest_model_hours = oldest_age
    
    def get_cache_stats(self) -> CacheStats:
        """Get current cache statistics"""
        self._update_cache_stats()
        return self.stats
    
    def get_cached_models_info(self) -> List[Dict[str, Any]]:
        """Get information about cached models"""
        with self._lock:
            return [
                {
                    "model_id": model.model_id,
                    "model_name": model.model_name,
                    "model_type": model.model_type,
                    "memory_mb": model.memory_mb,
                    "age_hours": model.age_hours,
                    "idle_hours": model.idle_hours,
                    "access_count": model.access_count,
                    "is_expired": model.is_expired,
                    "ttl_hours": model.ttl_hours
                }
                for model in self.cached_models.values()
            ]
    
    async def clear_cache(self) -> int:
        """Clear all cached models"""
        return await self.emergency_cleanup()


# Global cache instance
_global_model_cache: Optional[IntelligentModelCache] = None


def get_model_cache() -> IntelligentModelCache:
    """Get global model cache instance"""
    global _global_model_cache
    
    if _global_model_cache is None:
        # Default configuration
        default_config = {
            "enable_cache": True,
            "cache_dir": "cache/models",
            "max_cached_models": 3,
            "cache_ttl_hours": 24,
            "lazy_loading": True,
            "memory_pressure_unload": True
        }
        
        _global_model_cache = IntelligentModelCache(default_config)
        logger.info("Global model cache initialized")
    
    return _global_model_cache


def initialize_model_cache(config: Dict[str, Any]) -> IntelligentModelCache:
    """Initialize model cache with configuration"""
    global _global_model_cache
    _global_model_cache = IntelligentModelCache(config)
    return _global_model_cache


# Convenience functions
async def load_whisper_model(model_name: str = "medium") -> Optional[Any]:
    """Load Whisper model through cache"""
    cache = get_model_cache()
    return await cache.get_model(model_name, "whisper")


async def preload_whisper_model(model_name: str = "medium") -> bool:
    """Preload Whisper model in background"""
    try:
        cache = get_model_cache()
        model = await cache.get_model(model_name, "whisper")
        return model is not None
    except Exception as e:
        logger.error(f"Failed to preload Whisper model: {e}")
        return False


def get_cache_info() -> Dict[str, Any]:
    """Get cache information"""
    cache = get_model_cache()
    stats = cache.get_cache_stats()
    
    return {
        "enabled": cache.enable_cache,
        "stats": stats,
        "cached_models": cache.get_cached_models_info()
    }


# Export main functions
__all__ = [
    "IntelligentModelCache",
    "CachedModel",
    "CacheStats", 
    "get_model_cache",
    "initialize_model_cache",
    "load_whisper_model",
    "preload_whisper_model",
    "get_cache_info"
]