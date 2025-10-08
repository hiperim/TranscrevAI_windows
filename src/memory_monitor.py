# memory_monitor.py - Simplified implementation
"""
Memory Monitor with Intelligent Cache for TranscrevAI
"""

import logging
import time
import threading
from typing import Dict, Any, Optional
from dataclasses import dataclass
import sys

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class MemorySnapshot:
    timestamp: float
    process_memory_mb: float
    system_memory_mb: float

class IntelligentCache:
    """LRU Cache with automatic memory management"""
    def __init__(self, max_size_mb: float = 256.0):
        self.max_size_mb = max_size_mb
        self._cache: Dict[str, Any] = {}
        self._access_times: Dict[str, float] = {}
        self._item_sizes: Dict[str, float] = {}
        self._total_size_mb = 0.0
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._cache:
                self._access_times[key] = time.time()
                return self._cache[key]
            return None

    def put(self, key: str, value: Any, size_mb: Optional[float] = None):
        with self._lock:
            size_mb = size_mb or (sys.getsizeof(value) / (1024 * 1024))
            if size_mb > self.max_size_mb: return
            if key in self._cache: self.remove(key)
            while self._total_size_mb + size_mb > self.max_size_mb:
                if not self._evict_lru(): break
            self._cache[key] = value
            self._access_times[key] = time.time()
            self._item_sizes[key] = size_mb
            self._total_size_mb += size_mb

    def _evict_lru(self) -> bool:
        with self._lock:
            if not self._access_times: return False
            lru_key = min(self._access_times, key=lambda k: self._access_times[k])
            return self.remove(lru_key)

    def remove(self, key: str) -> bool:
        with self._lock:
            if key in self._cache:
                self._total_size_mb -= self._item_sizes[key]
                del self._cache[key], self._access_times[key], self._item_sizes[key]
                return True
            return False

class MemoryMonitor:
    """Simplified memory monitor with intelligent cache management"""
    def __init__(self, max_memory_mb: float = 2048.0):
        self.max_memory_mb = max_memory_mb
        self.caches: Dict[str, IntelligentCache] = {}

    def start_monitoring(self):
        """Placeholder for compatibility - monitoring is passive"""
        logger.debug("MemoryMonitor: passive monitoring mode")

    def stop_monitoring(self):
        """Placeholder for compatibility"""
        pass

    def register_cache(self, name: str, max_size_mb: float = 256.0) -> IntelligentCache:
        cache = IntelligentCache(max_size_mb)
        self.caches[name] = cache
        return cache

    def get_cache(self, name: str) -> Optional[IntelligentCache]:
        return self.caches.get(name)

_global_monitor: Optional[MemoryMonitor] = None

def get_memory_monitor() -> MemoryMonitor:
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = MemoryMonitor()
    return _global_monitor

def get_intelligent_cache(name: str, max_size_mb: float = 256.0) -> IntelligentCache:
    monitor = get_memory_monitor()
    cache = monitor.get_cache(name)
    if cache is None:
        cache = monitor.register_cache(name, max_size_mb)
    return cache
