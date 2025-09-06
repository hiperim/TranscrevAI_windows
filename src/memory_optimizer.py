"""
Memory Management Optimizer for Real-Time Performance
Implements the memory optimization strategies from fixes.txt
"""

import gc
import psutil
import numpy as np
from pathlib import Path
import time
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class MemoryOptimizedProcessor:
    """
    Memory-conscious processor implementing strategies from fixes.txt
    """
    
    def __init__(self, max_memory_mb: int = 512):
        self.max_memory = max_memory_mb * 1024 * 1024  # Convert to bytes
        self.current_memory = 0
        self.model_cache = {}
        self.temp_files = set()
        self.last_cleanup = time.time()
        
    def get_memory_usage(self) -> float:
        """Get current process memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def check_memory_pressure(self) -> bool:
        """Check if memory usage is above threshold"""
        current_usage = self.get_memory_usage()
        return current_usage > (self.max_memory / 1024 / 1024 * 0.8)  # 80% threshold
    
    def process_with_memory_limit(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Process audio data with memory constraints
        Implementation of memory optimization from fixes.txt
        """
        audio_size = audio_data.nbytes
        
        # Check memory usage before processing
        if self.check_memory_pressure() or (self.current_memory + audio_size > self.max_memory):
            logger.info("Memory pressure detected, initiating cleanup")
            self._cleanup_memory()
        
        # Process in chunks if data is too large
        if audio_size > self.max_memory // 2:
            logger.info(f"Large audio data ({audio_size / 1024 / 1024:.1f}MB), processing in chunks")
            return self._process_chunked(audio_data)
        
        return self._process_normal(audio_data)
    
    def _process_chunked(self, audio_data: np.ndarray, chunk_size: int = 32768) -> np.ndarray:
        """Process large audio data in chunks to manage memory"""
        processed_chunks = []
        
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            processed_chunk = self._process_chunk(chunk)
            processed_chunks.append(processed_chunk)
            
            # Force garbage collection between chunks
            if i % (chunk_size * 4) == 0:
                gc.collect()
        
        return np.concatenate(processed_chunks)
    
    def _process_normal(self, audio_data: np.ndarray) -> np.ndarray:
        """Normal processing for reasonably-sized audio"""
        return self._process_chunk(audio_data)
    
    def _process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Process individual audio chunk"""
        # Simple normalization to avoid memory-heavy operations
        if np.max(np.abs(chunk)) > 0:
            return chunk / np.max(np.abs(chunk)) * 0.8
        return chunk
    
    def _cleanup_memory(self) -> None:
        """
        Aggressive memory cleanup implementation from fixes.txt
        """
        initial_memory = self.get_memory_usage()
        
        # Clear model cache if needed
        if self.model_cache:
            logger.info(f"Clearing {len(self.model_cache)} cached models")
            self.model_cache.clear()
        
        # Remove temporary files
        self._cleanup_temp_files()
        
        # Force garbage collection
        collected = gc.collect()
        
        final_memory = self.get_memory_usage()
        freed_mb = initial_memory - final_memory
        
        logger.info(f"Memory cleanup: freed {freed_mb:.1f}MB, collected {collected} objects")
        self.last_cleanup = time.time()
    
    def _cleanup_temp_files(self) -> None:
        """Remove temporary files to free disk space"""
        cleaned_count = 0
        for temp_file in list(self.temp_files):
            try:
                temp_path = Path(temp_file)
                if temp_path.exists():
                    temp_path.unlink()
                    cleaned_count += 1
                self.temp_files.remove(temp_file)
            except Exception as e:
                logger.debug(f"Failed to remove temp file {temp_file}: {e}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} temporary files")
    
    def register_temp_file(self, file_path: str) -> None:
        """Register a temporary file for cleanup"""
        self.temp_files.add(file_path)
    
    def cache_model(self, model_key: str, model: Any, max_cache_size: int = 2) -> None:
        """Cache model with size limit"""
        # Remove oldest model if cache is full
        if len(self.model_cache) >= max_cache_size:
            oldest_key = next(iter(self.model_cache))
            del self.model_cache[oldest_key]
            logger.info(f"Removed oldest cached model: {oldest_key}")
        
        self.model_cache[model_key] = {
            'model': model,
            'timestamp': time.time()
        }
        logger.info(f"Cached model: {model_key}")
    
    def get_cached_model(self, model_key: str) -> Optional[Any]:
        """Retrieve cached model"""
        if model_key in self.model_cache:
            return self.model_cache[model_key]['model']
        return None
    
    def should_cleanup(self, interval_seconds: int = 30) -> bool:
        """Check if cleanup should be performed based on time interval"""
        return time.time() - self.last_cleanup > interval_seconds

# Global memory optimizer instance
memory_optimizer = MemoryOptimizedProcessor()

def optimize_audio_processing(audio_data: np.ndarray) -> np.ndarray:
    """
    Global function to optimize audio processing with memory management
    """
    return memory_optimizer.process_with_memory_limit(audio_data)

def cleanup_if_needed() -> None:
    """
    Perform cleanup if memory pressure is detected
    """
    if memory_optimizer.check_memory_pressure() or memory_optimizer.should_cleanup():
        memory_optimizer._cleanup_memory()