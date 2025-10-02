#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Models Module - OpenAI Whisper Only
Simplified version without ONNX complexity for maximum reliability
"""

import gc
import logging
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Set up logging
logger = logging.getLogger("transcrevai.models")

# Whisper imports
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError as e:
    logging.warning(f"OpenAI Whisper not available: {e}")
    WHISPER_AVAILABLE = False

class SimpleModelManager:
    """Simplified model manager for OpenAI Whisper only"""

    def __init__(self, cache_dir: str = "models"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_monitor = None
        logger.info(f"SimpleModelManager initialized: {self.cache_dir}")

    def get_system_info(self) -> Dict:
        """Get system information for compatibility checks"""
        return {
            "whisper_available": WHISPER_AVAILABLE,
            "cache_directory": str(self.cache_dir),
            "cache_size_mb": self._get_cache_size_mb(),
            "architecture": "whisper_only"
        }

    def _get_cache_size_mb(self) -> float:
        """Calculate cache directory size in MB"""
        try:
            total_size = sum(
                f.stat().st_size for f in self.cache_dir.rglob('*') if f.is_file()
            )
            return total_size / (1024 * 1024)
        except Exception:
            return 0.0

    def load_whisper_model(self, model_name: str = "medium", device: str = "cpu") -> Optional[object]:
        """Load OpenAI Whisper model"""
        if not WHISPER_AVAILABLE:
            logger.error("OpenAI Whisper not available")
            return None

        try:
            logger.info(f"Loading Whisper model: {model_name}")
            start_time = time.time()

            model = whisper.load_model(model_name, device=device)

            load_time = time.time() - start_time
            logger.info(f"Whisper model loaded in {load_time:.2f}s")

            return model

        except Exception as e:
            logger.error(f"Failed to load Whisper model {model_name}: {e}")
            return None

    def cleanup_models(self, force: bool = False) -> Dict:
        """Clean up models and memory"""
        cleanup_result = {
            "memory_freed_mb": 0,
            "models_unloaded": 0,
            "cache_cleaned": False
        }

        try:
            # Force garbage collection
            before_gc = gc.get_count()
            gc.collect()
            after_gc = gc.get_count()

            cleanup_result["gc_collections"] = sum(before_gc) - sum(after_gc)
            logger.info("Memory cleanup completed")

            if force:
                # Clean cache directory
                cache_size_before = self._get_cache_size_mb()

                # Keep only essential files, remove temporary files
                for temp_file in self.cache_dir.rglob("*.tmp"):
                    temp_file.unlink()

                cache_size_after = self._get_cache_size_mb()
                cleanup_result["memory_freed_mb"] = cache_size_before - cache_size_after
                cleanup_result["cache_cleaned"] = True

            return cleanup_result

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return cleanup_result

    def get_model_info(self, model_name: str = "medium") -> Dict:
        """Get information about a specific model"""
        return {
            "name": model_name,
            "type": "whisper",
            "available": WHISPER_AVAILABLE,
            "device": "cpu",
            "architecture": "transformer",
            "quantization": "fp32"
        }

# Global instance for compatibility
simple_model_manager = SimpleModelManager()

def get_model_manager():
    """Get the simplified model manager instance"""
    return simple_model_manager

def get_system_info():
    """Get system information"""
    return simple_model_manager.get_system_info()

# Legacy compatibility exports
ConsolidatedModelManager = SimpleModelManager
consolidated_model_manager = simple_model_manager

if __name__ == "__main__":
    # Test the simplified model manager
    manager = SimpleModelManager()
    system_info = manager.get_system_info()

    print("=== Simple Model Manager Test ===")
    print(f"  Whisper Available: {system_info['whisper_available']}")
    print(f"  Cache Directory: {system_info['cache_directory']}")
    print(f"  Cache Size: {system_info['cache_size_mb']:.2f} MB")
    print(f"  Architecture: {system_info['architecture']}")