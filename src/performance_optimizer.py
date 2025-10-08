# performance_optimizer.py - Minimal placeholder implementation
"""
Performance Optimizer - Minimal implementation for compatibility
Note: This module exists for backward compatibility with main.py
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

class ProcessType:
    """Process type constants"""
    TRANSCRIPTION = "transcription"
    DIARIZATION = "diarization"

class AdaptivePerformanceOptimizer:
    """Minimal optimizer placeholder - no actual optimization performed"""

    def __init__(self):
        logger.debug("AdaptivePerformanceOptimizer initialized (placeholder)")

_global_optimizer: Optional[AdaptivePerformanceOptimizer] = None

def get_production_optimizer() -> AdaptivePerformanceOptimizer:
    """Get singleton optimizer instance"""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = AdaptivePerformanceOptimizer()
    return _global_optimizer
