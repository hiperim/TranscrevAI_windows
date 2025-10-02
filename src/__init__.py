"""
TranscrevAI Source Package - Production-Ready Structure

This package provides AI transcription capabilities with multiprocessing support,
model optimization, and comprehensive audio processing features.

Core modules:
- transcription: Main transcription services with PT-BR optimization
- audio_processing: Audio capture and real-time processing
- diarization: Speaker diarization with dynamic detection
- performance_optimizer: Multiprocessing and resource management
- file_manager: File operations and caching
- subtitle_generator: SRT subtitle generation
- logging_setup: Professional logging configuration

Note: models.py removed - model management now handled by dual_whisper_system.py
"""

# Core modules
from . import audio_processing
from . import diarization
from . import file_manager
from . import logging_setup
# models removed - now using dual_whisper_system.py directly
from . import performance_optimizer
from . import subtitle_generator
from . import transcription

__all__ = [
    'transcription',
    'audio_processing',
    'diarization',
    'performance_optimizer',
    # 'models',  # Removed
    'file_manager',
    'subtitle_generator',
    'logging_setup',
]
