"""
TranscrevAI Source Package - Clean Consolidated Structure

This package provides AI transcription capabilities with multiprocessing support,
model optimization, and comprehensive audio processing features.

Essential modules:
- performance_optimizer: Core multiprocessing and resource management
- models: Model management and INT8 quantization
- transcription: Transcription services
- diarization: Speaker diarization
- audio_processing: Audio capture and processing
- file_manager: File operations and management
- logging_setup: Logging configuration
- platform_manager: Platform-specific management
- subtitle_generator: Subtitle generation
"""

# Essential modules
from . import audio_processing
from . import diarization
from . import file_manager
from . import logging_setup
from . import models
from . import performance_optimizer
from . import platform_manager
from . import subtitle_generator
from . import transcription

__all__ = [
    'audio_processing',
    'diarization',
    'file_manager',
    'logging_setup',
    'models',
    'performance_optimizer',
    'platform_manager',
    'subtitle_generator',
    'transcription',
]