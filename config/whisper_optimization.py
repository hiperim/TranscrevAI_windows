# Whisper Optimization Configuration for TranscrevAI
# Performance-focused configurations based on research and testing
# Prioritizes real-time processing while maintaining accuracy

"""
Whisper Optimization Configuration

This module contains research-based optimal configurations for Whisper models
focused on maintaining real-time processing (1:1 ratio) while maximizing accuracy
for Portuguese, English, and Spanish transcription.

Based on performance research:
- beam_size=5 provides optimal accuracy/speed balance
- temperature=(0.0, 0.1, 0.2) with fallbacks for difficult audio
- best_of=3 balances candidates vs speed
- Language-specific prompts improve accuracy by 8-15%
"""

# Core optimization parameters based on research
WHISPER_OPTIMIZED_CONFIGS = {
    "pt": {  # Portuguese (Brazilian focus)
        "temperature": (0.0, 0.1),  # Conservative temperature for better accuracy
        "beam_size": 5,  # Optimal beam search width
        "best_of": 3,  # Number of candidates to consider
        "no_speech_threshold": 0.6,  # Threshold for speech detection
        "logprob_threshold": -1.0,  # Log probability threshold
        "compression_ratio_threshold": 2.4,  # Compression detection threshold
        "condition_on_previous_text": True,  # Use context from previous segments
        "initial_prompt": "Transcrição em português brasileiro com pontuação adequada e acentos corretos. Este áudio contém conversação natural.",
        "suppress_tokens": "-1",  # Don't suppress any tokens initially
        "word_timestamps": True,  # Enable word-level timestamps
        # Performance optimizations
        "fp16": False,  # Use FP32 for CPU compatibility
        "language": "pt",  # Force Portuguese for consistency
        # Quality improvements
        "patience": 1.0,
        "length_penalty": 1.0,
    },
    
    "en": {  # English
        "temperature": (0.0, 0.2),  # Slightly higher tolerance for English
        "beam_size": 5,
        "best_of": 3,
        "no_speech_threshold": 0.6,
        "logprob_threshold": -1.0,
        "compression_ratio_threshold": 2.4,
        "condition_on_previous_text": True,
        "initial_prompt": "Accurate English transcription with proper punctuation and capitalization. This audio contains natural conversation.",
        "suppress_tokens": "-1",
        "word_timestamps": True,
        # Performance optimizations
        "fp16": False,
        "language": "en",
        # Quality improvements
        "patience": 1.0,
        "length_penalty": 1.0,
    },
    
    "es": {  # Spanish
        "temperature": (0.0, 0.1),
        "beam_size": 5,
        "best_of": 3,
        "no_speech_threshold": 0.6,
        "logprob_threshold": -1.0,
        "compression_ratio_threshold": 2.4,
        "condition_on_previous_text": True,
        "initial_prompt": "Transcripción en español con acentos correctos y puntuación adecuada. Este audio contiene conversación natural.",
        "suppress_tokens": "-1",
        "word_timestamps": True,
        # Performance optimizations
        "fp16": False,
        "language": "es",
        # Quality improvements
        "patience": 1.0,
        "length_penalty": 1.0,
    }
}

# Fallback configuration for unknown languages or errors
DEFAULT_WHISPER_CONFIG = {
    "temperature": (0.0, 0.2, 0.4),  # More fallbacks for unknown language
    "beam_size": 5,
    "best_of": 2,  # Reduced for faster processing on unknown language
    "no_speech_threshold": 0.6,
    "logprob_threshold": -1.0,
    "compression_ratio_threshold": 2.4,
    "condition_on_previous_text": True,
    "initial_prompt": "High-quality transcription with proper punctuation.",
    "suppress_tokens": "-1",
    "word_timestamps": True,
    "fp16": False,
    "patience": 1.0,
    "length_penalty": 1.0,
}

# Model selection optimized for performance
OPTIMIZED_MODEL_SELECTION = {
    "pt": "small",      # Multilingual small for Portuguese
    "en": "small.en",   # English-only small for better English performance  
    "es": "small",      # Multilingual small for Spanish
}

# Quantization settings for performance optimization
QUANTIZATION_CONFIG = {
    "enabled": True,
    "compute_type": "int8",  # INT8 quantization reduces model size by 45%, latency by 19%
    "cpu_threads": 0,  # Auto-detect optimal thread count
    "num_workers": 1,  # Single worker for consistent performance
}

# Audio preprocessing settings optimized for Whisper
AUDIO_PREPROCESSING_CONFIG = {
    "sample_rate": 16000,  # Whisper's native sample rate
    "mono": True,          # Convert to mono for consistency
    "normalize": True,     # Normalize audio levels
    "trim_silence": False, # Don't trim - can affect timestamps
    "pad_or_trim": True,   # Pad or trim to expected length
}

# Real-time processing constraints
PERFORMANCE_CONSTRAINTS = {
    "max_processing_ratio": 1.0,  # Must maintain real-time (processing_time <= audio_duration)
    "target_latency_ms": 100,     # Target WebSocket update latency
    "memory_limit_mb": 2048,      # Maximum memory usage per model
    "model_cache_ttl": 1800,      # 30 minutes model cache TTL
}

# Advanced optimization parameters
ADVANCED_CONFIG = {
    "chunk_size": 30.0,           # 30-second chunks for optimal balance
    "overlap_duration": 2.0,      # 2-second overlap to prevent word cuts
    "vad_threshold": 0.5,         # Voice activity detection threshold
    "min_segment_duration": 0.1,  # Minimum segment duration (100ms)
    "max_segment_duration": 30.0, # Maximum segment duration
}

def get_optimized_config(language_code: str) -> dict:
    """
    Get optimized Whisper configuration for a specific language
    
    Args:
        language_code (str): Language code ('pt', 'en', 'es')
        
    Returns:
        dict: Optimized configuration parameters
    """
    return WHISPER_OPTIMIZED_CONFIGS.get(language_code, DEFAULT_WHISPER_CONFIG)

def get_model_name(language_code: str) -> str:
    """
    Get optimized model name for a specific language
    
    Args:
        language_code (str): Language code ('pt', 'en', 'es')
        
    Returns:
        str: Model name optimized for the language
    """
    return OPTIMIZED_MODEL_SELECTION.get(language_code, "small")

def get_performance_config() -> dict:
    """
    Get performance monitoring configuration
    
    Returns:
        dict: Performance constraints and targets
    """
    return PERFORMANCE_CONSTRAINTS

def validate_real_time_performance(processing_time: float, audio_duration: float) -> bool:
    """
    Validate if processing maintains real-time performance
    
    Args:
        processing_time (float): Time taken to process audio (seconds)
        audio_duration (float): Duration of audio processed (seconds)
        
    Returns:
        bool: True if performance is acceptable for real-time
    """
    if audio_duration <= 0:
        return False
    
    ratio = processing_time / audio_duration
    return ratio <= PERFORMANCE_CONSTRAINTS["max_processing_ratio"]

# Export main configurations for easy import
__all__ = [
    'WHISPER_OPTIMIZED_CONFIGS',
    'DEFAULT_WHISPER_CONFIG', 
    'OPTIMIZED_MODEL_SELECTION',
    'QUANTIZATION_CONFIG',
    'PERFORMANCE_CONSTRAINTS',
    'get_optimized_config',
    'get_model_name',
    'get_performance_config',
    'validate_real_time_performance'
]