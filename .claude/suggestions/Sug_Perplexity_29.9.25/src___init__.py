"""
TranscrevAI Optimized - Source Package Initialization
Sistema modular de transcrição e diarização otimizado para PT-BR
"""

# Version information
__version__ = "1.0.0"
__author__ = "TranscrevAI Team"
__description__ = "Sistema otimizado de transcrição e diarização para português brasileiro"

# Core module imports
from .audio_processing import (
    AudioProcessor,
    BrowserSafeAudioRecorder,
    AudioFileProcessor,
    get_supported_formats,
    estimate_processing_time
)

from .transcription import (
    TranscriptionEngine,
    BrowserSafeTranscriber,
    PTBRTranscriptionOptimizer,
    quick_transcribe,
    estimate_transcription_time
)

from .speaker_diarization import (
    SpeakerDiarization,
    BrowserSafeSpeakerClustering,
    OverlappingSpeechDetector,
    quick_diarize,
    estimate_diarization_time
)

from .subtitle_generator import (
    SubtitleGenerator,
    IntelligentAlignment,
    TimestampValidator,
    quick_generate_srt,
    validate_srt_file,
    get_srt_info
)

from .progressive_loader import (
    ProgressiveLoader,
    BrowserSafeLoader,
    LoadingPhase,
    get_progressive_loader,
    load_essential,
    load_everything,
    load_with_progress
)

from .memory_optimizer import (
    MemoryOptimizer,
    AdaptiveMemoryCleanup,
    CleanupLevel,
    get_memory_optimizer,
    optimize_memory_now,
    start_auto_optimization,
    stop_auto_optimization,
    get_memory_stats
)

from .concurrent_engine import (
    ConcurrentEngine,
    BrowserSafeConcurrency,
    ProcessingTask,
    ProcessingResult,
    ProcessingMode,
    TaskPriority,
    get_concurrent_engine,
    process_files_concurrent,
    run_pipeline_concurrent,
    get_concurrent_stats
)

# Main exports for easy access
__all__ = [
    # Audio Processing
    "AudioProcessor",
    "BrowserSafeAudioRecorder", 
    "AudioFileProcessor",
    "get_supported_formats",
    "estimate_processing_time",
    
    # Transcription
    "TranscriptionEngine",
    "BrowserSafeTranscriber",
    "PTBRTranscriptionOptimizer", 
    "quick_transcribe",
    "estimate_transcription_time",
    
    # Speaker Diarization
    "SpeakerDiarization",
    "BrowserSafeSpeakerClustering",
    "OverlappingSpeechDetector",
    "quick_diarize", 
    "estimate_diarization_time",
    
    # Subtitle Generation
    "SubtitleGenerator",
    "IntelligentAlignment",
    "TimestampValidator",
    "quick_generate_srt",
    "validate_srt_file",
    "get_srt_info",
    
    # Progressive Loading
    "ProgressiveLoader",
    "BrowserSafeLoader",
    "LoadingPhase",
    "get_progressive_loader",
    "load_essential",
    "load_everything", 
    "load_with_progress",
    
    # Memory Optimization
    "MemoryOptimizer",
    "AdaptiveMemoryCleanup",
    "CleanupLevel",
    "get_memory_optimizer",
    "optimize_memory_now",
    "start_auto_optimization",
    "stop_auto_optimization",
    "get_memory_stats",
    
    # Concurrent Processing
    "ConcurrentEngine",
    "BrowserSafeConcurrency", 
    "ProcessingTask",
    "ProcessingResult",
    "ProcessingMode",
    "TaskPriority",
    "get_concurrent_engine",
    "process_files_concurrent",
    "run_pipeline_concurrent",
    "get_concurrent_stats",
]


def get_version():
    """Get TranscrevAI version"""
    return __version__


def get_system_info():
    """Get system information for debugging"""
    import platform
    import sys
    
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        cpu_count = psutil.cpu_count()
    except ImportError:
        memory_gb = "unknown"
        cpu_count = "unknown"
    
    return {
        "transcrevai_version": __version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu_count": cpu_count,
        "memory_gb": f"{memory_gb:.1f}" if isinstance(memory_gb, float) else memory_gb,
        "architecture": platform.architecture()[0]
    }


def initialize_system():
    """Initialize TranscrevAI system with all optimizations"""
    print(f"🚀 Initializing TranscrevAI Optimized v{__version__}")
    print("   Sistema otimizado para português brasileiro")
    
    # System info
    system_info = get_system_info()
    print(f"   Python: {system_info['python_version'].split()[0]}")
    print(f"   Platform: {system_info['platform']}")
    print(f"   Resources: {system_info['cpu_count']} cores, {system_info['memory_gb']}GB RAM")
    
    print("✅ TranscrevAI Optimized ready!")
    return system_info


# Module-level initialization message
print(f"TranscrevAI Optimized v{__version__} - PT-BR Exclusive System Loaded")