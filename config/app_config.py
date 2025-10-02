# CRITICAL FIX: Enhanced app configuration with adaptive processing capabilities
import os
from pathlib import Path

# Application metadata
APP_PACKAGE_NAME = "transcrevai"

# Base directories - Docker-compatible with environment variable override
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv('DATA_DIR', str(BASE_DIR / "data")))
TEMP_DIR = Path(os.getenv('TEMP_DIR', str(DATA_DIR / "temp")))

def _ensure_directories_created():
    """Ensure all required directories exist"""
    directories = [
        DATA_DIR,
        DATA_DIR / "inputs",
        DATA_DIR / "outputs", 
        DATA_DIR / "transcripts",
        DATA_DIR / "recordings",
        DATA_DIR / "logs",
        DATA_DIR / "models",
        TEMP_DIR,
        DATA_DIR / "processed"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Whisper model configuration - Docker-compatible
# Model name for faster-whisper (will download and cache automatically)
# SPRINT 3: CTranslate2-optimized medium model with enhanced PT-BR prompts
# - Default: "medium" (CTranslate2 format, compatible with faster-whisper)
# - Enhancement: PT-BR accuracy improved via adaptive prompts and VAD
# Note: pierreguillou/whisper-medium-portuguese requires conversion to CT2 format
WHISPER_MODEL_PATH = os.getenv('WHISPER_MODEL_PATH', "medium")


# Whisper configuration - PT-BR only
# SPRINT 3: Enhanced with PT-BR specific optimizations
WHISPER_CONFIG = {
    "word_timestamps": True,
    "condition_on_previous_text": False,  # Better for conversations
    "language": "pt",  # Fixed to Portuguese Brazilian
    "initial_prompt": "Transcrição em português brasileiro. Pontuação correta. Acentuação correta. Maiúsculas em nomes próprios."
}

# Adaptive prompts for PT-BR based on audio complexity
# SPRINT 3: Enhanced prompts for better PT-BR accuracy
ADAPTIVE_PROMPTS = {
    "general": "Português brasileiro. Pontuação, acentuação e maiúsculas corretas. Transcrição precisa.",
    "finance": "Português brasileiro financeiro. Termos: balanço, lucro, receita, despesa, EBITDA, ROI, fluxo de caixa, juros, investimento, ativo, passivo.",
    "it": "Português brasileiro técnico. Termos: API, banco de dados, SQL, Python, JavaScript, servidor, nuvem, AWS, Azure, Docker, deploy, commit, merge.",
    "medical": "Português brasileiro médico. Termos: diagnóstico, tratamento, paciente, sintoma, medicação, anatomia, exame, prescrição, patologia.",
    "legal": "Português brasileiro jurídico. Termos: petição, contrato, jurisprudência, liminar, acórdão, legislação, processo, réu, autor, sentença.",
    "lecture": "Palestra em português brasileiro. Discurso formal. Pontuação clara. Nomes próprios com maiúsculas.",
    "conversation": "Conversa em português brasileiro. Múltiplos falantes. Linguagem coloquial. Identificação precisa.",
    "complex_dialogue": "Diálogo complexo em português brasileiro. Sobreposições. Interrupções. Transições rápidas entre falantes."
}

# Real-time processing configuration
REALTIME_CONFIG = {
    "performance": {
        "chunk_duration": 1.5,  # Reduced for faster processing
        "max_processing_time": 20.0,  # Reduced time limit
        "memory_limit_mb": 256  # Reduzido para hardware limitado
    },
    "quality": {
        "sample_rate": 16000,
        "channels": 1,
        "bit_depth": 16
    }
}

# Processing configuration - Fixed optimized settings for PT-BR medium model
PROCESSING_CONFIG = {
    "transcription_model": "medium",  # Fixed medium model
    "language": "pt",  # Fixed Portuguese Brazilian
    "whisper_beam_size": 1,  # Optimized for speed
    "whisper_patience": 0.5,  # Balanced setting
    "diarization_method": "standard",  # Standard diarization
    "preprocessing": "standard",  # Standard preprocessing
    "target_latency": 1.0  # Balanced latency
}

# Audio processing configuration
AUDIO_CONFIG = {
    "supported_formats": [".wav", ".mp3", ".mp4", ".m4a", ".flac"],
    "max_file_size_mb": 100,
    "preprocessing": {
        "normalize_audio": True,
        "noise_reduction": True,
        "high_pass_filter": True,
        "target_lufs": -23.0
    }
}

# Speaker diarization configuration
DIARIZATION_CONFIG = {
    "min_speakers": 1,
    "max_speakers": 6,
    "default_method": "adaptive",  # CRITICAL FIX: Use adaptive method
    "confidence_threshold": 0.5,
    "segment_min_duration": 0.5,  # seconds
    "merge_threshold": 0.1,  # seconds gap for merging
    # CRITICAL FIX: Enhanced speaker analysis parameters  
    "analysis_thresholds": {
        "single_speaker": 0.15,      # REVERTED from 0.4
        "multi_speaker": 0.4,        # REVERTED from 0.6
        "short_audio_threshold": 5.0  # REVERTED from 10.0
    }
}

# VAD (Voice Activity Detection) configuration
VAD_CONFIG = {
    "threshold": 0.3,               # FASE 10: Mais sensível (0.5→0.3) para detectar fala suave
    "min_speech_duration_ms": 100,  # FASE 10: Reduzido (250→100) para aceitar falas curtas
    "min_silence_duration_ms": 300, # FASE 10: Reduzido (1000→300) menos pausa necessária
    "speech_pad_ms": 200            # Padding around detected speech
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_rotation": True,
    "max_file_size_mb": 10,
    "backup_count": 5
}

# Memory management configuration
MEMORY_CONFIG = {
    "max_model_cache_mb": 1024,  # 1GB for model caching
    "cleanup_threshold": 0.8,    # Cleanup when 80% memory used
    "enable_garbage_collection": True,
    "gc_interval": 10.0,         # Run GC every 10 seconds
    "temp_cleanup_interval": 60.0  # Clean temp files every 60 seconds
}

# Model management configuration
MODEL_CONFIG = {
    "cache_timeout": 1800,  # 30 minutes
    "concurrent_downloads": 2,
    "download_timeout": 300,  # 5 minutes
    "retry_attempts": 3,
    "cache_models": True
}

# CRITICAL FIX: Quality assurance thresholds
QUALITY_CONFIG = {
    "transcription": {
        "min_confidence": 0.3,
        "word_accuracy_threshold": 0.8,
        "segment_min_duration": 0.1
    },
    "diarization": {
        "min_confidence": 0.5,
        "speaker_balance_threshold": 0.1,
        "temporal_consistency": True
    },
    "alignment": {
        "min_overlap_ratio": 0.3,
        "max_time_drift": 0.5  # seconds
    }
}

# Error handling configuration
ERROR_CONFIG = {
    "max_retries": 3,
    "retry_delay": 1.0,  # seconds
    "timeout_handling": "graceful",
    "fallback_modes": True
}

# Initialize directories on import
_ensure_directories_created()