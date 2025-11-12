"""
Dependency injection for FastAPI
"""
import os
import queue
import threading
import asyncio
from pathlib import Path
from typing import Optional
from functools import lru_cache
from src.transcription import TranscriptionService
from src.diarization import PyannoteDiarizer
from src.audio_processing import AudioQualityAnalyzer, SessionManager, LiveAudioProcessor
from src.file_manager import FileManager
from config.app_config import get_config

# Global service instances (singleton pattern)
_services = {}
_lock = threading.RLock()  # Reentrant lock to allow nested get_* calls


@lru_cache()
def get_config_cached():
    """Cached config instance"""
    return get_config()


def get_file_manager() -> FileManager:
    """Get or create FileManager instance"""
    with _lock:
        if 'file_manager' not in _services:
            config = get_config_cached()
            data_dir = Path(os.getenv("DATA_DIR", config.data_dir or "./data")).resolve()
            _services['file_manager'] = FileManager(data_dir=data_dir)
        return _services['file_manager']


def get_transcription_service() -> TranscriptionService:
    """Get or create TranscriptionService instance"""
    with _lock:
        if 'transcription_service' not in _services:
            config = get_config_cached()
            _services['transcription_service'] = TranscriptionService(
                model_name=config.model_name,
                device=config.device
            )
        return _services['transcription_service']


def get_diarization_service() -> PyannoteDiarizer:
    """Get or create PyannoteDiarizer instance"""
    with _lock:
        if 'diarization_service' not in _services:
            config = get_config_cached()
            _services['diarization_service'] = PyannoteDiarizer(device=config.device)
        return _services['diarization_service']


def get_audio_quality_analyzer() -> AudioQualityAnalyzer:
    """Get or create AudioQualityAnalyzer instance"""
    with _lock:
        if 'audio_quality_analyzer' not in _services:
            _services['audio_quality_analyzer'] = AudioQualityAnalyzer()
        return _services['audio_quality_analyzer']


def get_session_manager() -> SessionManager:
    """Get or create SessionManager instance"""
    with _lock:
        if 'session_manager' not in _services:
            _services['session_manager'] = SessionManager(session_timeout_hours=24)
        return _services['session_manager']


def get_live_audio_processor() -> LiveAudioProcessor:
    """Get or create LiveAudioProcessor instance"""
    with _lock:
        if 'live_audio_processor' not in _services:
            file_manager = get_file_manager()
            _services['live_audio_processor'] = LiveAudioProcessor(file_manager=file_manager)
        return _services['live_audio_processor']


def get_transcription_queue() -> queue.Queue:
    """Get or create transcription queue for worker"""
    with _lock:
        if 'transcription_queue' not in _services:
            _services['transcription_queue'] = queue.Queue()
        return _services['transcription_queue']


def get_worker_thread(loop: Optional[asyncio.AbstractEventLoop] = None) -> threading.Thread:
    """Get or create worker thread"""
    with _lock:
        if 'worker_thread' not in _services:
            from src.worker import transcription_worker
            if loop is None:
                loop = asyncio.get_running_loop()
            
            assert loop is not None # Ensure loop is not None for the type checker

            # Create worker thread
            worker_thread = threading.Thread(
                target=transcription_worker,
                args=(
                    get_transcription_queue(),
                    get_file_manager(),
                    get_live_audio_processor(),
                    get_transcription_service(),
                    get_session_manager(),
                    loop
                ),
                daemon=True
            )
            worker_thread.start()
            _services['worker_thread'] = worker_thread
        return _services['worker_thread']


def cleanup_services():
    """Cleanup all services on shutdown"""
    with _lock:
        # Signal worker to stop
        if 'transcription_queue' in _services:
            _services['transcription_queue'].put(None)

        # Wait for worker
        if 'worker_thread' in _services:
            _services['worker_thread'].join(timeout=5)

        # Clear all services
        _services.clear()
