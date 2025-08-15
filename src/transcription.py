import asyncio
import wave
import json
import logging
import os
import requests
import zipfile
import shutil
from typing import AsyncGenerator, Tuple, List, Dict, Any
from pathlib import Path
from src.file_manager import FileManager
from config.app_config import MODEL_DIR, LANGUAGE_MODELS
from src.logging_setup import setup_app_logging

# Use proper logging setup first
logger = setup_app_logging(logger_name="transcrevai.transcription")

# Import Vosk with graceful fallback
try:
    from vosk import Model, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Vosk not available: {e}")
    VOSK_AVAILABLE = False
    
    # Create dummy classes for graceful degradation
    class DummyModel:
        def __init__(self, *args, **kwargs):
            pass
    
    class DummyKaldiRecognizer:
        def __init__(self, *args, **kwargs):
            pass
        
        def AcceptWaveform(self, data):
            return True
        
        def Result(self):
            return '{"text": "Vosk not available"}'
        
        def FinalResult(self):
            return '{"text": "Vosk not available"}'
    
    Model = DummyModel
    KaldiRecognizer = DummyKaldiRecognizer

class TranscriptionError(Exception):
    pass

# Model management removed - handled by main.py

class AsyncTranscriptionService:
    """Enhanced transcription service with automatic model management"""
    
    def __init__(self):
        self._models = {}
        self._model_lock = asyncio.Lock()
    
    async def load_language_model(self, language_code: str) -> Any:
        """Load cached language model (assumes model already exists)"""
        if not VOSK_AVAILABLE:
            logger.warning("Vosk not available, returning dummy model")
            return Model()  # Return dummy model
        
        async with self._model_lock:
            if language_code not in self._models:
                try:
                    # Get model path (assumes model already downloaded by main.py)
                    model_path = os.path.join(MODEL_DIR, language_code)
                    
                    # Load model in thread pool
                    loop = asyncio.get_event_loop()
                    model = await loop.run_in_executor(None, Model, model_path)
                    
                    self._models[language_code] = model
                    logger.info(f"Model loaded successfully for {language_code}")
                    
                except Exception as e:
                    logger.error(f"Failed to load model for {language_code}: {e}")
                    raise TranscriptionError(f"Model loading failed for {language_code}: {str(e)}")
            
            return self._models[language_code]

class TranscriptionService:
    """Synchronous transcription service for real-time streaming"""
    
    def __init__(self):
        self.model = None
        self.recognizer = None
        self.language = None
    
    async def initialize(self, language_code: str) -> bool:
        """Initialize the transcription service with a language model"""
        try:
            if not VOSK_AVAILABLE:
                logger.warning("Vosk not available for streaming transcription")
                return False
            
            # Get model path
            model_path = os.path.join(MODEL_DIR, language_code)
            
            if not os.path.exists(model_path):
                logger.error(f"Model not found for language: {language_code}")
                return False
            
            # Load model in thread pool
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(None, Model, model_path)
            self.recognizer = KaldiRecognizer(self.model, 16000)
            self.language = language_code
            
            logger.info(f"Transcription service initialized for {language_code}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize transcription service: {e}")
            return False
    
    def transcribe_chunk_sync(self, audio_data, sample_rate: int = 16000):
        """
        Synchronous transcription for real-time chunks.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            Dict with transcription results or None
        """
        try:
            if not self.recognizer:
                logger.warning("Recognizer not initialized for chunk transcription")
                return None
            
            import numpy as np
            
            # Ensure audio is in correct format
            if hasattr(audio_data, 'dtype') and audio_data.dtype != np.int16:
                if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                    # Convert float to int16
                    audio_data = (audio_data * 32767).astype(np.int16)
                else:
                    audio_data = audio_data.astype(np.int16)
            
            # Convert to bytes
            audio_bytes = audio_data.tobytes()
            
            # Recognize
            if self.recognizer.AcceptWaveform(audio_bytes):
                result = json.loads(self.recognizer.Result())
            else:
                result = json.loads(self.recognizer.PartialResult())
            
            # Extract text and confidence
            text = result.get('text', '').strip()
            confidence = result.get('confidence', 0.0)
            
            if text:
                return {
                    "text": text,
                    "confidence": confidence,
                    "words": result.get('words', []),
                    "duration": len(audio_data) / sample_rate
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Chunk transcription error: {e}")
            return None

# Global service instance
transcription_service = AsyncTranscriptionService()

async def transcribe_audio_with_progress(
    wav_file: str,
    model_path: str,
    language_code: str,
    sample_rate: int = 16000
) -> AsyncGenerator[Tuple[int, List[Dict]], None]:
    """Enhanced transcription (assumes model already available)"""
    try:
        logger.info(f"Starting transcription for {wav_file} with language {language_code}")
        
        # Check if Vosk is available
        if not VOSK_AVAILABLE:
            logger.warning("Vosk not available, returning dummy transcription")
            yield 100, [{"start": 0.0, "end": 1.0, "text": "Vosk speech recognition not available"}]
            return
        
        # Load model (assumes already downloaded)
        model = await transcription_service.load_language_model(language_code)
        
        # Validate audio file
        if not os.path.exists(wav_file):
            raise FileNotFoundError(f"Audio file not found: {wav_file}")
        
        if os.path.getsize(wav_file) == 0:
            raise ValueError("Audio file is empty")
        
        # Process audio with enhanced error handling
        loop = asyncio.get_event_loop()
        
        def get_wave_info():
            try:
                with wave.open(wav_file, "rb") as wf:
                    return {
                        "channels": wf.getnchannels(),
                        "sample_width": wf.getsampwidth(),
                        "framerate": wf.getframerate(),
                        "total_frames": wf.getnframes()
                    }
            except Exception as e:
                raise ValueError(f"Invalid audio file format: {str(e)}")
        
        wave_info = await loop.run_in_executor(None, get_wave_info)
        
        if wave_info["total_frames"] == 0:
            raise ValueError("Audio file contains no audio data")
        
        # Create recognizer
        recognizer = KaldiRecognizer(model, wave_info["framerate"])
        
        # Process audio in chunks with progress updates
        chunk_size = 16384  # Increased for better word recognition
        transcription_data = []
        processed_frames = 0
        
        def process_audio():
            nonlocal processed_frames
            chunk_results = []
            
            try:
                with wave.open(wav_file, "rb") as wf:
                    while True:
                        data = wf.readframes(chunk_size)
                        if len(data) == 0:
                            break
                        
                        processed_frames += len(data) // wave_info["sample_width"]
                        
                        # Process chunk
                        if recognizer.AcceptWaveform(data):
                            result = json.loads(recognizer.Result())
                            if result.get("text", "").strip():
                                chunk_results.append({
                                    "start": processed_frames / wave_info["framerate"] - 
                                            len(data) / (wave_info["sample_width"] * wave_info["framerate"]),
                                    "end": processed_frames / wave_info["framerate"],
                                    "text": result.get("text", "")
                                })
                        
                        # Yield progress
                        progress = min(100, int((processed_frames / wave_info["total_frames"]) * 100))
                        chunk_results.append(("progress", progress))
                
                # Get final result
                final_result = json.loads(recognizer.FinalResult())
                if final_result.get("text", "").strip():
                    chunk_results.append({
                        "start": max(0, processed_frames / wave_info["framerate"] - 1),
                        "end": processed_frames / wave_info["framerate"],
                        "text": final_result.get("text", "")
                    })
                
                return chunk_results
                
            except Exception as e:
                logger.error(f"Audio processing error: {e}")
                return [("error", str(e))]
        
        # Process in executor
        chunk_results = await loop.run_in_executor(None, process_audio)
        
        # Yield results
        for item in chunk_results:
            if isinstance(item, tuple):
                if item[0] == "progress":
                    yield int(item[1]), transcription_data
                elif item[0] == "error":
                    raise TranscriptionError(f"Audio processing failed: {item[1]}")
            else:
                transcription_data.append(item)
                progress = min(100, len(transcription_data) * 10)
                # Yield without filtering during processing (filtering happens at end)
                yield progress, transcription_data
        
        # Filter duplicates and clean up transcription data
        def filter_transcription_duplicates(segments):
            """Filter segments with identical text and overlapping timestamps"""
            if not segments:
                return segments
            
            filtered = []
            for current in segments:
                # Skip empty or very short transcriptions
                if not current.get('text') or len(current.get('text', '').strip()) < 2:
                    continue
                
                should_add = True
                current_text = current.get('text', '').strip().lower()
                current_start = current.get('start', 0)
                current_end = current.get('end', 0)
                
                # Check against previous segments for duplicates
                for prev in filtered[-3:]:  # Only check last 3 segments for efficiency
                    prev_text = prev.get('text', '').strip().lower()
                    prev_start = prev.get('start', 0)
                    prev_end = prev.get('end', 0)
                    
                    # Check if texts are identical and timestamps overlap
                    if current_text == prev_text:
                        # Check for timestamp overlap
                        overlap = min(current_end, prev_end) - max(current_start, prev_start)
                        if overlap > 0:  # Timestamps overlap
                            should_add = False
                            break
                
                if should_add:
                    filtered.append(current)
            
            return filtered
        
        # Apply filtering
        transcription_data = filter_transcription_duplicates(transcription_data)
        
        # Final yield
        yield 100, transcription_data
        
        logger.info(f"Transcription completed: {len(transcription_data)} segments (after filtering)")
        
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise TranscriptionError(f"Transcription error: {str(e)}")