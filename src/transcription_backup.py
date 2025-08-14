# Enhanced Transcription Module with Portuguese Model Fix

import asyncio
import wave
import json
import logging
import os
import aiofiles
import requests
import zipfile
import shutil
from typing import AsyncGenerator, Tuple, List, Dict
from pathlib import Path
from src.file_manager import FileManager
from vosk import Model, KaldiRecognizer
from config.app_config import MODEL_DIR, LANGUAGE_MODELS

logger = logging.getLogger(__name__)

class TranscriptionError(Exception):
    # Custom exception for transcription errors
    pass

class ModelManager:
    # Enhanced model management with automatic download and validation
    
    @staticmethod
    async def ensure_model_available(language_code: str) -> str:
        # Ensure language model is available, download if necessary
        model_path = os.path.join(MODEL_DIR, language_code)
        
        # Check if model already exists and is valid
        if await ModelManager._validate_model(model_path):
            logger.info(f"Valid model found for {language_code}")
            return model_path
        
        # Download and extract model
        logger.info(f"Downloading model for {language_code}")
        if language_code not in LANGUAGE_MODELS:
            raise ValueError(f"Unsupported language: {language_code}")
        
        model_url = LANGUAGE_MODELS[language_code]
        await ModelManager._download_and_extract_model(model_url, language_code, model_path)
        
        # Validate downloaded model
        if not await ModelManager._validate_model(model_path):
            raise RuntimeError(f"Downloaded model for {language_code} is invalid")
        
        return model_path
    
    @staticmethod
    async def _validate_model(model_path: str) -> bool:
        # Validate that all required model files exist
        if not os.path.exists(model_path):
            return False
        
        required_files = [
            "am/final.mdl",
            "conf/model.conf",
            "graph/phones/word_boundary.int",
            "graph/Gr.fst",
            "graph/HCLr.fst",
            "ivector/final.ie"
        ]
        
        for file_path in required_files:
            full_path = os.path.join(model_path, file_path)
            if not os.path.exists(full_path) or os.path.getsize(full_path) == 0:
                logger.warning(f"Missing or empty model file: {full_path}")
                return False
        
        return True
    
    @staticmethod
    async def _download_and_extract_model(url: str, language_code: str, model_path: str):
        # Download and extract model with enhanced error handling
        try:
            # Create temporary download directory
            temp_dir = os.path.join(MODEL_DIR, f"temp_{language_code}")
            os.makedirs(temp_dir, exist_ok=True)
            
            zip_path = os.path.join(temp_dir, f"{language_code}.zip")
            
            # Download with progress
            logger.info(f"Downloading model from {url}")
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            if downloaded % (10 * 1024 * 1024) == 0:  # Log every 10MB
                                logger.info(f"Download progress: {progress:.1f}%")
            
            logger.info("Download completed, extracting model...")
            
            # Extract model
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Determine model directory: flat or nested layout
            required = ["am/final.mdl", "conf/model.conf"]
        
            # Check flat layout (folders at temp_dir root)
            if all(os.path.isdir(os.path.join(temp_dir, folder)) for folder in ["am", "conf", "graph", "ivector"]):
                extracted_model_path = temp_dir
            else:
                # Nested layout: find first subfolder containing required files
                model_dir_name = None
                for item in os.listdir(temp_dir):
                    item_path = os.path.join(temp_dir, item)
                    if os.path.isdir(item_path) and all(os.path.exists(os.path.join(item_path, r)) for r in required):
                            model_dir_name = item_path
                            break
                if not model_dir_name:
                    raise RuntimeError("Could not locate model directory after extraction")
                extracted_model_path = os.path.join(temp_dir, model_dir_name)
            # Move to final location
            if os.path.exists(model_path):
                shutil.rmtree(model_path)
            shutil.move(extracted_model_path, model_path)
            # Clean up
            shutil.rmtree(temp_dir)
            logger.info(f"Model extracted successfully to {model_path}")
        except Exception as e:
            # Clean up on error
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            raise RuntimeError(f"Failed to download/extract model: {str(e)}")

class AsyncTranscriptionService:
    # Enhanced transcription service with automatic model management

    def __init__(self):
        self._models = {}
        self._model_lock = asyncio.Lock()

    async def load_language_model(self, language_code: str) -> Model:
        """Load or get cached language model with automatic download"""
        async with self._model_lock:
            if language_code not in self._models:
                try:
                    # Ensure model is available (download if necessary)
                    model_path = await ModelManager.ensure_model_available(language_code)
                    
                    # Load model in thread pool
                    loop = asyncio.get_event_loop()
                    model = await loop.run_in_executor(None, Model, model_path)
                    
                    self._models[language_code] = model
                    logger.info(f"Model loaded successfully for {language_code}")
                    
                except Exception as e:
                    logger.error(f"Failed to load model for {language_code}: {e}")
                    raise TranscriptionError(f"Model loading failed for {language_code}: {str(e)}")
            
            return self._models[language_code]

# Global service instance
transcription_service = AsyncTranscriptionService()

async def transcribe_audio_with_progress(
    wav_file: str,
    model_path: str,
    language_code: str,
    sample_rate: int = 16000
) -> AsyncGenerator[Tuple[int, List[Dict]], None]:
    """Enhanced transcription with automatic model management"""
    try:
        logger.info(f"Starting transcription for {wav_file} with language {language_code}")
        
        # Load model with automatic download if needed
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
        chunk_size = 4096
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
                    yield item[1], transcription_data
                elif item[0] == "error":
                    raise TranscriptionError(f"Audio processing failed: {item[1]}")
            else:
                transcription_data.append(item)
                progress = min(100, len(transcription_data) * 10)
                yield progress, transcription_data
        
        # Final yield
        yield 100, transcription_data
        logger.info(f"Transcription completed: {len(transcription_data)} segments")
        
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise TranscriptionError(f"Transcription error: {str(e)}")
