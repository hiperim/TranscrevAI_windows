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

class ModelManager:
    
    @staticmethod
    async def ensure_model_available(language_code: str) -> str:
        """Ensure language model is available, download if necessary"""
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
        """Simple model validation - only check for essential file"""
        if not os.path.exists(model_path):
            return False
        
        # Only check for the one file that always exists in Vosk models
        essential_file = os.path.join(model_path, "final.mdl")
        if os.path.exists(essential_file) and os.path.getsize(essential_file) > 0:
            logger.info(f"Vosk model validation passed for: {model_path}")
            return True
        
        logger.warning(f"Model validation failed - missing essential files in: {model_path}")
        return False
    
    @staticmethod
    def _validate_zip_file(zip_path: str):
        """Validate ZIP file integrity"""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Test the ZIP file integrity
                bad_files = zip_ref.testzip()
                if bad_files:
                    raise RuntimeError(f"ZIP file contains corrupted files: {bad_files}")
                
                # Check if ZIP contains some model files
                all_files = zip_ref.namelist()
                
                # Look for key Vosk model files
                key_files = ["final.mdl"]
                found_files = []
                
                for file_path in all_files:
                    file_name = os.path.basename(file_path)
                    if file_name in key_files:
                        found_files.append(file_name)
                
                if not found_files:
                    logger.warning("ZIP file may not contain a valid Vosk model")
                else:
                    logger.info(f"ZIP file validation passed: found model files")
                    
        except zipfile.BadZipFile:
            raise RuntimeError("Invalid or corrupted ZIP file")
        except Exception as e:
            raise RuntimeError(f"ZIP validation failed: {e}")
    
    @staticmethod
    async def _download_and_extract_model(url: str, language_code: str, model_path: str):
        """Download and extract Vosk model with enhanced error handling"""
        try:
            # Create temporary download directory
            temp_dir = os.path.join(MODEL_DIR, f"temp_{language_code}")
            os.makedirs(temp_dir, exist_ok=True)
            zip_path = os.path.join(temp_dir, f"{language_code}.zip")
            
            # Download with retry mechanism
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logger.info(f"Downloading model from {url} (attempt {attempt + 1}/{max_retries})")
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
                    
                    # Validate downloaded file size
                    if total_size > 0 and downloaded != total_size:
                        raise RuntimeError(f"Download incomplete: {downloaded}/{total_size} bytes")
                    
                    # Validate ZIP file integrity
                    if not await asyncio.to_thread(zipfile.is_zipfile, zip_path):
                        raise RuntimeError("Downloaded file is not a valid ZIP file")
                    
                    logger.info("Download completed, validating ZIP file...")
                    await asyncio.to_thread(ModelManager._validate_zip_file, zip_path)
                    logger.info("ZIP file validation passed, extracting model...")
                    
                    break
                    
                except Exception as e:
                    logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2)  # Wait before retry
                        continue
                    else:
                        raise RuntimeError(f"Download failed after {max_retries} attempts: {e}")
            
            # Extract model
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    all_files = zip_ref.namelist()
                    logger.info(f"ZIP contains {len(all_files)} files")
                    
                    # Extract all files to temp directory
                    await asyncio.to_thread(zip_ref.extractall, temp_dir)
                    
                    # Verify extraction
                    extracted_files = []
                    for root, dirs, files in os.walk(temp_dir):
                        for file in files:
                            if not file.endswith('.zip'):  # Skip the ZIP file itself
                                extracted_files.append(os.path.join(root, file))
                    
                    logger.info(f"Extracted {len(extracted_files)} files successfully")
                    
            except zipfile.BadZipFile as e:
                raise RuntimeError(f"ZIP file is corrupted: {e}")
            except Exception as e:
                raise RuntimeError(f"Extraction failed: {e}")
            
            # Find and move model files
            if os.path.exists(model_path):
                await asyncio.to_thread(shutil.rmtree, model_path)
            await asyncio.to_thread(os.makedirs, model_path, exist_ok=True)
            
            def find_and_move_model_files():
                """Find model files in extracted directory and move to final location"""
                # Look for directory containing final.mdl or just final.mdl itself
                model_source_dir = None
                
                for root, dirs, files in os.walk(temp_dir):
                    if 'final.mdl' in files:
                        model_source_dir = root
                        logger.info(f"Found model files in: {model_source_dir}")
                        break
                
                if not model_source_dir:
                    # Look for any directory that contains model-like files
                    for root, dirs, files in os.walk(temp_dir):
                        # Look for directories with many files (likely model directories)
                        if len(files) > 5:  
                            model_source_dir = root
                            logger.info(f"Found potential model directory: {model_source_dir}")
                            break
                
                if not model_source_dir:
                    raise RuntimeError("Could not find model files in extracted archive")
                
                # Move all files from source to destination
                for item in os.listdir(model_source_dir):
                    src_path = os.path.join(model_source_dir, item)
                    dst_path = os.path.join(model_path, item)
                    
                    if os.path.isdir(src_path):
                        # Copy directory (like ivector/)
                        if os.path.exists(dst_path):
                            shutil.rmtree(dst_path)
                        shutil.copytree(src_path, dst_path)
                    else:
                        # Copy file
                        shutil.copy2(src_path, dst_path)
                
                logger.info(f"Successfully moved model files to {model_path}")
                
                # Verify at least the essential file exists
                essential_file = os.path.join(model_path, "final.mdl")
                if not os.path.exists(essential_file):
                    # Try to find it in subdirectories
                    for root, dirs, files in os.walk(model_path):
                        if 'final.mdl' in files:
                            logger.info(f"Found final.mdl in subdirectory: {root}")
                            return model_path
                    
                    raise RuntimeError("Missing essential model file after extraction: final.mdl")
                
                return model_path
            
            # Run file operations in thread pool
            await asyncio.to_thread(find_and_move_model_files)
            
            # Clean up
            shutil.rmtree(temp_dir)
            
            logger.info(f"Model extracted successfully to {model_path}")
            
        except Exception as e:
            # Clean up on error
            cleanup_paths = []
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                cleanup_paths.append(temp_dir)
            
            for cleanup_path in cleanup_paths:
                try:
                    shutil.rmtree(cleanup_path)
                except Exception as cleanup_error:
                    logger.warning(f"Cleanup failed: {cleanup_error}")
            
            raise RuntimeError(f"Failed to download/extract model: {str(e)}")

class AsyncTranscriptionService:
    """Enhanced transcription service with automatic model management"""
    
    def __init__(self):
        self._models = {}
        self._model_lock = asyncio.Lock()
    
    async def load_language_model(self, language_code: str) -> Any:
        """Load or get cached language model with automatic download"""
        if not VOSK_AVAILABLE:
            logger.warning("Vosk not available, returning dummy model")
            return Model()  # Return dummy model
        
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
        
        # Check if Vosk is available
        if not VOSK_AVAILABLE:
            logger.warning("Vosk not available, returning dummy transcription")
            yield 100, [{"start": 0.0, "end": 1.0, "text": "Vosk speech recognition not available"}]
            return
        
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
                    yield int(item[1]), transcription_data
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