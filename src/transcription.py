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
    pass

class ModelManager:
    
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
        
        # Check for traditional Kaldi structure first
        required_dirs = ["am", "conf", "graph", "ivector"]
        has_traditional_structure = all(
            os.path.exists(os.path.join(model_path, dir_name)) and 
            os.path.isdir(os.path.join(model_path, dir_name))
            for dir_name in required_dirs
        )
        
        if has_traditional_structure:
            # Check for essential files in traditional structure
            essential_files = [
                "am/final.mdl",
                "conf/model.conf"
            ]
            
            for file_path in essential_files:
                full_path = os.path.join(model_path, file_path)
                if not os.path.exists(full_path) or os.path.getsize(full_path) == 0:
                    logger.warning(f"Missing or empty essential model file: {full_path}")
                    return False
            
            logger.info(f"Traditional model validation passed for: {model_path}")
            return True
        
        # Check for Vosk model structure
        vosk_model_files = [
            "final.mdl",     # Acoustic model
            "Gr.fst",        # Grammar FST
            "HCLr.fst",      # HCL FST
            "mfcc.conf"      # MFCC configuration
        ]
        
        # Look for essential vosk files in the main directory
        found_files = []
        for root, dirs, files in os.walk(model_path):
            for file in files:
                if file in vosk_model_files:
                    found_files.append(file)
                    logger.debug(f"Found vosk model file: {file} in {root}")
        
        # We need at least final.mdl and one FST file for a valid vosk model
        required_vosk_files = ["final.mdl"]
        fst_files = ["Gr.fst", "HCLr.fst"]
        
        has_required = all(f in found_files for f in required_vosk_files)
        has_fst = any(f in found_files for f in fst_files)
        
        if has_required and has_fst:
            logger.info(f"Vosk model validation passed for: {model_path}")
            return True
        
        logger.warning(f"Model validation failed for: {model_path}")
        logger.warning(f"Found files: {found_files}")
        logger.warning(f"Required files: {required_vosk_files + fst_files}")
        return False
    
    @staticmethod
    def _validate_zip_file(zip_path: str):
        # Validate ZIP file integrity and required content
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Test the ZIP file integrity
                bad_files = zip_ref.testzip()
                if bad_files:
                    raise RuntimeError(f"ZIP file contains corrupted files: {bad_files}")
                
                # Check if ZIP contains expected model directories
                all_files = zip_ref.namelist()
                required_dirs = ["am/", "conf/", "graph/", "ivector/"]
                found_dirs = []
                
                for file_path in all_files:
                    for req_dir in required_dirs:
                        if req_dir in file_path and req_dir not in found_dirs:
                            found_dirs.append(req_dir)
                
                missing_dirs = [d for d in required_dirs if d not in found_dirs]
                if missing_dirs:
                    logger.warning(f"ZIP file may be incomplete, missing directories: {missing_dirs}")
                    # Don't fail here, as directory structure might vary
                
                logger.info(f"ZIP file validation passed: {len(all_files)} files")
                
        except zipfile.BadZipFile:
            raise RuntimeError("Invalid or corrupted ZIP file")
        except Exception as e:
            raise RuntimeError(f"ZIP validation failed: {e}")

    
    @staticmethod
    async def _download_and_extract_model(url: str, language_code: str, model_path: str):
        # Download and extract model with enhanced error handling
        try:
            # Create temporary download directory
            temp_dir = os.path.join(MODEL_DIR, f"temp_{language_code}")
            os.makedirs(temp_dir, exist_ok=True)
            
            zip_path = os.path.join(temp_dir, f"{language_code}.zip")
            
            # Download with progress and retry mechanism
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
                    
                    # Test ZIP file integrity
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

            # Extract model with enhanced error handling
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    # Get all files in the ZIP
                    all_files = zip_ref.namelist()
                    logger.info(f"ZIP contains {len(all_files)} files")
                    
                    # Extract all files
                    await asyncio.to_thread(zip_ref.extractall, temp_dir)
                    
                    # Verify extraction
                    extracted_files = []
                    for root, dirs, files in os.walk(temp_dir):
                        for file in files:
                            extracted_files.append(os.path.join(root, file))
                    
                    logger.info(f"Extracted {len(extracted_files)} files successfully")
                    
            except zipfile.BadZipFile as e:
                raise RuntimeError(f"ZIP file is corrupted: {e}")
            except Exception as e:
                raise RuntimeError(f"Extraction failed: {e}")

            # Create the final model directory
            if os.path.exists(model_path):
                await asyncio.to_thread(shutil.rmtree, model_path)
            await asyncio.to_thread(os.makedirs, model_path, exist_ok=True)

            required_dirs = ["am", "conf", "graph", "ivector"]

            logger.info(f"Searching for required directories in: {temp_dir}")

            def find_model_directories():
                required_dirs = ["am", "conf", "graph", "ivector"]
                
                # First, try to find a directory with all required subdirs
                for root, dirs, files in os.walk(temp_dir):
                    logger.debug(f"Scanning directory: {root}")
                    logger.debug(f"Found subdirs: {dirs}")
                    logger.debug(f"Found files: {files[:10]}...")  # Show first 10 files
                    # Check if ALL required directories exist at this level
                    if all(folder in dirs for folder in required_dirs):
                        logger.info(f"Found complete model directory at: {root}")
                        return root
                
                # If not found, try to find a vosk model directory pattern
                for root, dirs, files in os.walk(temp_dir):
                    dir_name = os.path.basename(root).lower()
                    if 'vosk' in dir_name and 'model' in dir_name:
                        # Check if this directory contains model files
                        model_files = ['final.mdl', 'Gr.fst', 'HCLr.fst', 'mfcc.conf']
                        if any(f in files for f in model_files):
                            logger.info(f"Found vosk model directory (alternate structure): {root}")
                            return root
                        # Check subdirectories for model files
                        for subdir in dirs:
                            subdir_path = os.path.join(root, subdir)
                            try:
                                subdir_files = os.listdir(subdir_path)
                                if any(f in subdir_files for f in model_files):
                                    logger.info(f"Found vosk model directory in subdir: {subdir_path}")
                                    return subdir_path
                            except (OSError, PermissionError):
                                continue
                
                # Enhanced error logging if no complete directory found
                logger.error("Failed to find complete model directory")
                logger.error("Directory structure analysis:")
                for root, dirs, files in os.walk(temp_dir):
                    level = root.replace(temp_dir, '').count(os.sep)
                    indent = ' ' * 2 * level
                    logger.error(f"{indent}{os.path.basename(root)}/")
                    subindent = ' ' * 2 * (level + 1)
                    for dir_name in dirs:
                        logger.error(f"{subindent}{dir_name}/")
                    for file_name in files[:5]:  # Show first 5 files
                        logger.error(f"{subindent}{file_name}")
                    if len(files) > 5:
                        logger.error(f"{subindent}... and {len(files) - 5} more files")
                return None

            # Run directory search in thread pool
            model_source_dir = await asyncio.to_thread(find_model_directories)

            if not model_source_dir:
                raise RuntimeError(f"Could not locate directory containing all required model folders: {required_dirs}")

            # Check if we found a traditional structure or vosk model structure
            has_traditional_structure = all(
                os.path.exists(os.path.join(model_source_dir, folder)) 
                for folder in required_dirs
            )
            
            if has_traditional_structure:
                # Copy all required directories asynchronously
                async def copy_model_directory(src_folder, dst_folder):
                    src_path = os.path.join(model_source_dir, src_folder)
                    dst_path = os.path.join(model_path, dst_folder)
                    
                    # Remove destination if it exists
                    if os.path.exists(dst_path):
                        await asyncio.to_thread(shutil.rmtree, dst_path)
                    
                    # Copy directory
                    await asyncio.to_thread(shutil.copytree, src_path, dst_path)
                    logger.info(f"Copied {src_folder} from {src_path} to {dst_path}")

                # Copy all directories concurrently
                copy_tasks = [copy_model_directory(folder, folder) for folder in required_dirs]
                await asyncio.gather(*copy_tasks)

                missing = []
                for folder in required_dirs:
                    dst_path = os.path.join(model_path, folder)
                    if not os.path.exists(dst_path):
                        missing.append(folder)

                if missing:
                    raise RuntimeError(f"Failed to copy model folders: {missing}")
            else:
                # Copy the entire vosk model directory
                logger.info(f"Copying entire vosk model from {model_source_dir} to {model_path}")
                if os.path.exists(model_path):
                    await asyncio.to_thread(shutil.rmtree, model_path)
                await asyncio.to_thread(shutil.copytree, model_source_dir, model_path)
                logger.info(f"Successfully copied vosk model to {model_path}")
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
        # Load or get cached language model with automatic download
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
    sample_rate: int = 16000) -> AsyncGenerator[Tuple[int, List[Dict]], None]:
    # Enhanced transcription with automatic model management
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
                        "total_frames": wf.getnframes()}
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
                                    "text": result.get("text", "")})
                        
                        # Yield progress
                        progress = min(100, int((processed_frames / wave_info["total_frames"]) * 100))
                        chunk_results.append(("progress", progress))
                
                # Get final result
                final_result = json.loads(recognizer.FinalResult())
                if final_result.get("text", "").strip():
                    chunk_results.append({
                        "start": max(0, processed_frames / wave_info["framerate"] - 1),
                        "end": processed_frames / wave_info["framerate"],
                        "text": final_result.get("text", "")})
                
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