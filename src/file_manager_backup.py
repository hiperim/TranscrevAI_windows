import os
import logging
import zipfile
import requests
import asyncio
import time
import tempfile
import sys
import shutil   
from pathlib import Path
from typing import Union
from src.logging_setup import setup_app_logging
from config.app_config import APP_PACKAGE_NAME

logger = setup_app_logging()

def sanitize_path(user_input, base_dir):  
    resolved_path = Path(base_dir).joinpath(user_input).resolve()  
    if not resolved_path.is_relative_to(Path(base_dir).resolve()):  
        raise SecurityError("Attempted directory traversal")  
    return str(resolved_path)  

class SecurityError(RuntimeError):
    def __init__(self, message):
        logger.error(f"Security violation: {message}")
        super().__init__(message)

class FileManager():
    @staticmethod
    def get_base_directory(subdir=""):
        base = Path(__file__).resolve().parent.parent.parent
        return str(base / subdir) if subdir else str(base)
        
    @staticmethod
    def get_data_path(subdir="") -> str:
        base = Path(__file__).resolve().parent.parent.parent / "data"
        full_path = base / subdir
        return os.path.normpath(str(full_path)) # was: "return full_path.as_posix()"

    @staticmethod
    def get_unified_temp_dir() -> str:
        base_temp = FileManager.get_data_path("temp")
        FileManager.ensure_directory_exists(base_temp)
        temp_dir = tempfile.mkdtemp(dir=base_temp, prefix=f"temp_{os.getpid()}_",suffix=f"_{int(time.time())}")
        return FileManager.validate_path(temp_dir)

    @staticmethod
    def ensure_directory_exists(path):
        try:
            os.makedirs(path, exist_ok=True)
        except Exception as e:
            logger.error(f"Directory creation failed: {path}")
            raise RuntimeError(f"Filesystem error: {str(e)}")
        
    @staticmethod
    def validate_path(user_path: str) -> str:
        try:
            resolved = Path(user_path).resolve(strict=False)
            allowed_dirs = []
            # Desktop paths
            base_dir = Path(__file__).parent.parent.parent / "data"
            allowed_dirs.append(base_dir)
            # Temporary directory is also allowed
            import tempfile
            allowed_dirs.append(Path(tempfile.gettempdir()))
            # Require at least one valid allowed directory
            if not allowed_dirs:
                raise SecurityError("No valid directories configured")
            # Check that path is under an allowed directory
            if not any(resolved.is_relative_to(d) for d in allowed_dirs if d.exists()):
                logger.error(f"Path validation failed: {resolved} not in allowed directories")
                raise SecurityError(f"Path violation: {resolved}")
            return str(resolved)
        except ValueError as e:
            logger.error(f"Path validation failed: {e}")
            raise SecurityError("Invalid path") from e

    @staticmethod
    def save_audio(data, filename="output.wav") -> str:
        try:
            safe_dir = FileManager.validate_path("inputs")
            output_path = os.path.join(safe_dir, filename)
            FileManager.ensure_directory_exists(os.path.dirname(output_path))
            with open(output_path, 'wb') as f:
                f.write(data)
            logger.info(f"Audio file saved: {output_path}")
            return output_path
        except OSError as ose:
            logger.error(f"File system error: {ose.strerror}")
            raise 
        
    @staticmethod
    def save_transcript(data: Union[str, list], filename="output.txt") -> None:
        try:
            output_dir = FileManager.get_data_path("transcripts")
            output_path = os.path.join(output_dir, filename)
            FileManager.ensure_directory_exists(os.path.dirname(output_path))
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(str(data))
            logger.info(f"Transcript saved: {output_path}")
        except Exception as e:
            logger.error(f"Transcript save failed: {e}")
            raise

    @staticmethod
    def _sync_download_and_extract(url, language_code, output_dir):
        model_path = os.path.join(output_dir, language_code)
        if os.path.exists(model_path) and any(os.listdir(model_path)):
            logger.info(f"Existing model found: {language_code}")
            return model_path
        zip_path = os.path.join(output_dir, f"{language_code}.zip")
        for attempt in range(3):
            try:
                logger.info(f"Downloading model: {language_code}")
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        f.write(chunk)
                # Create a temporary directory for extraction
                temp_extract_dir = os.path.join(output_dir, f"temp_{language_code}")
                if os.path.exists(temp_extract_dir):
                    shutil.rmtree(temp_extract_dir)
                os.makedirs(temp_extract_dir, exist_ok=True)
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(temp_extract_dir)
                # Checks if there is only one file and if folder is empty
                contents = os.listdir(temp_extract_dir)
                if len(contents) == 1 and os.path.isdir(os.path.join(temp_extract_dir, contents[0])):
                    nested_dir = os.path.join(temp_extract_dir, contents[0])
                    # Create the final model directory
                    if os.path.exists(model_path):
                        shutil.rmtree(model_path)
                    os.makedirs(model_path, exist_ok=True)
                    # Only copy the required folders
                    required_folders = ["am", "conf", "graph", "ivector"]
                    for folder in required_folders:
                        src_folder = os.path.join(nested_dir, folder)
                        dst_folder = os.path.join(model_path, folder)
                        if os.path.exists(src_folder):
                            shutil.copytree(src_folder, dst_folder)
                else:
                    # No nested directory: directly copy required folders
                    if os.path.exists(model_path):
                        shutil.rmtree(model_path)
                    os.makedirs(model_path, exist_ok=True)
                    required_folders = ["am", "conf", "graph", "ivector"]
                    for folder in required_folders:
                        src_folder = os.path.join(temp_extract_dir, folder)
                        dst_folder = os.path.join(model_path, folder)
                        if os.path.exists(src_folder):
                            shutil.copytree(src_folder, dst_folder)
                    # Clean up
                if os.path.exists(temp_extract_dir):
                    shutil.rmtree(temp_extract_dir)
                os.remove(zip_path)
                logger.info(f"Model extracted: {model_path}")
                # Verify required files exist
                required_files = ["am/final.mdl", "conf/model.conf", "graph/phones/word_boundary.int", "graph/Gr.fst", "graph/HCLr.fst", "ivector/final.ie"]
                missing = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
                if missing:
                    logger.warning(f"Some model files missing after extraction: {missing}")
                return model_path
            except Exception as e:
                logger.error(f"Model download failed on attempt {attempt + 1}: {e}")
                time.sleep(2 * (attempt + 1)) 
        if os.path.exists(zip_path):
            os.remove(zip_path)
            raise RuntimeError(f"Failed to download and extract model")

    @staticmethod
    async def download_and_extract_model(url, language_code, output_dir):
        from urllib.parse import urlparse
        parsed = urlparse(url)
        if not parsed.scheme.startswith('http'):
            raise ValueError("Invalid model URL")
        with requests.Session() as session:
            session.mount(url, requests.adapters.HTTPAdapter(max_retries=3))
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, FileManager._sync_download_and_extract, url, language_code, output_dir)
    
    """Could this replace both _sync_download_and_extract and download_and_extract_model???"""
    """@staticmethod
    async def download_and_extract_model(url, language_code, output_dir):
        from urllib.parse import urlparse
        parsed = urlparse(url)
        if not parsed.scheme.startswith('http'):
            raise ValueError("Invalid model URL")

        model_path = os.path.join(output_dir, language_code)
        if os.path.exists(model_path):
            return model_path

        zip_path = os.path.join(output_dir, f"{language_code}.zip")
        
        async with requests.Session() as session:
            async with session.get(url, timeout=30) as response:
                response.raise_for_status()
                with open(zip_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(1024):
                        f.write(chunk)"""
    
    @staticmethod 
    def cleanup_temp_dirs():
        base_temp = FileManager.get_data_path("temp")
        for temp_dir in os.listdir(base_temp):
            dir_path = os.path.join(base_temp, temp_dir)
            try:
                if os.path.isdir(dir_path):
                    shutil.rmtree(dir_path)
            except Exception as e:
                logger.warning(f"Temp cleanup failed: {dir_path}")