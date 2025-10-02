"""
Fase 8 - Step 3: Real ONNX Model Download System
Download and cache Whisper ONNX models with progress reporting
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Union
import time
import hashlib

# GEMINI SOLUTION: Use huggingface-hub for robust downloads
try:
    from huggingface_hub import hf_hub_download, HfApi
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

# Always import urllib for fallback downloads
import urllib.request
import urllib.parse

logger = logging.getLogger(__name__)

class ONNXModelDownloader:
    """Real ONNX model downloader with caching and progress reporting"""

    def __init__(self, cache_dir: str = "models/onnx"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # GEMINI SOLUTION: Use HuggingFace Hub repos with fallbacks
        self.hf_repos = [
            "Xenova/whisper-medium",
            "openai/whisper-medium",
            "microsoft/whisper-medium"
        ]

        self.model_configs = {
            "whisper-medium-encoder": {
                "hf_filename": "onnx/encoder_model.onnx",
                "local_filename": "whisper_medium_encoder.onnx",
                "expected_size": 1229312445,  # Actual size: ~1.2GB
                "fallback_urls": [
                    "https://huggingface.co/Xenova/whisper-medium/resolve/main/onnx/encoder_model.onnx"
                ]
            },
            "whisper-medium-decoder": {
                "hf_filename": "onnx/decoder_model_merged.onnx",
                "local_filename": "whisper_medium_decoder.onnx",
                "expected_size": 1828728265,  # Actual size: ~1.8GB
                "fallback_urls": [
                    "https://huggingface.co/Xenova/whisper-medium/resolve/main/onnx/decoder_model_merged.onnx"
                ]
            },
            "whisper-medium-tokenizer": {
                "hf_filename": "tokenizer.json",
                "local_filename": "whisper_medium_tokenizer.json",
                "expected_size": 2480466,  # Actual size: ~2.5MB
                "fallback_urls": [
                    "https://huggingface.co/openai/whisper-medium/resolve/main/tokenizer.json"
                ]
            }
        }

        self.download_progress_callback: Union[Callable, None] = None
        self.download_status: Dict[str, Any] = {}

    def set_progress_callback(self, callback: Callable[[str, int, int, int], None]):
        """Set callback for download progress reporting"""
        self.download_progress_callback = callback

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def _verify_cached_model(self, model_key: str) -> bool:
        """Verify if cached model is valid"""
        try:
            model_info = self.model_configs[model_key]
            file_path = self.cache_dir / model_info["local_filename"]

            if not file_path.exists():
                return False

            # Check file size
            file_size = file_path.stat().st_size
            expected_size = model_info["expected_size"]

            # Allow 5% variance in file size
            size_variance = 0.05
            min_size = expected_size * (1 - size_variance)
            max_size = expected_size * (1 + size_variance)

            if not (min_size <= file_size <= max_size):
                logger.warning(f"Cached model {model_key} has unexpected size: {file_size} vs expected {expected_size}")
                return False

            # GEMINI SOLUTION: ONNX validation check
            self._validate_onnx_model(file_path)

            logger.info(f"Cached model {model_key} is valid")

            logger.info(f"Cached model {model_key} is valid")
            return True

        except Exception as e:
            logger.error(f"Error verifying cached model {model_key}: {e}")
            return False

    def _validate_onnx_model(self, file_path: Path) -> bool:
        """GEMINI SOLUTION: Validate ONNX model by attempting to load it"""
        # Skip ONNX validation for JSON files (tokenizer)
        if file_path.suffix.lower() == '.json':
            try:
                import json
                with open(file_path, 'r', encoding='utf-8') as f:
                    json.load(f)  # Validate JSON structure
                logger.info(f"JSON file validated: {file_path}")
                return True
            except Exception as e:
                logger.error(f"JSON validation failed for {file_path}: {e}")
                try:
                    file_path.unlink()
                    logger.info(f"Removed corrupted JSON file: {file_path}")
                except:
                    pass
                raise e

        # ONNX validation for .onnx files
        try:
            import onnxruntime as ort
            # Quick validation: try to create session
            session = ort.InferenceSession(str(file_path), providers=['CPUExecutionProvider'])
            del session  # Clean up immediately
            return True
        except Exception as e:
            logger.error(f"ONNX validation failed for {file_path}: {e}")
            # Remove corrupted file
            try:
                file_path.unlink()
                logger.info(f"Removed corrupted ONNX file: {file_path}")
            except:
                pass
            raise e

    def _download_with_hf_hub(self, model_key: str) -> Optional[Path]:
        """GEMINI SOLUTION: Use huggingface-hub for robust downloads"""
        if not HF_HUB_AVAILABLE:
            return None

        model_config = self.model_configs[model_key]

        for repo_id in self.hf_repos:
            try:
                logger.info(f"Attempting HF Hub download: {repo_id}/{model_config['hf_filename']}")

                # Use huggingface-hub for atomic downloads with verification
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=model_config['hf_filename'],
                    cache_dir=str(self.cache_dir.parent),
                    resume_download=True,  # Automatic resume capability
                    force_download=False  # Use cache if valid
                )

                # Copy to our expected location
                final_path = self.cache_dir / model_config['local_filename']
                if Path(downloaded_path) != final_path:
                    import shutil
                    shutil.copy2(downloaded_path, final_path)

                # GEMINI SOLUTION: Validate immediately after download
                self._validate_onnx_model(final_path)

                logger.info(f"Successfully downloaded {model_key} via HF Hub")
                return final_path

            except Exception as e:
                logger.warning(f"HF Hub download failed for {repo_id}: {e}")
                continue

        return None

    def _download_with_progress(self, url: str, file_path: Path, model_key: str) -> bool:
        """Download file with progress reporting"""
        original_priority = None
        try:
            logger.info(f"Downloading {model_key} from {url}")

            # NÍVEL 1: Baixar prioridade do processo durante download
            try:
                import psutil
                process = psutil.Process()
                original_priority = process.nice()
                if hasattr(psutil, 'BELOW_NORMAL_PRIORITY_CLASS'):
                    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                elif hasattr(process, 'nice'):
                    process.nice(10)  # Lower priority on Unix
                logger.debug(f"Lowered process priority for download: {model_key}")
            except Exception as e:
                logger.warning(f"Could not lower process priority: {e}")

            # Create request with better headers for HuggingFace compatibility
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 TranscrevAI/1.0')
            req.add_header('Accept', '*/*')
            req.add_header('Accept-Encoding', 'gzip, deflate')
            req.add_header('Connection', 'keep-alive')

            # Start download with timeout
            with urllib.request.urlopen(req, timeout=30) as response:
                total_size = int(response.headers.get('Content-Length', 0))

                self.download_status[model_key] = {
                    "status": "downloading",
                    "total_size": total_size,
                    "downloaded": 0,
                    "percentage": 0,
                    "start_time": time.time()
                }

                with open(file_path, 'wb') as f:
                    downloaded = 0
                    chunk_size = 4096  # NÍVEL 1: Reduzido para 4KB para otimização de RAM
                    buffer_limit = 50 * 1024 * 1024  # NÍVEL 1: Buffer máximo de 50MB
                    buffer_used = 0
                    gc_counter = 0

                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break

                        f.write(chunk)
                        downloaded += len(chunk)
                        buffer_used += len(chunk)
                        gc_counter += 1

                        # NÍVEL 1: Flush frequente para disco quando buffer excede limite
                        if (buffer_used or 0) >= (buffer_limit or 0):
                            f.flush()
                            os.fsync(f.fileno())  # Força escrita no disco
                            buffer_used = 0
                            logger.debug(f"Buffer flushed to disk for {model_key}")

                        # NÍVEL 1: Garbage collection a cada 10% do download
                        if total_size > 0:
                            percentage = int((downloaded / total_size) * 100)
                            if percentage > 0 and percentage % 10 == 0 and gc_counter % 1000 == 0:
                                import gc
                                gc.collect()
                                logger.debug(f"GC triggered at {percentage}% for {model_key}")
                        else:
                            percentage = 0

                        self.download_status[model_key].update({
                            "downloaded": downloaded,
                            "percentage": percentage
                        })

                        # Call progress callback
                        if self.download_progress_callback:
                            self.download_progress_callback(model_key, downloaded, total_size, percentage)

                        # NÍVEL 1: Yield CPU entre chunks para não travar o sistema
                        if gc_counter % 100 == 0:  # A cada 100 chunks (~400KB)
                            time.sleep(0.001)  # 1ms yield para permitir outros processos

                        # Log progress every 10%
                        if percentage > 0 and percentage % 10 == 0:
                            mb_downloaded = downloaded / (1024 * 1024)
                            mb_total = total_size / (1024 * 1024)
                            logger.info(f"Downloading {model_key}: {percentage}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")

                # Calculate final hash
                file_hash = self._calculate_file_hash(file_path)
                # Note: SHA256 can be stored in model_configs if needed

                self.download_status[model_key].update({
                    "status": "completed",
                    "percentage": 100,
                    "end_time": time.time(),
                    "file_hash": file_hash
                })

                logger.info(f"Successfully downloaded {model_key} ({downloaded / (1024 * 1024):.1f} MB)")
                return True

        except Exception as e:
            logger.error(f"Download failed for {model_key}: {e}")
            self.download_status[model_key] = {
                "status": "failed",
                "error": str(e),
                "percentage": 0
            }

            # Clean up partial download
            if file_path.exists():
                file_path.unlink()

            return False

        finally:
            # NÍVEL 1: Restaurar prioridade original do processo
            if original_priority is not None:
                try:
                    import psutil
                    process = psutil.Process()
                    process.nice(original_priority)
                    logger.debug(f"Restored original process priority for {model_key}")
                except Exception as e:
                    logger.warning(f"Could not restore process priority: {e}")

    def download_model(self, model_key: str, force_redownload: bool = False) -> bool:
        """GEMINI SOLUTION: Robust model download with multiple fallbacks"""
        if model_key not in self.model_configs:
            raise ValueError(f"Unknown model: {model_key}")

        model_config = self.model_configs[model_key]
        file_path = self.cache_dir / model_config["local_filename"]

        # Check if model is already cached and valid
        if not force_redownload and self._verify_cached_model(model_key):
            logger.info(f"Model {model_key} already cached at {file_path}")
            self.download_status[model_key] = {
                "status": "cached",
                "percentage": 100,
                "file_path": str(file_path)
            }
            return True

        # GEMINI SOLUTION: Try HuggingFace Hub first (most robust)
        logger.info(f"Attempting robust download for {model_key}")

        # Method 1: HuggingFace Hub (automatic resume, verification, atomic writes)
        if HF_HUB_AVAILABLE:
            try:
                result_path = self._download_with_hf_hub(model_key)
                if result_path:
                    self.download_status[model_key] = {
                        "status": "completed",
                        "percentage": 100,
                        "file_path": str(result_path)
                    }
                    return True
                logger.warning("HuggingFace Hub download failed, trying fallback URLs")
            except Exception as e:
                logger.warning(f"HF Hub method failed: {e}")

        # Method 2: Fallback to direct URLs with enhanced error handling
        for i, url in enumerate(model_config["fallback_urls"]):
            try:
                logger.info(f"Attempting fallback download {i+1}/{len(model_config['fallback_urls'])} from: {url}")
                if self._download_with_progress(url, file_path, model_key):
                    # GEMINI SOLUTION: Validate after each download attempt
                    self._validate_onnx_model(file_path)
                    return True
            except Exception as e:
                logger.warning(f"Failed to download from {url}: {e}")
                # Clean up any partial/corrupted file
                if file_path.exists():
                    file_path.unlink()

        logger.error(f"All download methods failed for {model_key}")
        return False

    def download_whisper_medium_models(self, force_redownload: bool = False) -> Dict[str, bool]:
        """Download all required Whisper medium models"""
        results = {}

        logger.info("Starting download of Whisper medium ONNX models...")

        # Download encoder first (smaller, faster)
        encoder_success = self.download_model("whisper-medium-encoder", force_redownload)
        results["encoder"] = encoder_success

        # Download decoder second (larger)
        decoder_success = self.download_model("whisper-medium-decoder", force_redownload)
        results["decoder"] = decoder_success

        return results

    def get_model_path(self, model_key: str) -> Optional[Path]:
        """Get path to cached model"""
        if model_key not in self.model_configs:
            return None

        file_path = self.cache_dir / self.model_configs[model_key]["local_filename"]

        if self._verify_cached_model(model_key):
            return file_path
        else:
            return None

    def get_download_status(self, model_key: Union[str, None] = None) -> Dict[str, Any]:
        """Get download status for model(s)"""
        if model_key:
            return self.download_status.get(model_key, {"status": "not_started"})
        else:
            return self.download_status.copy()

    def clear_cache(self) -> bool:
        """Clear all cached models"""
        try:
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)

            self.download_status.clear()
            logger.info("Model cache cleared")
            return True

        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached models"""
        info = {
            "cache_dir": str(self.cache_dir),
            "models": {}
        }

        total_size = 0

        for model_key, model_info in self.model_configs.items():
            file_path = self.cache_dir / model_info["local_filename"]

            if file_path.exists():
                file_size = file_path.stat().st_size
                total_size += file_size

                info["models"][model_key] = {
                    "filename": model_info["filename"],
                    "file_size": file_size,
                    "file_size_mb": file_size / (1024 * 1024),
                    "cached": True,
                    "valid": self._verify_cached_model(model_key)
                }
            else:
                info["models"][model_key] = {
                    "filename": model_info["filename"],
                    "cached": False,
                    "valid": False
                }

        info["total_cache_size"] = total_size
        info["total_cache_size_mb"] = total_size / (1024 * 1024)

        return info

# Global instance
onnx_downloader = ONNXModelDownloader()