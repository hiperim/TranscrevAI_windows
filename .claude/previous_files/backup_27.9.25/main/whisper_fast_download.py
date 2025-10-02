# Whisper Fast Download Integration
import os
import shutil
import hashlib
from pathlib import Path
from typing import Optional
import whisper
from fast_downloader import FastDownloader
from src.logging_setup import setup_app_logging

logger = setup_app_logging(logger_name="transcrevai.whisper_fast")

# Whisper model URLs and checksums
WHISPER_MODELS = {
    "tiny.en": "https://openaipublic.azureedge.net/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt",
    "tiny": "https://openaipublic.azureedge.net/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
    "base.en": "https://openaipublic.azureedge.net/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt",
    "base": "https://openaipublic.azureedge.net/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
    "small.en": "https://openaipublic.azureedge.net/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt",
    "small": "https://openaipublic.azureedge.net/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
    "medium.en": "https://openaipublic.azureedge.net/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt",
    "medium": "https://openaipublic.azureedge.net/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
    "large-v1": "https://openaipublic.azureedge.net/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt",
    "large-v2": "https://openaipublic.azureedge.net/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
    "large-v3": "https://openaipublic.azureedge.net/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
    "large": "https://openaipublic.azureedge.net/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
    "large-v3-turbo": "https://openaipublic.azureedge.net/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt",
    "turbo": "https://openaipublic.azureedge.net/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt",
}

class WhisperFastLoader:
    """Fast Whisper model loader with multi-threaded downloads"""

    def __init__(self, num_threads: int = 8):
        self.downloader = FastDownloader(num_threads=num_threads)
        self.cache_dir = self._get_cache_dir()

    def _get_cache_dir(self) -> Path:
        """Get Whisper cache directory"""
        # Windows cache location
        cache_home = os.environ.get('USERPROFILE', '')
        cache_dir = Path(cache_home) / '.cache' / 'whisper'
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _calculate_sha256(self, file_path: str) -> str:
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def _verify_model_checksum(self, model_path: str, expected_hash: str) -> bool:
        """Verify model file integrity"""
        if not os.path.exists(model_path):
            return False

        try:
            actual_hash = self._calculate_sha256(model_path)
            match = actual_hash == expected_hash

            if match:
                logger.info(f"Checksum verified: {model_path}")
            else:
                logger.warning(f"Checksum mismatch: {actual_hash} != {expected_hash}")

            return match

        except Exception as e:
            logger.error(f"Error verifying checksum: {e}")
            return False

    def _get_expected_hash(self, url: str) -> Optional[str]:
        """Extract expected hash from URL"""
        # URLs contain the hash in the path
        # e.g., .../345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt
        try:
            path_parts = url.split('/')
            hash_part = path_parts[-2]  # Hash is second to last part
            if len(hash_part) == 64:  # SHA256 is 64 hex characters
                return hash_part
        except:
            pass
        return None

    def download_model(self, model_name: str) -> Optional[str]:
        """Download Whisper model with fast multi-threaded downloader"""

        if model_name not in WHISPER_MODELS:
            logger.error(f"Unknown model: {model_name}")
            return None

        url = WHISPER_MODELS[model_name]
        model_file = f"{model_name}.pt"
        model_path = self.cache_dir / model_file
        temp_path = self.cache_dir / f"{model_file}.tmp"

        logger.info(f"Downloading Whisper model: {model_name}")

        # Check if model already exists and is valid
        expected_hash = self._get_expected_hash(url)
        if model_path.exists() and expected_hash:
            if self._verify_model_checksum(str(model_path), expected_hash):
                logger.info(f"Model already exists and verified: {model_path}")
                return str(model_path)
            else:
                logger.warning(f"Existing model failed verification, re-downloading")
                model_path.unlink()

        # Progress callback
        def progress_callback(percent: float, downloaded: int, total: int):
            mb_downloaded = downloaded / 1024 / 1024
            mb_total = total / 1024 / 1024
            logger.info(f"Download progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")

        # Download with fast downloader
        success = self.downloader.download(url, str(temp_path), progress_callback)

        if not success:
            logger.error(f"Download failed: {model_name}")
            if temp_path.exists():
                temp_path.unlink()
            return None

        # Verify downloaded file
        if expected_hash and not self._verify_model_checksum(str(temp_path), expected_hash):
            logger.error(f"Downloaded model failed verification: {model_name}")
            temp_path.unlink()
            return None

        # Move to final location
        try:
            shutil.move(str(temp_path), str(model_path))
            logger.info(f"Model download completed: {model_path}")
            return str(model_path)

        except Exception as e:
            logger.error(f"Error moving downloaded file: {e}")
            if temp_path.exists():
                temp_path.unlink()
            return None

    def load_model_fast(self, model_name: str = "medium", device: Optional[str] = None):
        """Load Whisper model with fast download if needed"""

        # First try to load normally (uses cache if available)
        try:
            logger.info(f"Attempting to load model: {model_name}")
            model = whisper.load_model(model_name, device=device)
            logger.info(f"Model loaded successfully: {model_name}")
            return model

        except Exception as e:
            logger.warning(f"Standard load failed, trying fast download: {e}")

            # Download with fast downloader
            model_path = self.download_model(model_name)
            if not model_path:
                raise Exception(f"Failed to download model: {model_name}")

            # Load from downloaded path
            try:
                model = whisper.load_model(model_path, device=device)
                logger.info(f"Model loaded after fast download: {model_name}")
                return model

            except Exception as e:
                logger.error(f"Failed to load downloaded model: {e}")
                raise


# Global instance
fast_whisper = WhisperFastLoader(num_threads=8)

def load_whisper_fast(model_name: str = "medium", device: Optional[str] = None):
    """Convenience function for fast Whisper model loading"""
    return fast_whisper.load_model_fast(model_name, device)


# Test function
def test_fast_whisper():
    """Test fast Whisper loading"""
    try:
        logger.info("Testing fast Whisper model loading...")
        model = load_whisper_fast("tiny")  # Use tiny for quick test
        logger.info(f"Fast Whisper test successful: {type(model)}")
        return True
    except Exception as e:
        logger.error(f"Fast Whisper test failed: {e}")
        return False


if __name__ == "__main__":
    test_fast_whisper()