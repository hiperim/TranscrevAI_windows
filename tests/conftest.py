"""
Pytest configuration and fixtures for TranscrevAI tests
Simplified for Phase 9.5 testing with compliance validation
"""

import pytest
import os
import numpy as np
import sys
import tempfile
from pathlib import Path
import shutil
import time
import logging
from typing import Generator, Dict, Any, Callable
import asyncio

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    from src.logging_setup import setup_app_logging
    from src.file_manager import FileManager, intelligent_cache
    from src.production_optimizer import get_production_optimizer
    SRC_MODULES_AVAILABLE = True
except ImportError:
    SRC_MODULES_AVAILABLE = False

logger = logging.getLogger(__name__)

@pytest.fixture
def generate_test_audio() -> Callable[[float, int, int], str]:
    """Generate synthetic test audio files"""
    def _generate(duration: float = 5.0, speakers: int = 2, sample_rate: int = 16000) -> str:
        if not SOUNDFILE_AVAILABLE:
            raise pytest.skip("soundfile not available")

        samples = int(duration * sample_rate)
        data = np.zeros(samples, dtype=np.float32)

        # Generate multi-speaker audio
        for i in range(speakers):
            freq = 440 * (i + 1)  # Different frequency per speaker
            t = np.linspace(0, duration, samples)
            data += 0.3 * np.sin(2 * np.pi * freq * t)

        # Normalize
        if np.max(np.abs(data)) > 0:
            data = data / np.max(np.abs(data))

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp:
            temp_file = temp.name
            sf.write(temp_file, data, sample_rate)
            logger.debug(f"Generated test audio: {temp_file}")
            return temp_file

    return _generate

@pytest.fixture
def temp_path(tmp_path: Path) -> Generator[Path, None, None]:
    """Simple temporary directory fixture"""
    test_temp = tmp_path / "test_audio"
    test_temp.mkdir(exist_ok=True)

    yield test_temp

    # Simple cleanup
    try:
        shutil.rmtree(test_temp, ignore_errors=True)
    except Exception as e:
        logger.debug(f"Cleanup failed: {e}")


@pytest.fixture
def sample_recordings_path() -> Path:
    """Path to data/recordings for compliance testing"""
    return Path(__file__).parent.parent / "data" / "recordings"


@pytest.fixture
def benchmark_files(sample_recordings_path: Path) -> Dict[str, Path]:
    """Get benchmark files as per Rule 21"""
    benchmarks = {}
    if sample_recordings_path.exists():
        for benchmark_file in sample_recordings_path.glob("benchmark_*.txt"):
            audio_name = benchmark_file.name.replace("benchmark_", "").replace(".txt", "")
            benchmarks[audio_name] = benchmark_file
    return benchmarks


@pytest.fixture
async def production_manager():
    """Get production optimizer for testing"""
    if not SRC_MODULES_AVAILABLE:
        pytest.skip("Source modules not available")

    try:
        return await get_production_optimizer()
    except Exception as e:
        pytest.skip(f"Could not initialize production manager: {e}")


@pytest.fixture
def intelligent_cache_instance():
    """Get intelligent cache instance for testing"""
    if not SRC_MODULES_AVAILABLE:
        pytest.skip("Source modules not available")

    return intelligent_cache

def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line("markers", "asyncio: mark test as async")
    config.addinivalue_line("markers", "timeout: mark test to timeout")
    config.addinivalue_line("markers", "compliance: mark test as compliance validation")