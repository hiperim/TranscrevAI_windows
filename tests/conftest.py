# tests/conftest.py
"""
Pytest configuration and fixtures for TranscrevAI tests
Enhanced for the new unified pipeline architecture.
"""

import pytest
import asyncio
from pathlib import Path
from typing import Iterator
import shutil
import tempfile


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the session."""
    try:
        loop = asyncio.get_event_loop_policy().new_event_loop()
        yield loop
    finally:
        loop.close()

@pytest.fixture(scope="module")
def sample_recordings_path() -> Path:
    """Provides the absolute path to the data/recordings directory."""
    path = Path(__file__).parent.parent / "data" / "recordings"
    if not path.exists():
        pytest.skip(f"Test recordings directory not found at: {path}")
    return path

@pytest.fixture(scope="module")
def temp_output_dir() -> Iterator[Path]:
    """Create a temporary directory for test outputs."""
    temp_dir = Path(tempfile.mkdtemp(prefix="transcrevai_test_"))
    try:
        yield temp_dir
    finally:
        # Cleanup after tests are done
        shutil.rmtree(temp_dir)


