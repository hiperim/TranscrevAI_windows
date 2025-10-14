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
from src.worker import init_worker

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

# --- New Fixture for Production-Ready Testing ---

@pytest.fixture(scope="session")
def worker_services_fixture():
    """
    Session-scoped fixture to initialize the worker services ONCE.
    This simulates the production environment where models are loaded
    once when the worker process starts. This is essential for getting
    accurate performance metrics.
    """
    print("\n--- (SESSION START) Initializing worker services for the entire test session ---")
    mock_config = {
        "model_name": "medium",
        "device": "cpu"
    }
    init_worker(mock_config)
    print("--- (SESSION START) Worker services initialized. Running tests... ---")
    yield
    print("\n--- (SESSION END) All tests finished. ---")
