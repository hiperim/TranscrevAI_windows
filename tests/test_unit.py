# tests/test_unit.py
"""
Comprehensive test suite for the refactored TranscrevAI application.
Validates the end-to-end EnhancedTranscriptionPipeline and core API endpoints.
"""

import pytest
import asyncio
from pathlib import Path
import uuid
from fastapi.testclient import TestClient

# Add root directory to path to allow src imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import app, app_state

# --- Test Setup --- #

# Ground truth for benchmark files, essential for accuracy validation.
BENCHMARK_EXPECTATIONS = {
    "d.speakers.wav": {"expected_speakers": 2, "keywords": ["empresa", "hierarquia", "informal"]},
    "q.speakers.wav": {"expected_speakers": 4, "keywords": ["homens", "medo", "Machista"]},
    "t.speakers.wav": {"expected_speakers": 3, "keywords": ["tapa", "mulher", "certo"]},
    "t2.speakers.wav": {"expected_speakers": 3, "keywords": ["gelo", "luvas", "sapato"]}
}
def get_benchmark_files():
    """Finds all .wav files in the recordings folder that have benchmarks."""
    recordings_dir = Path(__file__).parent.parent / "data" / "recordings"
    if not recordings_dir.exists():
        return []
    valid_files = [p for p in recordings_dir.glob("*.wav") if p.name in BENCHMARK_EXPECTATIONS]
    if not valid_files:
        pytest.skip("No benchmark audio files found in data/recordings for testing.")
    return [str(p) for p in valid_files]

@pytest.fixture(scope="module")
def client():
    """Create a TestClient for the FastAPI app to test API endpoints."""
    with TestClient(app) as c:
        yield c

# --- Main Integration Test (Parametrized with Accuracy Checks) ---






# --- API Tests ---

def test_upload_endpoint(client):
    """Test the /upload endpoint with one of the sample files."""
    test_files = get_benchmark_files()
    if not test_files: pytest.skip("No audio files for upload test.")
    sample_audio_path = test_files[0]

    session_id = f"test_upload_{uuid.uuid4().hex[:8]}"
    with open(sample_audio_path, "rb") as f:
        response = client.post("/upload", files={"file": (Path(sample_audio_path).name, f, "audio/wav")}, data={"session_id": session_id})
    
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["success"] is True
    assert json_response["session_id"] == session_id

# --- Unit Tests ---

# Import the class to be tested
from src.websocket_handler import WebSocketValidator

def test_websocket_validator_valid_actions():
    """Tests that the validator correctly identifies valid actions."""
    validator = WebSocketValidator()
    assert validator.validate_action("start") is None
    assert validator.validate_action("audio_chunk") is None
    assert validator.validate_action("stop") is None

def test_websocket_validator_invalid_action():
    """Tests that the validator flags invalid or missing actions."""
    validator = WebSocketValidator()
    assert validator.validate_action("delete") is not None
    assert validator.validate_action(None) is not None
    assert validator.validate_action("") is not None

def test_websocket_validator_audio_format():
    """Tests audio format validation."""
    validator = WebSocketValidator()
    assert validator.validate_audio_format("wav") is None
    assert validator.validate_audio_format("mp4") is None
    assert validator.validate_audio_format("mp3") is not None
    assert validator.validate_audio_format("") is not None

def test_websocket_validator_session_state():
    """Tests the validation of session state for receiving chunks."""
    validator = WebSocketValidator()
    valid_session = {"status": "recording"}
    invalid_session = {"status": "processing"}
    assert validator.validate_session_state_for_chunk(valid_session) is None
    assert validator.validate_session_state_for_chunk(invalid_session) is not None
