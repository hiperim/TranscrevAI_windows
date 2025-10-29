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

# (Add this import at the top of tests/test_unit.py)
import multiprocessing
from queue import Empty

from src.worker import process_audio_task


# --- New Main Integration Test ---

@pytest.mark.parametrize("audio_file_path", get_benchmark_files())
def test_worker_process_audio_task_accuracy(worker_services_fixture, audio_file_path: str):
    """
    Tests the core `process_audio_task` in the worker module to validate
    the entire pipeline's accuracy from transcription to diarization.
    """
    file_name = Path(audio_file_path).name
    expectations = BENCHMARK_EXPECTATIONS[file_name]
    test_session_id = f"test_worker_{Path(audio_file_path).stem}"

    # The worker communicates via a queue, so we must simulate it
    manager = multiprocessing.Manager()
    communication_queue = manager.Queue()

    # Mock the app config required by the worker
    mock_config = {
        "model_name": "medium",
        "device": "cpu"
    }

    # Execute the worker task directly
    process_audio_task(
        audio_path=audio_file_path,
        session_id=test_session_id,
        config=mock_config,
        communication_queue=communication_queue
    )

    # Retrieve the final result from the queue
    final_result = None
    while True:
        try:
            message = communication_queue.get(timeout=60) # 60-second timeout
            if message.get('type') == 'complete':
                final_result = message.get('result')
                break
            if message.get('type') == 'error':
                pytest.fail(f"Worker task failed for {file_name}: {message.get('message')}")
        except Empty:
            pytest.fail(f"Worker task timed out for {file_name}. No 'complete' message received.")

    # --- Validate the Final Result ---
    assert final_result is not None, f"Worker did not produce a final result for {file_name}"

    # 1. Validate Diarization Accuracy
    detected_speakers = final_result.get("num_speakers", 0)
    expected_speakers = expectations["expected_speakers"]
    assert abs(detected_speakers - expected_speakers) <= 1, f"Speaker count mismatch for {file_name}. Expected {expected_speakers} (tolerance: +/-1), got {detected_speakers}."

    # 2. Validate Transcription Accuracy (by checking segments)
    full_text = " ".join(seg.get('text', '') for seg in final_result.get("segments", [])).lower()
    assert len(full_text) > 10, f"Transcription text is too short for {file_name}"

    missing_keywords = [kw for kw in expectations["keywords"] if kw.lower() not in full_text]
    assert len(missing_keywords) <= 1, f"Transcription for {file_name} is missing too many keywords. Expected at most 1 missing, but found {len(missing_keywords)}: {', '.join(missing_keywords)}"

    # 3. Validate SRT Generation
    assert "srt_path" in final_result and final_result["srt_path"], f"SRT file path missing for {file_name}"
    srt_path = Path(final_result["srt_path"])
    assert srt_path.exists(), f"SRT file was not created for {file_name}"
    assert srt_path.stat().st_size > 50, f"SRT file is empty for {file_name}"

    # Cleanup the generated SRT file
    srt_path.unlink()

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
