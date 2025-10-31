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
from src.exceptions import ValidationError
from src.error_messages import get_user_message

def test_websocket_validator_valid_actions():
    """Tests that the validator correctly identifies valid actions."""
    validator = WebSocketValidator()
    assert validator.validate_action("start") is None
    assert validator.validate_action("audio_chunk") is None
    assert validator.validate_action("stop") is None

def test_websocket_validator_invalid_action_raises_exception():
    """Tests that the validator raises ValidationError for invalid or missing actions."""
    validator = WebSocketValidator()
    with pytest.raises(ValidationError, match="Ação inválida ou ausente"):
        validator.validate_action("delete")
    with pytest.raises(ValidationError):
        validator.validate_action(None)
    with pytest.raises(ValidationError):
        validator.validate_action("")

def test_websocket_validator_audio_format():
    """Tests audio format validation, checking for valid formats and raising on invalid ones."""
    validator = WebSocketValidator()
    assert validator.validate_audio_format("wav") is None
    assert validator.validate_audio_format("mp4") is None
    with pytest.raises(ValidationError, match="Formato de arquivo inválido ou não suportado"):
        validator.validate_audio_format("mp3")
    with pytest.raises(ValidationError):
        validator.validate_audio_format("")

def test_websocket_validator_session_state():
    """Tests the validation of session state, raising an exception for invalid states."""
    validator = WebSocketValidator()
    valid_session = {"status": "recording"}
    invalid_session = {"status": "processing"}
    assert validator.validate_session_state_for_chunk(valid_session) is None
    with pytest.raises(ValidationError, match="Gravação não está ativa"):
        validator.validate_session_state_for_chunk(invalid_session)

# --- E2E Quality and Performance Tests ---

import librosa
import time
from tests.metrics import calculate_dual_wer

# Configuration for the E2E pipeline quality tests
PIPELINE_TEST_CONFIG = {
    "d.speakers.wav": {
        "ground_truth_file": "d_speakers.txt",
        "expected_speakers": 2
    },
    "q.speakers.wav": {
        "ground_truth_file": "q_speakers.txt",
        "expected_speakers": 4
    }
}

# Helper functions adapted from the reference test
def load_ground_truth(file_path: Path) -> str:
    """Load and return ground truth text."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def get_audio_duration(audio_path: Path) -> float:
    """Get duration of audio file."""
    return librosa.get_duration(path=str(audio_path))

@pytest.mark.parametrize("audio_name", PIPELINE_TEST_CONFIG.keys())
@pytest.mark.asyncio
async def test_pipeline_quality_and_performance(client, audio_name: str):
    """
    Runs the full transcription and diarization pipeline for a given audio file
    and asserts that the results meet quality and performance standards.
    """
    # --- 1. Setup ---
    print(f"\n--- Running E2E Quality Test for: {audio_name} ---")
    config = PIPELINE_TEST_CONFIG[audio_name]
    audio_path = Path(__file__).parent.parent / "data" / "recordings" / audio_name
    ground_truth_path = Path(__file__).parent / "ground_truth" / config["ground_truth_file"]

    assert audio_path.exists(), f"Audio file not found: {audio_path}"
    assert ground_truth_path.exists(), f"Ground truth file not found: {ground_truth_path}"

    ground_truth_text = load_ground_truth(ground_truth_path)
    audio_duration = get_audio_duration(audio_path)

    # Get initialized services from the global app_state
    transcription_service = app_state.transcription_service
    diarizer = app_state.diarization_service
    assert transcription_service is not None, "Transcription service not initialized"
    assert diarizer is not None, "Diarization service not initialized"

    # --- 2. Execution ---
    start_time = time.time()

    # Step A: Transcription
    transcription_result = await transcription_service.transcribe_with_enhancements(
        str(audio_path), word_timestamps=True
    )

    # Step B: Diarization
    diarization_result = await diarizer.diarize(
        str(audio_path), transcription_result.segments
    )

    total_time = time.time() - start_time

    # --- 3. Metrics Calculation ---
    # Transcription Accuracy
    dual_wer = calculate_dual_wer(ground_truth_text, transcription_result.text)
    transcription_accuracy_normalized = dual_wer['accuracy_normalized_percent']

    # Diarization Accuracy
    speakers_detected = diarization_result['num_speakers']
    expected_speakers = config["expected_speakers"]

    # Performance
    processing_ratio = total_time / audio_duration if audio_duration > 0 else 0

    # --- 4. Validation (Asserts) ---
    print(f"Results for {audio_name}: Accuracy={transcription_accuracy_normalized:.2f}%, Speakers={speakers_detected}, Ratio={processing_ratio:.2f}x")

    assert speakers_detected == expected_speakers, f"Diarization failed for {audio_name}: expected {expected_speakers}, got {speakers_detected}"
    assert transcription_accuracy_normalized >= 85.0, f"Transcription accuracy for {audio_name} is too low: {transcription_accuracy_normalized:.2f}% (target >= 85%)"
    assert processing_ratio < 2.0, f"Processing speed for {audio_name} is too slow: {processing_ratio:.2f}x (target < 2.0x)"
