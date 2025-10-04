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
    "d.speakers.wav": {"expected_speakers": 2, "keywords": ["transcrição", "áudio", "precisa"]},
    "q.speakers.wav": {"expected_speakers": 4, "keywords": ["teste", "sistema", "fala"]},
    "t.speakers.wav": {"expected_speakers": 3, "keywords": ["gravação", "qualidade", "boa"]},
    "t2.speakers.wav": {"expected_speakers": 3, "keywords": ["inteligente", "silicone", "luvas"]}
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

@pytest.mark.parametrize("audio_file_path", get_benchmark_files())
@pytest.mark.asyncio
async def test_full_pipeline_with_accuracy_validation(audio_file_path: str):
    """
    This is the main integration test. It runs each benchmark audio file through the full 
    EnhancedTranscriptionPipeline and validates the output against ground truth for accuracy.
    """
    assert getattr(app_state, "pipeline", None) is not None, "Pipeline must be initialized before testing"
    
    file_name = Path(audio_file_path).name
    expectations = BENCHMARK_EXPECTATIONS[file_name]
    test_session_id = f"test_{Path(audio_file_path).stem}"
    
    progress_updates = []
    async def mock_progress_callback(progress, message):
        progress_updates.append({"progress": progress, "message": message})

    # Execute the full pipeline
    pipeline = getattr(app_state, "pipeline", None)
    assert pipeline is not None, "Pipeline must be initialized before testing"
    result = await pipeline.process_audio_file(
        audio_path=audio_file_path,
        session_id=test_session_id,
        progress_callback=mock_progress_callback
    )

    # --- Validate the Final Result ---
    assert result is not None, f"Pipeline returned None for {file_name}"
    
    # 1. Validate Diarization Accuracy
    detected_speakers = result.get("num_speakers", 0)
    expected_speakers = expectations["expected_speakers"]
    assert detected_speakers == expected_speakers, f"Speaker count mismatch for {file_name}. Expected {expected_speakers}, got {detected_speakers}."

    # 2. Validate Transcription Accuracy
    transcribed_text = result.get("text", "").lower()
    assert len(transcribed_text) > 10, f"Transcription text is too short for {file_name}"
    
    missing_keywords = [kw for kw in expectations["keywords"] if kw.lower() not in transcribed_text]
    assert not missing_keywords, f"Transcription for {file_name} is missing keywords: {', '.join(missing_keywords)}"

    # 3. Validate SRT Generation
    assert "srt_file_path" in result and result["srt_file_path"], f"SRT file path missing for {file_name}"
    srt_path = Path(result["srt_file_path"])
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
