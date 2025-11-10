# tests/test_integration.py

import pytest
import asyncio
import base64
from pathlib import Path
from starlette.testclient import TestClient

# Add root directory to path to allow src imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import app
from src.transcription import TranscriptionResult

@pytest.mark.timeout(30)
def test_websocket_full_flow(monkeypatch):
    """
    Tests the full WebSocket flow for live recording.
    Uses synchronous TestClient to avoid event loop conflicts.
    """
    # --- Mock Operations ---

    # 1. Mock ffmpeg conversion
    def mock_convert_webm_to_wav(self, input_path, output_path, sample_rate=16000):
        import wave
        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(b'')

    monkeypatch.setattr(
        "src.audio_processing.LiveAudioProcessor._convert_webm_to_wav",
        mock_convert_webm_to_wav
    )

    # 2. Mock TranscriptionService - NO NEED TO MOCK
    # The real service will work fine in tests, just fast with small files
    # Let it run normally to avoid event loop conflicts

    session_id = "integration_test_123"

    # --- Test with synchronous TestClient ---
    with TestClient(app) as client:
        with client.websocket_connect(f"/ws/{session_id}") as websocket:

            print("\n--- WebSocket Integration Test ---")

            # 1. Send START
            print("📤 Sending START...")
            websocket.send_json({"action": "start", "format": "wav"})
            response = websocket.receive_json()
            print(f"📥 {response}")
            assert response["type"] == "recording_started"
            print("✅ START successful")

            # 2. Send audio chunks
            print("📤 Sending 12 audio chunks...")
            chunk_b64 = base64.b64encode(b"\x00" * 16000).decode('utf-8')

            for i in range(12):
                websocket.send_json({"action": "audio_chunk", "data": chunk_b64})

            # Wait for transcription from worker (with timeout)
            print("⏳ Waiting for transcription from worker...")
            import time
            start_time = time.time()
            transcription_received = False

            while time.time() - start_time < 10:
                try:
                    response = websocket.receive_json(timeout=1)
                    print(f"📥 {response}")
                    if response.get("type") == "transcription_chunk":
                        transcription_received = True
                        print(f"✅ Transcription: {response['data']['text']}")
                        break
                except:
                    continue

            assert transcription_received, "No transcription received from worker"

            # 3. Send STOP
            print("📤 Sending STOP...")
            websocket.send_json({"action": "stop"})

            # Collect remaining messages
            final_batch_received = False
            for _ in range(5):
                try:
                    response = websocket.receive_json(timeout=2)
                    print(f"📥 {response}")
                    if response.get("type") == "transcription_chunk":
                        if response.get("data", {}).get("is_final_batch"):
                            final_batch_received = True
                            break
                except:
                    break

            assert final_batch_received, "Final batch not received"
            print("✅ Test PASSED")
