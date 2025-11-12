"""
Integration test that runs the REAL server to validate the complete flow.
This tests: WebSocket → Handler → Worker → Transcription → Response
"""
import pytest
import asyncio
import base64
import subprocess
import time
import websockets
import json
import requests
from pathlib import Path

SERVER_URL = "ws://localhost:8000"
STARTUP_TIMEOUT = 60  # seconds - increased for DI initialization

@pytest.fixture(scope="module")
def server_process():
    """Start the real uvicorn server for testing"""
    import sys
    python_exe = sys.executable

    # Start server process
    process = subprocess.Popen(
        [python_exe, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    # Wait for server to be ready
    start_time = time.time()
    server_ready = False

    while time.time() - start_time < STARTUP_TIMEOUT:
        try:
            import requests
            response = requests.get("http://localhost:8000/health", timeout=1)
            if response.status_code == 200:
                server_ready = True
                break
        except Exception:
            time.sleep(0.5)

    if not server_ready:
        process.kill()
        pytest.fail("Server failed to start within timeout")

    yield process

    # Cleanup
    process.terminate()
    process.wait(timeout=5)


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_complete_websocket_flow(server_process):
    """
    Complete integration test with REAL server:
    1. Connect WebSocket
    2. Start recording
    3. Send audio chunks
    4. Verify worker processes chunks
    5. Receive transcription
    6. Stop and verify final transcription
    """
    session_id = "live_test_" + str(int(time.time()))

    print(f"\n{'='*60}")
    print(f"LIVE SERVER INTEGRATION TEST")
    print(f"Session ID: {session_id}")
    print(f"{'='*60}\n")

    async with websockets.connect(f"{SERVER_URL}/ws/{session_id}") as ws:

        # 1. START recording
        print("📤 Step 1: Sending START command...")
        await ws.send(json.dumps({"action": "start", "format": "wav"}))

        response = await asyncio.wait_for(ws.recv(), timeout=5)
        data = json.loads(response)
        print(f"📥 Response: {data}")
        assert data["type"] == "recording_started", f"Expected recording_started, got {data}"
        print("✅ Recording started successfully\n")

        # 2. Send audio chunks (simulate 6 seconds of audio)
        print("📤 Step 2: Sending 12 audio chunks (6 seconds)...")

        # Create a minimal valid WAV file (16kHz, mono, 0.5 second of silence)
        # This tests the complete flow without requiring WebM→WAV conversion
        import struct
        sample_rate = 16000
        num_samples = sample_rate // 2  # 0.5 second per chunk
        audio_data = struct.pack('<' + 'h' * num_samples, *([0] * num_samples))

        # WAV header
        wav_chunk = (
            b'RIFF' +
            struct.pack('<I', 36 + len(audio_data)) +  # ChunkSize
            b'WAVE' +
            b'fmt ' +
            struct.pack('<I', 16) +  # Subchunk1Size
            struct.pack('<H', 1) +   # AudioFormat (PCM)
            struct.pack('<H', 1) +   # NumChannels (mono)
            struct.pack('<I', sample_rate) +  # SampleRate
            struct.pack('<I', sample_rate * 2) +  # ByteRate
            struct.pack('<H', 2) +   # BlockAlign
            struct.pack('<H', 16) +  # BitsPerSample
            b'data' +
            struct.pack('<I', len(audio_data)) +  # Subchunk2Size
            audio_data
        )

        chunk_b64 = base64.b64encode(wav_chunk).decode('utf-8')

        for i in range(12):
            await ws.send(json.dumps({"action": "audio_chunk", "data": chunk_b64}))
            if i % 3 == 0:
                print(f"   Sent chunk {i+1}/12")
            await asyncio.sleep(0.1)  # Simulate realistic timing

        print("✅ All chunks sent\n")

        # 3. Wait for worker to process and send transcription
        print("⏳ Step 3: Waiting for worker to process chunks...")
        print("   (Worker should batch ~5 seconds, then transcribe)")

        transcription_received = False
        start_wait = time.time()

        while time.time() - start_wait < 30:  # 30 second timeout
            try:
                response = await asyncio.wait_for(ws.recv(), timeout=2)
                data = json.loads(response)
                print(f"📥 Worker response: {data}")

                if data.get("type") == "transcription_chunk":
                    transcription_received = True
                    text = data.get("data", {}).get("text", "")
                    is_final = data.get("data", {}).get("is_final_batch", False)
                    print(f"✅ Transcription received: '{text}'")
                    print(f"   Is final batch: {is_final}\n")
                    break

            except asyncio.TimeoutError:
                elapsed = time.time() - start_wait
                print(f"   Still waiting... ({elapsed:.1f}s elapsed)")
                continue

        assert transcription_received, "❌ FAILED: Worker did not send transcription within 30 seconds"

        # 4. Send STOP command
        print("📤 Step 4: Sending STOP command...")
        await ws.send(json.dumps({"action": "stop"}))

        # 5. Collect remaining messages (recording_stopped, processing, final transcription)
        print("⏳ Step 5: Collecting final messages...\n")
        final_batch_received = False
        messages_received = []

        for attempt in range(10):
            try:
                response = await asyncio.wait_for(ws.recv(), timeout=3)
                data = json.loads(response)
                messages_received.append(data)
                print(f"📥 Message {len(messages_received)}: {data.get('type')}")

                if data.get("type") == "transcription_chunk":
                    if data.get("data", {}).get("is_final_batch"):
                        final_batch_received = True
                        print(f"✅ Final transcription: '{data.get('data', {}).get('text')}'")
                        break

            except asyncio.TimeoutError:
                print(f"   No more messages (timeout on attempt {attempt+1})")
                break

        # Validation
        print(f"\n{'='*60}")
        print("VALIDATION RESULTS:")
        print(f"{'='*60}")
        print(f"✅ WebSocket connected: YES")
        print(f"✅ Recording started: YES")
        print(f"✅ Chunks sent: 12/12")
        print(f"✅ Worker processed chunks: {transcription_received}")
        print(f"✅ Transcription received: {transcription_received}")
        print(f"✅ Final batch received: {final_batch_received}")
        print(f"📊 Total messages received: {len(messages_received)}")
        print(f"{'='*60}\n")

        assert transcription_received, "Worker did not process chunks"
        assert final_batch_received, "Did not receive final transcription batch"

        print("🎉 ALL TESTS PASSED - Live recording flow is working!\n")


def test_upload_endpoint(server_process):
    """Test the /upload endpoint with real server"""
    # Create minimal test WAV file
    import struct
    sample_rate = 16000
    duration = 1  # 1 second
    num_samples = sample_rate * duration
    audio_data = struct.pack('<' + 'h' * num_samples, *([0] * num_samples))

    wav_data = (
        b'RIFF' +
        struct.pack('<I', 36 + len(audio_data)) +
        b'WAVE' +
        b'fmt ' +
        struct.pack('<I', 16) +
        struct.pack('<H', 1) +
        struct.pack('<H', 1) +
        struct.pack('<I', sample_rate) +
        struct.pack('<I', sample_rate * 2) +
        struct.pack('<H', 2) +
        struct.pack('<H', 16) +
        b'data' +
        struct.pack('<I', len(audio_data)) +
        audio_data
    )

    session_id = f"test_upload_{int(time.time())}"
    files = {"file": ("test.wav", wav_data, "audio/wav")}
    data = {"session_id": session_id}

    response = requests.post("http://localhost:8000/upload", files=files, data=data)

    assert response.status_code == 200
    json_response = response.json()
    assert json_response["success"] is True
    assert json_response["session_id"] == session_id

    print(f"✅ Upload endpoint test passed for session {session_id}")
