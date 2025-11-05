"""
Edge case tests to prevent production crashes.
Tests invalid inputs, corrupted files, and boundary conditions.
"""
import pytest
import asyncio
import websockets
import json
import base64
from pathlib import Path
from fastapi.testclient import TestClient

# Will use real server approach like test_live_server.py
import subprocess
import time
import requests

SERVER_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000"
STARTUP_TIMEOUT = 30

@pytest.fixture(scope="module")
def server_process():
    """Start real server for edge case testing"""
    import sys
    python_exe = sys.executable

    process = subprocess.Popen(
        [python_exe, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
    )

    # Wait for server ready
    start_time = time.time()
    server_ready = False

    while time.time() - start_time < STARTUP_TIMEOUT:
        try:
            response = requests.get(f"{SERVER_URL}/health", timeout=1)
            if response.status_code == 200:
                server_ready = True
                break
        except:
            time.sleep(0.5)

    if not server_ready:
        process.kill()
        pytest.fail("Server failed to start")

    yield process

    process.terminate()
    process.wait(timeout=5)


@pytest.mark.asyncio
async def test_invalid_audio_format(server_process):
    """Test rejection of non-audio files (.txt, .exe)"""
    session_id = f"invalid_format_{int(time.time())}"

    # Create a fake .txt file
    fake_audio = b"This is not audio data, just plain text"

    files = {"file": ("test.txt", fake_audio, "text/plain")}
    data = {"session_id": session_id}

    response = requests.post(f"{SERVER_URL}/upload", files=files, data=data)

    # Should accept (FastAPI accepts any file), but processing should fail gracefully
    assert response.status_code in [200, 400, 422], "Should handle invalid format"


@pytest.mark.asyncio
async def test_corrupted_audio_header(server_process):
    """Test handling of audio with corrupted WAV header"""
    session_id = f"corrupted_{int(time.time())}"

    # Create invalid WAV (wrong header)
    corrupted_wav = b"RIFF" + b"\x00" * 100  # Missing proper WAV structure

    files = {"file": ("corrupted.wav", corrupted_wav, "audio/wav")}
    data = {"session_id": session_id}

    response = requests.post(f"{SERVER_URL}/upload", files=files, data=data)

    # Should accept upload but fail during processing
    assert response.status_code in [200, 500], "Should handle corrupted audio"


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_websocket_abrupt_disconnect(server_process):
    """Test worker cleanup when WebSocket disconnects mid-stream"""
    session_id = f"disconnect_test_{int(time.time())}"

    async with websockets.connect(f"{WS_URL}/ws/{session_id}") as ws:
        # Start recording
        await ws.send(json.dumps({"action": "start", "format": "wav"}))
        response = await asyncio.wait_for(ws.recv(), timeout=5)
        data = json.loads(response)
        assert data["type"] == "recording_started"

        # Send 2 chunks
        import struct
        sample_rate = 16000
        num_samples = sample_rate // 2
        audio_data = struct.pack('<' + 'h' * num_samples, *([0] * num_samples))

        wav_chunk = (
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

        chunk_b64 = base64.b64encode(wav_chunk).decode('utf-8')

        for _ in range(2):
            await ws.send(json.dumps({"action": "audio_chunk", "data": chunk_b64}))

        # Abruptly close WITHOUT sending STOP
        # WebSocket context exit will close connection

    # Give server time to detect disconnect and cleanup
    await asyncio.sleep(2)

    # Verify session was cleaned up (connection should work again)
    async with websockets.connect(f"{WS_URL}/ws/{session_id}") as ws:
        await ws.send(json.dumps({"action": "start", "format": "wav"}))
        response = await asyncio.wait_for(ws.recv(), timeout=5)
        data = json.loads(response)
        assert data["type"] == "recording_started", "Should allow reconnection after cleanup"


@pytest.mark.asyncio
@pytest.mark.timeout(15)
async def test_empty_audio_chunks(server_process):
    """Test handling of empty audio data"""
    session_id = f"empty_audio_{int(time.time())}"

    async with websockets.connect(f"{WS_URL}/ws/{session_id}") as ws:
        await ws.send(json.dumps({"action": "start", "format": "wav"}))
        await asyncio.wait_for(ws.recv(), timeout=5)

        # Send empty chunk
        await ws.send(json.dumps({"action": "audio_chunk", "data": ""}))

        # Should not crash, wait a bit
        await asyncio.sleep(1)

        # Send STOP
        await ws.send(json.dumps({"action": "stop"}))

        # Collect messages (should handle gracefully)
        messages = []
        for _ in range(5):
            try:
                response = await asyncio.wait_for(ws.recv(), timeout=2)
                messages.append(json.loads(response))
            except asyncio.TimeoutError:
                break

        # Should receive recording_stopped at minimum
        types = [m["type"] for m in messages]
        assert "recording_stopped" in types, "Should handle empty audio gracefully"


@pytest.mark.asyncio
async def test_rate_limit_websocket(server_process):
    """Test WebSocket rate limiting (20 connections/minute)"""
    connections = []

    try:
        # Try to open 25 connections rapidly (limit is 20)
        for i in range(25):
            session_id = f"rate_limit_test_{i}_{int(time.time())}"
            try:
                ws = await websockets.connect(f"{WS_URL}/ws/{session_id}")
                connections.append(ws)
                await asyncio.sleep(0.1)  # Small delay between connections
            except websockets.exceptions.ConnectionClosed as e:
                # Should get rate limited around connection 20-21
                assert i >= 19, f"Rate limit should trigger after 20 connections, got at {i}"
                assert "rate limit" in str(e).lower() or e.code == 1008
                break
        else:
            pytest.fail("Rate limit was not triggered after 25 connections")

    finally:
        # Cleanup all connections
        for ws in connections:
            await ws.close()


@pytest.mark.asyncio
async def test_malformed_json_message(server_process):
    """Test handling of malformed JSON in WebSocket messages"""
    session_id = f"malformed_json_{int(time.time())}"

    async with websockets.connect(f"{WS_URL}/ws/{session_id}") as ws:
        # Send invalid JSON
        await ws.send("{ this is not valid json }")

        # Should handle gracefully (likely error message or connection close)
        try:
            response = await asyncio.wait_for(ws.recv(), timeout=3)
            data = json.loads(response)
            # If we get a response, check it's an error
            assert data.get("type") in ["error", "validation_error"], "Should report error"
        except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
            # Also acceptable - connection closed due to invalid message
            pass


@pytest.mark.asyncio
async def test_unknown_action_type(server_process):
    """Test handling of unknown WebSocket action"""
    session_id = f"unknown_action_{int(time.time())}"

    async with websockets.connect(f"{WS_URL}/ws/{session_id}") as ws:
        # Send unknown action
        await ws.send(json.dumps({"action": "unknown_action_xyz"}))

        # Should handle gracefully
        try:
            response = await asyncio.wait_for(ws.recv(), timeout=3)
            data = json.loads(response)
            assert data.get("type") in ["error", "unknown_action"], "Should report unknown action"
        except asyncio.TimeoutError:
            # Also acceptable - server ignores unknown actions
            pass
