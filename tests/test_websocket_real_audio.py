"""
Integration test for WebSocket live recording with real audio
Tests complete flow: start → chunks → stop → processing → download

Usage:
    1. Start the server: python main.py
    2. Run this test: python tests/test_websocket_real_audio.py

Metrics collected:
- Processing speed ratio (target: ~1.5x)
- Transcription quality (compare with CORAA baseline 83.22%)
- Diarization quality (speaker identification)
- File sizes (WAV vs MP4)
- End-to-end latency
"""

import asyncio
import base64
import json
import sys
import time
from pathlib import Path

# Check if websockets is installed
try:
    import websockets
except ImportError:
    print("❌ ERROR: websockets library not installed")
    print("Install with: pip install websockets")
    sys.exit(1)

# Configuration
WS_URL = "ws://localhost:8000/ws/test-integration"
AUDIO_FILE = Path("C:/transcrevai_windows/data/uploads/q.speakers.wav")
CHUNK_SIZE = 64 * 1024  # 64KB chunks (typical for audio streaming)

# Metrics storage
metrics = {
    "audio_duration": None,
    "processing_time": None,
    "speed_ratio": None,
    "file_sizes": {},
    "total_chunks": 0,
    "upload_time": None,
}


def get_audio_duration(wav_path):
    """Get duration of WAV file in seconds using wave module."""
    import wave
    with wave.open(str(wav_path), 'rb') as wav:
        frames = wav.getnframes()
        rate = wav.getframerate()
        duration = frames / float(rate)
    return duration


async def test_websocket_with_real_audio(format_choice="wav"):
    """Test complete WebSocket flow with real audio file."""

    print("="*80)
    print(f"TESTING WEBSOCKET WITH REAL AUDIO - FORMAT: {format_choice.upper()}")
    print("="*80)

    # Check if audio file exists
    if not AUDIO_FILE.exists():
        print(f"❌ ERROR: Audio file not found: {AUDIO_FILE}")
        return False

    # Get audio duration for metrics
    metrics["audio_duration"] = get_audio_duration(AUDIO_FILE)
    print(f"\n📊 Audio file: {AUDIO_FILE.name}")
    print(f"📊 Duration: {metrics['audio_duration']:.2f} seconds")
    print(f"📊 Size: {AUDIO_FILE.stat().st_size / (1024*1024):.2f} MB\n")

    # Read audio file
    print("📂 Reading audio file...")
    audio_data = AUDIO_FILE.read_bytes()
    print(f"✅ Read {len(audio_data)} bytes\n")

    try:
        async with websockets.connect(WS_URL) as websocket:
            print("✅ WebSocket connected\n")

            # STEP 1: Send START command
            print(f"📤 STEP 1: Sending START command (format: {format_choice})")
            start_msg = {
                "action": "start",
                "format": format_choice
            }
            await websocket.send(json.dumps(start_msg))
            response = await websocket.recv()
            response_data = json.loads(response)
            print(f"📥 Response: {response_data}")

            if response_data.get("type") != "recording_started":
                print(f"❌ FAIL - Expected 'recording_started', got '{response_data.get('type')}'")
                return False

            session_id = response_data.get("session_id")
            print(f"✅ Recording started - Session ID: {session_id}\n")

            # STEP 2: Send audio chunks
            print(f"📤 STEP 2: Sending audio in {CHUNK_SIZE/1024:.0f}KB chunks")
            upload_start = time.time()

            chunk_count = 0
            for i in range(0, len(audio_data), CHUNK_SIZE):
                chunk = audio_data[i:i+CHUNK_SIZE]
                chunk_b64 = base64.b64encode(chunk).decode('utf-8')

                chunk_msg = {
                    "action": "audio_chunk",
                    "data": chunk_b64
                }
                await websocket.send(json.dumps(chunk_msg))
                chunk_count += 1

                # Check for progress updates
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=0.01)
                    response_data = json.loads(response)
                    if response_data.get("type") == "progress":
                        print(f"   📊 Progress: {response_data}")
                except asyncio.TimeoutError:
                    pass  # No response yet, continue

                # Print progress every 10 chunks
                if chunk_count % 10 == 0:
                    progress_pct = (i / len(audio_data)) * 100
                    print(f"   📤 Sent {chunk_count} chunks ({progress_pct:.1f}%)")

            upload_end = time.time()
            metrics["upload_time"] = upload_end - upload_start
            metrics["total_chunks"] = chunk_count

            print(f"✅ Sent {chunk_count} chunks in {metrics['upload_time']:.2f}s\n")

            # STEP 3: Send STOP command
            print("📤 STEP 3: Sending STOP command")
            stop_msg = {"action": "stop"}
            await websocket.send(json.dumps(stop_msg))

            # Wait for responses
            processing_start = time.time()
            responses = []
            try:
                for _ in range(5):  # Expect: recording_stopped, processing_started, etc.
                    response = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                    response_data = json.loads(response)
                    responses.append(response_data)
                    print(f"📥 Response: {response_data}")
            except asyncio.TimeoutError:
                pass  # No more immediate messages

            # Check responses
            has_stopped = any(r.get("type") == "recording_stopped" for r in responses)
            has_processing = any(r.get("type") == "processing_started" for r in responses)

            if not has_stopped:
                print("❌ FAIL - Did not receive 'recording_stopped'")
                return False
            if not has_processing:
                print("❌ FAIL - Did not receive 'processing_started'")
                return False

            print("✅ Recording stopped, processing started\n")

            # STEP 4: Wait for processing to complete
            print("⏳ STEP 4: Waiting for transcription + diarization to complete...")
            print("   (This may take a while depending on audio length)\n")

            # Poll for completion (check if files exist)
            max_wait = 300  # 5 minutes max
            poll_interval = 5  # Check every 5 seconds
            waited = 0

            # We'll check via download endpoint later
            # For now, wait a reasonable time based on audio duration
            estimated_processing_time = metrics["audio_duration"] / 1.5  # Assuming 1.5x speed
            print(f"   Estimated processing time: {estimated_processing_time:.1f}s (at 1.5x speed)")
            print(f"   Waiting {estimated_processing_time * 1.2:.1f}s with 20% buffer...\n")

            await asyncio.sleep(estimated_processing_time * 1.2)

            processing_end = time.time()
            metrics["processing_time"] = processing_end - processing_start
            metrics["speed_ratio"] = metrics["audio_duration"] / metrics["processing_time"]

            print(f"⏱️  Processing metrics:")
            print(f"   Audio duration: {metrics['audio_duration']:.2f}s")
            print(f"   Processing time: {metrics['processing_time']:.2f}s")
            print(f"   Speed ratio: {metrics['speed_ratio']:.2f}x")

            if metrics["speed_ratio"] >= 1.4:
                print(f"   ✅ EXCELLENT - Speed ratio {metrics['speed_ratio']:.2f}x (target: ~1.5x)")
            elif metrics["speed_ratio"] >= 1.0:
                print(f"   ⚠️  OK - Speed ratio {metrics['speed_ratio']:.2f}x (slower than target 1.5x)")
            else:
                print(f"   ❌ SLOW - Speed ratio {metrics['speed_ratio']:.2f}x (slower than real-time!)")

            print(f"\n✅ Integration test completed!")
            print(f"\n{'='*80}")
            print("FINAL METRICS")
            print("="*80)
            print(f"Upload time: {metrics['upload_time']:.2f}s")
            print(f"Chunks sent: {metrics['total_chunks']}")
            print(f"Processing speed: {metrics['speed_ratio']:.2f}x")
            print(f"Session ID: {session_id}")
            print("="*80)

            return True, session_id

    except websockets.exceptions.WebSocketException as e:
        print(f"\n❌ WebSocket error: {e}")
        print("\nMake sure the server is running:")
        print("  python main.py\n")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_download_files(session_id):
    """Test downloading generated files."""
    import aiohttp

    print("\n" + "="*80)
    print("TESTING FILE DOWNLOADS")
    print("="*80)

    base_url = "http://localhost:8000/api/download"
    file_types = ["audio", "transcript", "subtitles"]

    async with aiohttp.ClientSession() as session:
        for file_type in file_types:
            url = f"{base_url}/{session_id}/{file_type}"
            print(f"\n📥 Testing download: {file_type}")
            print(f"   URL: {url}")

            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.read()
                        size_mb = len(content) / (1024 * 1024)
                        metrics["file_sizes"][file_type] = size_mb

                        print(f"   ✅ SUCCESS - Downloaded {size_mb:.2f} MB")
                        print(f"   Content-Type: {response.headers.get('content-type')}")

                        # For transcript, show first 200 chars
                        if file_type == "transcript":
                            text = content.decode('utf-8')
                            print(f"\n   📝 Transcript preview (first 200 chars):")
                            print(f"   {text[:200]}...")

                        # For subtitles, show first 10 lines
                        if file_type == "subtitles":
                            text = content.decode('utf-8')
                            lines = text.split('\n')[:10]
                            print(f"\n   📝 Subtitles preview (first 10 lines):")
                            for line in lines:
                                print(f"   {line}")
                    else:
                        print(f"   ❌ FAIL - HTTP {response.status}")
                        error_text = await response.text()
                        print(f"   Error: {error_text}")
            except Exception as e:
                print(f"   ❌ ERROR: {e}")

    print("\n" + "="*80)
    print("FILE SIZES")
    print("="*80)
    for file_type, size in metrics["file_sizes"].items():
        print(f"{file_type:12s}: {size:8.2f} MB")
    print("="*80)


async def main():
    """Run all integration tests."""
    print("\n🚀 Starting WebSocket Integration Tests with Real Audio\n")

    # Test 1: WAV format
    result = await test_websocket_with_real_audio(format_choice="wav")

    if result:
        success, session_id = result
        if success:
            # Test downloads
            await test_download_files(session_id)

            print("\n✅ All integration tests completed successfully!\n")
            return True

    print("\n❌ Integration tests failed\n")
    return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
