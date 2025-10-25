"""
Manual test for WebSocket live recording endpoint
Tests the /ws/{session_id} endpoint with start, audio_chunk, stop actions

Usage:
    1. Start the server: python main.py
    2. In another terminal: python tests/test_websocket_live_recording.py
"""

import asyncio
import base64
import json
import sys
from pathlib import Path

# Check if websockets is installed
try:
    import websockets
except ImportError:
    print("❌ ERROR: websockets library not installed")
    print("Install with: pip install websockets")
    sys.exit(1)

WS_URL = "ws://localhost:8000/ws/test-session-123"


async def test_websocket_flow():
    """Test complete WebSocket flow: start -> chunk -> stop"""

    print("="*80)
    print("TESTING WEBSOCKET LIVE RECORDING")
    print("="*80)
    print(f"\nConnecting to: {WS_URL}\n")

    try:
        async with websockets.connect(WS_URL) as websocket:
            print("✅ WebSocket connected\n")

            # TEST 1: Send START command
            print("📤 TEST 1: Sending START command (format: wav)")
            start_msg = {
                "action": "start",
                "format": "wav"
            }
            await websocket.send(json.dumps(start_msg))
            response = await websocket.recv()
            response_data = json.loads(response)
            print(f"📥 Response: {response_data}")

            if response_data.get("type") == "recording_started":
                print("✅ PASS - Recording started\n")
            else:
                print(f"❌ FAIL - Expected 'recording_started', got '{response_data.get('type')}'\n")
                return

            # TEST 2: Send mock audio chunk
            print("📤 TEST 2: Sending mock audio chunk (1KB)")
            mock_audio = b"\x00" * 1024  # 1KB of zeros
            chunk_b64 = base64.b64encode(mock_audio).decode('utf-8')

            chunk_msg = {
                "action": "audio_chunk",
                "data": chunk_b64
            }
            await websocket.send(json.dumps(chunk_msg))

            # Audio chunks don't send immediate response unless it's a progress update
            print("✅ PASS - Audio chunk sent (no response expected)\n")

            # Wait a bit
            await asyncio.sleep(0.5)

            # TEST 3: Send STOP command
            print("📤 TEST 3: Sending STOP command")
            stop_msg = {
                "action": "stop"
            }
            await websocket.send(json.dumps(stop_msg))

            # Receive multiple responses (recording_stopped, processing_started)
            responses = []
            try:
                for _ in range(3):  # Receive up to 3 messages
                    response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    response_data = json.loads(response)
                    responses.append(response_data)
                    print(f"📥 Response: {response_data}")
            except asyncio.TimeoutError:
                pass  # No more messages

            # Check if we got recording_stopped
            has_stopped = any(r.get("type") == "recording_stopped" for r in responses)
            has_processing = any(r.get("type") == "processing_started" for r in responses)

            if has_stopped:
                print("✅ PASS - Recording stopped\n")
            else:
                print("❌ FAIL - Did not receive 'recording_stopped'\n")

            if has_processing:
                print("✅ PASS - Processing started\n")
            else:
                print("❌ FAIL - Did not receive 'processing_started'\n")

            # TEST 4: Send invalid action
            print("📤 TEST 4: Sending invalid action")
            invalid_msg = {
                "action": "invalid_action"
            }
            await websocket.send(json.dumps(invalid_msg))
            response = await websocket.recv()
            response_data = json.loads(response)
            print(f"📥 Response: {response_data}")

            if response_data.get("type") == "error":
                print("✅ PASS - Error message received for invalid action\n")
            else:
                print(f"❌ FAIL - Expected error, got '{response_data.get('type')}'\n")

            print("="*80)
            print("✅ ALL WEBSOCKET TESTS COMPLETED")
            print("="*80)
            print("\nNote: To fully test, check server logs for:")
            print("  - LiveAudioProcessor.start_recording() called")
            print("  - LiveAudioProcessor.process_audio_chunk() called")
            print("  - LiveAudioProcessor.stop_recording() called")
            print("  - process_audio_pipeline() triggered\n")

    except websockets.exceptions.WebSocketException as e:
        print(f"\n❌ WebSocket error: {e}")
        print("\nMake sure the server is running:")
        print("  python main.py\n")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


async def test_format_validation():
    """Test format validation (wav vs mp4)"""

    print("\n" + "="*80)
    print("TESTING FORMAT VALIDATION")
    print("="*80)
    print(f"\nConnecting to: {WS_URL}\n")

    try:
        async with websockets.connect(WS_URL) as websocket:
            print("✅ WebSocket connected\n")

            # TEST: Invalid format
            print("📤 TEST: Sending START with invalid format")
            start_msg = {
                "action": "start",
                "format": "invalid"
            }
            await websocket.send(json.dumps(start_msg))
            response = await websocket.recv()
            response_data = json.loads(response)
            print(f"📥 Response: {response_data}")

            if response_data.get("type") == "error" and "format" in response_data.get("message", "").lower():
                print("✅ PASS - Invalid format rejected\n")
            else:
                print("❌ FAIL - Invalid format not properly rejected\n")

            # TEST: Valid MP4 format
            print("📤 TEST: Sending START with mp4 format")
            start_msg = {
                "action": "start",
                "format": "mp4"
            }
            await websocket.send(json.dumps(start_msg))
            response = await websocket.recv()
            response_data = json.loads(response)
            print(f"📥 Response: {response_data}")

            if response_data.get("type") == "recording_started" and response_data.get("format") == "mp4":
                print("✅ PASS - MP4 format accepted\n")
            else:
                print("❌ FAIL - MP4 format not properly handled\n")

            print("="*80)
            print("✅ FORMAT VALIDATION TESTS COMPLETED")
            print("="*80)

    except Exception as e:
        print(f"\n❌ Error: {e}")


async def main():
    """Run all tests"""
    print("\n🚀 Starting WebSocket Live Recording Tests\n")

    await test_websocket_flow()
    await asyncio.sleep(1)
    await test_format_validation()

    print("\n✅ All tests complete!\n")


if __name__ == "__main__":
    asyncio.run(main())
