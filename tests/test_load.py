
import asyncio
import logging
from pathlib import Path
import sys
import time
import httpx
import websockets
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Test Configuration ---
CONCURRENT_USERS = 5
AUDIO_FILE_PATH = Path(__file__).parent.parent / "data" / "recordings" / "d.speakers.wav"
BASE_URL = "http://127.0.0.1:8000"

async def run_single_upload_and_monitor(client: httpx.AsyncClient, session_id: str):
    """
    Simulates a single user uploading a file and monitoring the progress via WebSocket.
    """
    start_time = time.time()
    
    try:
        # Connect to WebSocket
        async with websockets.connect(f"ws://127.0.0.1:8000/ws/{session_id}", ping_interval=120, ping_timeout=120) as websocket:
            logger.info(f"User {session_id}: WebSocket connected.")

            # Upload the file
            with open(AUDIO_FILE_PATH, "rb") as f:
                files = {"file": (AUDIO_FILE_PATH.name, f, "audio/wav")}
                data = {"session_id": session_id}
                response = await client.post(f"{BASE_URL}/upload", files=files, data=data)
                response.raise_for_status()
            
            logger.info(f"User {session_id}: File uploaded.")

            # Monitor for completion message
            while True:
                try:
                    message_str = await asyncio.wait_for(websocket.recv(), timeout=600.0) # 10 minute timeout
                    message = json.loads(message_str)
                    
                    if message.get("type") == "complete":
                        end_time = time.time()
                        processing_time = end_time - start_time
                        logger.info(f"✅ User {session_id}: Processing complete in {processing_time:.2f}s.")
                        return {"session_id": session_id, "status": "success", "processing_time": processing_time}
                    elif message.get("type") == "error":
                        logger.error(f"❌ User {session_id}: Received error: {message.get('message')}")
                        return {"session_id": session_id, "status": "error", "message": message.get('message')}
                except asyncio.TimeoutError:
                    logger.error(f"❌ User {session_id}: Timed out waiting for 'complete' message.")
                    return {"session_id": session_id, "status": "error", "message": "Timeout"}
    except Exception as e:
        logger.error(f"❌ User {session_id}: An error occurred: {e}", exc_info=True)
        return {"session_id": session_id, "status": "error", "message": str(e)}

async def run_load_test():
    """
    Runs a load test by simulating multiple concurrent users.
    """
    logger.info(f"--- Starting Load Test with {CONCURRENT_USERS} concurrent users ---")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        tasks = []
        for i in range(CONCURRENT_USERS):
            session_id = f"load_test_user_{i+1}"
            tasks.append(run_single_upload_and_monitor(client, session_id))
        
        results = await asyncio.gather(*tasks)
    
    logger.info("\n--- Load Test Results ---")
    
    success_count = 0
    total_processing_time = 0
    for result in results:
        if result["status"] == "success":
            success_count += 1
            total_processing_time += result["processing_time"]
    
    if success_count > 0:
        average_processing_time = total_processing_time / success_count
        logger.info(f"  Successful uploads: {success_count}/{CONCURRENT_USERS}")
        logger.info(f"  Average processing time: {average_processing_time:.2f}s")
    else:
        logger.info("  No successful uploads.")

    # Save results to a JSON file
    results_summary = {
        "concurrent_users": CONCURRENT_USERS,
        "successful_uploads": success_count,
        "average_processing_time": average_processing_time if success_count > 0 else 0,
        "results": results
    }
    
    logs_dir = Path(__file__).parent / "logs" / "load_test"
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_filename = logs_dir / f"load_test_{timestamp}.json"
    
    with open(log_filename, "w") as f:
        json.dump(results_summary, f, indent=2)
        
    logger.info(f"Load test results saved to: {log_filename}")

if __name__ == "__main__":
    # This test needs the FastAPI server to be running separately.
    # You can run it with: uvicorn main:app
    logger.info("Starting load test... Make sure the FastAPI server is running.")
    asyncio.run(run_load_test())
