
import asyncio

import logging

from pathlib import Path

import sys

import time

import httpx

import json

import websockets



# Add parent directory to path for imports

sys.path.insert(0, str(Path(__file__).parent.parent))



# Setup logging

logging.basicConfig(

    level=logging.INFO,

    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'

)

logger = logging.getLogger(__name__)



# --- Test Configuration ---

AUDIO_FILE_PATH = Path(__file__).parent.parent / "data" / "recordings" / "d.speakers.wav"

BASE_URL = "https://localhost:8000" # Use HTTPS



async def run_benchmark():

    """

    Runs a single-file benchmark and measures the processing time.

    """

    logger.info(f"--- Starting Single-File Benchmark ---")

    

    start_time = time.time()

    

    # Create a session

    session_id = f"benchmark_{int(time.time())}"

    

    import ssl

    

    

    

    # Create SSL context for self-signed certificate

    

    ssl_context = ssl.create_default_context()

    

    ssl_context.check_hostname = False

    

    ssl_context.verify_mode = ssl.CERT_NONE

    

    

    

    # Connect to WebSocket

    

    async with websockets.connect(f"wss://localhost:8000/ws/{session_id}", ssl=ssl_context) as websocket:

        logger.info(f"WebSocket connected.")



        # Upload the file

        async with httpx.AsyncClient(timeout=600.0, verify=False) as client:

            with open(AUDIO_FILE_PATH, "rb") as f:

                files = {"file": (AUDIO_FILE_PATH.name, f, "audio/wav")}

                data = {"session_id": session_id}

                response = await client.post(f"{BASE_URL}/upload", files=files, data=data)

                response.raise_for_status()

            

        logger.info(f"File uploaded.")



        # Monitor for completion message

        while True:

            try:

                message_str = await asyncio.wait_for(websocket.recv(), timeout=600.0) # 10 minute timeout

                message = json.loads(message_str)

                

                if message.get("type") == "complete":

                    end_time = time.time()

                    processing_time = end_time - start_time

                    logger.info(f"✅ Processing complete in {processing_time:.2f}s.")

                    logger.info(f"Processing ratio: {message['result']['processing_ratio']:.2f}x")

                    break

                elif message.get("type") == "error":

                    logger.error(f"❌ Received error: {message.get('message')}")

                    break

            except asyncio.TimeoutError:

                logger.error(f"❌ Timed out waiting for 'complete' message.")

                break



if __name__ == "__main__":

    logger.info("Starting benchmark... Make sure the FastAPI server is running.")

    asyncio.run(run_benchmark())


