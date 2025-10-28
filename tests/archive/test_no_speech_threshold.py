
import asyncio
import logging
from pathlib import Path
import sys
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transcription import TranscriptionService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

AUDIO_FILE = Path(__file__).parent.parent / "data" / "recordings" / "q.speakers.wav"

async def run_no_speech_threshold_test():
    """
    Tests different no_speech_threshold configurations to see if it affects the omission of short phrases.
    """
    transcription_service = TranscriptionService(model_name="medium", device="cpu")
    await transcription_service.initialize(compute_type="int8")

    no_speech_thresholds_to_test = [0.6, 0.5, 0.4, 0.3]

    logger.info(f"--- Starting No Speech Threshold Test on {AUDIO_FILE.name} ---")

    for threshold in no_speech_thresholds_to_test:
        logger.info(f"\n--- Testing no_speech_threshold: {threshold} ---")
        start_time = time.time()
        
        whisper_params = {"no_speech_threshold": threshold}
        
        result = await transcription_service.transcribe_with_enhancements(
            str(AUDIO_FILE),
            word_timestamps=True,
            whisper_params=whisper_params
        )
        
        processing_time = time.time() - start_time
        
        transcribed_text = result.text.lower()
        
        logger.info(f"Transcription: \"{transcribed_text}\"")
        
        if "sou solteira" in transcribed_text:
            logger.info(f"✅ SUCCESS: Found 'sou solteira' with no_speech_threshold: {threshold}")
        else:
            logger.info(f"❌ FAILED: Did not find 'sou solteira' with no_speech_threshold: {threshold}")
            
        logger.info(f"Processing time: {processing_time:.2f}s")

    await transcription_service.unload_model()

if __name__ == "__main__":
    asyncio.run(run_no_speech_threshold_test())
