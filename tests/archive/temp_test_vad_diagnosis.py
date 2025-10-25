
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

async def run_vad_diagnosis():
    """
    Tests different VAD configurations to see if the missing phrase can be recovered.
    """
    transcription_service = TranscriptionService(model_name="medium", device="cpu")
    await transcription_service.initialize(compute_type="int8")

    vad_configs_to_test = [
        {"threshold": 0.5, "min_speech_duration_ms": 250},  # Baseline
        {"threshold": 0.4, "min_speech_duration_ms": 250},
        {"threshold": 0.3, "min_speech_duration_ms": 250},
        {"threshold": 0.2, "min_speech_duration_ms": 150},
        {"threshold": 0.1, "min_speech_duration_ms": 100},
    ]

    logger.info(f"--- Starting VAD Omission Diagnosis on {AUDIO_FILE.name} ---")

    for config in vad_configs_to_test:
        logger.info(f"\n--- Testing VAD config: {config} ---")
        start_time = time.time()
        
        whisper_params = {"vad_parameters": config}
        
        result = await transcription_service.transcribe_with_enhancements(
            str(AUDIO_FILE),
            word_timestamps=True,
            whisper_params=whisper_params
        )
        
        processing_time = time.time() - start_time
        
        transcribed_text = result.text.lower()
        
        logger.info(f"Transcription: \"{transcribed_text}\"")
        
        if "sou solteira" in transcribed_text:
            logger.info(f"✅ SUCCESS: Found 'sou solteira' with config {config}")
        else:
            logger.info(f"❌ FAILED: Did not find 'sou solteira' with config {config}")
            
        logger.info(f"Processing time: {processing_time:.2f}s")

    await transcription_service.unload_model()

if __name__ == "__main__":
    asyncio.run(run_vad_diagnosis())
