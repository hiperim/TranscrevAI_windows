
import asyncio
import logging
from pathlib import Path
import sys
import time
import librosa

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transcription import TranscriptionService
from src.diarization import PyannoteDiarizer
from tests.metrics import calculate_dual_wer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

AUDIO_FILE_TO_TEST = {
    "name": "d.speakers.mp4",
    "expected_speakers": 2,
    "ground_truth_path": Path(__file__).parent / "ground_truth" / "d_speakers.txt"
}

async def run_mp4_test():
    """
    Tests the application's ability to handle MP4 files.
    """
    transcription_service = TranscriptionService(model_name="medium", device="cpu")
    await transcription_service.initialize(compute_type="int8")
    diarizer = PyannoteDiarizer()

    file_info = AUDIO_FILE_TO_TEST
    file_name = file_info["name"]
    expected_speakers = file_info["expected_speakers"]
    ground_truth_path = file_info["ground_truth_path"]
    
    audio_path = Path(__file__).parent.parent / "data" / "recordings" / file_name

    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        ground_truth = f.read().strip()

    logger.info(f"\n--- Testing MP4 Format: {file_name} ---")

    # Run transcription and measure time
    transcription_start_time = time.time()
    result = await transcription_service.transcribe_with_enhancements(str(audio_path), word_timestamps=True)
    transcription_time = time.time() - transcription_start_time
    
    transcribed_text = result.text

    # Run diarization and measure time
    diarization_start_time = time.time()
    diarization_result = await diarizer.diarize(str(audio_path), result.segments)
    diarization_time = time.time() - diarization_start_time

    # Calculate metrics
    metrics = calculate_dual_wer(ground_truth, transcribed_text)
    diarization_accuracy = 100.0 if diarization_result['num_speakers'] == expected_speakers else 0.0
    audio_duration = librosa.get_duration(path=str(audio_path))
    processing_ratio = (transcription_time + diarization_time) / audio_duration

    logger.info(f"Transcription: \"{transcribed_text}\" ")
    
    logger.info(f"\n--- Results for {file_name} ---")
    logger.info(f"  Transcription Accuracy (Normalized): {metrics['accuracy_normalized_percent']:.2f}%")
    logger.info(f"  Diarization Accuracy: {diarization_accuracy:.2f}% ({diarization_result['num_speakers']}/{expected_speakers} speakers)")
    logger.info(f"  Processing Speed: {processing_ratio:.2f}x")

    await transcription_service.unload_model()

if __name__ == "__main__":
    asyncio.run(run_mp4_test())
