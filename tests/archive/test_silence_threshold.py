
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

AUDIO_FILES_TO_TEST = [
    {
        "name": "d.speakers.wav",
        "expected_speakers": 2,
        "ground_truth_path": Path(__file__).parent / "ground_truth" / "d_speakers.txt"
    },
    {
        "name": "q.speakers.wav",
        "expected_speakers": 4,
        "ground_truth_path": Path(__file__).parent / "ground_truth" / "q_speakers.txt"
    }
]

async def run_silence_threshold_test():
    """
    Tests different silence_threshold configurations and their impact on transcription, diarization, and speed.
    """
    transcription_service = TranscriptionService(model_name="medium", device="cpu")
    await transcription_service.initialize(compute_type="int8")
    diarizer = PyannoteDiarizer()

    silence_thresholds_to_test = [None, 0.1, 0.2, 0.3]

    logger.info(f"--- Starting Silence Threshold Benchmark ---")

    for threshold in silence_thresholds_to_test:
        logger.info(f"\n{'='*80}")
        logger.info(f"--- Testing silence_threshold: {threshold} ---")
        logger.info(f"{'='*80}")

        for file_info in AUDIO_FILES_TO_TEST:
            file_name = file_info["name"]
            expected_speakers = file_info["expected_speakers"]
            ground_truth_path = file_info["ground_truth_path"]
            
            audio_path = Path(__file__).parent.parent / "data" / "recordings" / file_name

            with open(ground_truth_path, 'r', encoding='utf-8') as f:
                ground_truth = f.read().strip()

            logger.info(f"\n--- Testing {file_name} ---")

            # Run transcription and measure time
            transcription_start_time = time.time()
            whisper_params = {"silence_threshold": threshold}
            result = await transcription_service.transcribe_with_enhancements(
                str(audio_path), 
                word_timestamps=True,
                whisper_params=whisper_params
            )
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

            logger.info(f"Transcription: \"{transcribed_text}\"")
            
            if file_name == "q.speakers.wav":
                if "sou solteira" in transcribed_text.lower():
                    logger.info("✅ SUCCESS: Found 'sou solteira'")
                else:
                    logger.info("❌ FAILED: Did not find 'sou solteira'")
            
            logger.info(f"\n--- Results for {file_name} (silence_threshold={threshold}) ---")
            logger.info(f"  Transcription Accuracy (Normalized): {metrics['accuracy_normalized_percent']:.2f}%")
            logger.info(f"  Diarization Accuracy: {diarization_accuracy:.2f}% ({diarization_result['num_speakers']}/{expected_speakers} speakers)")
            logger.info(f"  Processing Speed: {processing_ratio:.2f}x")

    await transcription_service.unload_model()

if __name__ == "__main__":
    asyncio.run(run_silence_threshold_test())
