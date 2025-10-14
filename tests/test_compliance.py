# tests/test_compliance.py
"""
Compliance Test Suite for TranscrevAI
Validates adherence to project compliance rules (RAM, Speed, Accuracy)
Based on: .claude/compliance.md Rules 1, 7, 22, 23
"""

import pytest
import asyncio
import time
import psutil
import gc
from pathlib import Path
from typing import Dict, Any, List
import sys

# Add root directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import app_state
from config.app_config import get_config

# ==================== COMPLIANCE TARGETS ====================

class ComplianceTargets:
    """
    Compliance targets from .claude/compliance.md
    Updated based on pyannote pivot analysis (11/10/2025)
    """
    # Regra 7: Memory Management
    RAM_HARD_LIMIT_GB = 5.0          # Hard limit (updated from 3.5GB)
    RAM_TARGET_GB = 4.0              # Target (ideal)
    RAM_IDEAL_GB = 3.5               # Original ideal

    # Regra 1: Processing Speed
    SPEED_IDEAL_RATIO = 0.75         # Ideal: 0.75s per 1s audio
    SPEED_ACCEPTABLE_RATIO = 2.0     # Acceptable interim (for accuracy trade-off)
    SPEED_HARD_LIMIT_RATIO = 3.5     # Hard limit (current baseline)

    # Regra 1: Accuracy
    ACCURACY_TARGET_PERCENT = 90.0   # Target: 90%+
    ACCURACY_MIN_PERCENT = 85.0      # Minimum acceptable (interim)
    SPEAKER_ACCURACY_TOLERANCE = 1   # ±1 speaker acceptable


# ==================== BENCHMARK AUDIO FILES ====================

BENCHMARK_FILES = {
    "d.speakers.wav": {
        "expected_speakers": 2,
        "keywords": ["transcrição", "áudio"],
        "min_duration": 5.0  # seconds
    },
    "q.speakers.wav": {
        "expected_speakers": 4,
        "keywords": ["teste", "sistema"],
        "min_duration": 10.0
    },
    "t.speakers.wav": {
        "expected_speakers": 3,
        "keywords": ["gravação", "qualidade"],
        "min_duration": 8.0
    },
    "t2.speakers.wav": {
        "expected_speakers": 3,
        "keywords": ["inteligente", "silicone"],
        "min_duration": 8.0
    }
}


# ==================== HELPER FUNCTION =====================

def _get_available_benchmark_files():
    """Helper to generate benchmark file parameters for this module."""
    recordings_dir = Path(__file__).parent.parent / "data" / "recordings"
    if not recordings_dir.exists():
        return []
    
    files = []
    for filename, metadata in BENCHMARK_FILES.items():
        filepath = recordings_dir / filename
        if filepath.exists():
            # Create a pytest.param for better test IDs
            param = pytest.param(str(filepath), filename, metadata, id=filename)
            files.append(param)
    return files


@pytest.fixture(scope="function")
def memory_tracker():
    """Track memory usage during test"""
    gc.collect()  # Clean up before measurement
    process = psutil.Process()

    class MemoryTracker:
        def __init__(self):
            self.baseline_mb = process.memory_info().rss / (1024 * 1024)
            self.peak_mb = self.baseline_mb
            self.process = process

        def update(self):
            current_mb = self.process.memory_info().rss / (1024 * 1024)
            self.peak_mb = max(self.peak_mb, current_mb)
            return current_mb

        def get_peak_usage_mb(self):
            return self.peak_mb

        def get_delta_mb(self):
            return self.peak_mb - self.baseline_mb

    tracker = MemoryTracker()
    yield tracker

    # Cleanup after test
    gc.collect()


# ==================== COMPLIANCE TESTS ====================

import multiprocessing
from queue import Empty
from src.worker import process_audio_task

# ==================== HELPER FUNCTION =====================

def run_worker_and_get_result(audio_path: str, session_id: str) -> Dict[str, Any]:
    """
    Invokes the worker task directly and retrieves the final result from the queue.
    This is a synchronous helper for use in compliance tests.
    """
    manager = multiprocessing.Manager()
    communication_queue = manager.Queue()
    mock_config = {"model_name": "medium", "device": "cpu"}

    # Execute the worker task in the current process
    process_audio_task(
        audio_path=audio_path,
        session_id=session_id,
        config=mock_config,
        communication_queue=communication_queue
    )

    # Retrieve the final result from the queue
    final_result = None
    while True:
        try:
            # Using a longer timeout as some compliance tests can be slow
            message = communication_queue.get(timeout=180)
            if message.get('type') == 'complete':
                final_result = message.get('result')
                break
            if message.get('type') == 'error':
                pytest.fail(f"Worker task failed for {session_id}: {message.get('message')}")
        except Empty:
            pytest.fail(f"Worker task timed out for {session_id}. No 'complete' message received.")
    
    assert final_result is not None, f"Worker for {session_id} did not produce a final result."
    return final_result


# ==================== COMPLIANCE TESTS ====================

@pytest.mark.parametrize("audio_path,filename,metadata", _get_available_benchmark_files())
def test_ram_compliance_hard_limit(worker_services_fixture, audio_path, filename, metadata):
    """
    COMPLIANCE RULE 7: RAM Hard Limit
    Validates that processing does not exceed hard RAM limit using worker-reported memory.
    """
    import librosa
    audio_duration = librosa.get_duration(path=audio_path)
    session_id = f"test_ram_{Path(audio_path).stem}"

    # Run the task and get the result, which includes memory usage
    result = run_worker_and_get_result(audio_path, session_id)

    peak_memory_mb = result.get("peak_memory_mb")
    assert peak_memory_mb is not None, "Peak memory (peak_memory_mb) was not reported by the worker."

    peak_ram_gb = peak_memory_mb / 1024

    # Log results
    print(f"\n{'='*60}")
    print(f"RAM Compliance Test: {filename}")
    print(f"Audio Duration: {audio_duration:.2f}s")
    print(f"Peak RAM (Reported by Worker): {peak_ram_gb:.2f} GB")
    print(f"Hard Limit: {ComplianceTargets.RAM_HARD_LIMIT_GB} GB")
    print(f"Target: {ComplianceTargets.RAM_TARGET_GB} GB")
    print(f"{ '='*60}")

    # HARD LIMIT - Must pass
    assert peak_ram_gb <= ComplianceTargets.RAM_HARD_LIMIT_GB, \
        f"RAM usage {peak_ram_gb:.2f}GB exceeds hard limit {ComplianceTargets.RAM_HARD_LIMIT_GB}GB"


@pytest.mark.parametrize("audio_path,filename,metadata", _get_available_benchmark_files())
def test_speed_compliance_acceptable(worker_services_fixture, audio_path, filename, metadata):
    """
    COMPLIANCE RULE 1: Processing Speed (Acceptable Interim)
    Validates processing ratio ≤ 2.0x using worker-reported timing.
    """
    session_id = f"test_speed_{Path(audio_path).stem}"

    # Run the task and get the result, which includes the processing ratio
    result = run_worker_and_get_result(audio_path, session_id)

    processing_ratio = result.get("processing_ratio")
    assert processing_ratio is not None, "Processing ratio was not reported by the worker."

    # Log results
    print(f"\n{'='*60}")
    print(f"Speed Compliance Test: {filename}")
    print(f"Audio Duration: {result.get('audio_duration', 0):.2f}s")
    print(f"Processing Time: {result.get('processing_time', 0):.2f}s")
    print(f"Processing Ratio (Reported by Worker): {processing_ratio:.2f}x")
    print(f"Target (Ideal): {ComplianceTargets.SPEED_IDEAL_RATIO}x")
    print(f"Target (Acceptable): {ComplianceTargets.SPEED_ACCEPTABLE_RATIO}x")
    print(f"Hard Limit: {ComplianceTargets.SPEED_HARD_LIMIT_RATIO}x")
    print(f"{ '='*60}")

    # ACCEPTABLE INTERIM - Should pass
    assert processing_ratio <= ComplianceTargets.SPEED_ACCEPTABLE_RATIO, \
        f"Processing ratio {processing_ratio:.2f}x exceeds acceptable limit {ComplianceTargets.SPEED_ACCEPTABLE_RATIO}x"


@pytest.mark.parametrize("audio_path,filename,metadata", _get_available_benchmark_files())
def test_speaker_accuracy_compliance(worker_services_fixture, audio_path, filename, metadata):
    """
    COMPLIANCE RULE 1: Speaker Diarization Accuracy
    Validates speaker count detection within tolerance using worker-reported data.
    """
    session_id = f"test_accuracy_{Path(audio_path).stem}"

    # Run the task and get the result
    result = run_worker_and_get_result(audio_path, session_id)

    detected_speakers = result.get("num_speakers", 0)
    expected_speakers = metadata["expected_speakers"]
    tolerance = ComplianceTargets.SPEAKER_ACCURACY_TOLERANCE

    speaker_error = abs(detected_speakers - expected_speakers)

    # Log results
    print(f"\n{'='*60}")
    print(f"Accuracy Compliance Test: {filename}")
    print(f"Expected Speakers: {expected_speakers}")
    print(f"Detected Speakers: {detected_speakers}")
    print(f"Error: {speaker_error} (Tolerance: ±{tolerance})")
    print(f"Target: Error <= {tolerance}")
    print(f"{ '='*60}")

    # MINIMUM ACCURACY - Should pass with tolerance
    assert speaker_error <= tolerance, \
        f"Speaker detection error {speaker_error} exceeds tolerance ±{tolerance}. " \
        f"Expected {expected_speakers}, got {detected_speakers}"


def test_compliance_summary_report(worker_services_fixture):
    """
    Generate comprehensive compliance summary report.
    Tests all benchmark files and generates aggregate metrics using the new worker architecture.
    """
    results = {
        "ram": [],
        "speed": [],
        "accuracy": []
    }

    available_benchmark_files = _get_available_benchmark_files()
    if not available_benchmark_files:
        pytest.skip("No benchmark audio files found for summary report.")

    for param in available_benchmark_files:
        audio_path, filename, metadata = param.values
        session_id = f"test_summary_{Path(audio_path).stem}"

        # Run the worker and get the consolidated result
        result = run_worker_and_get_result(audio_path, session_id)

        # Extract data from the worker's result payload
        peak_ram_gb = result.get("peak_memory_mb", 0) / 1024
        processing_ratio = result.get("processing_ratio", 0)
        detected_speakers = result.get("num_speakers", 0)

        results["ram"].append({
            "file": filename,
            "ram_gb": peak_ram_gb,
            "passes_hard_limit": peak_ram_gb <= ComplianceTargets.RAM_HARD_LIMIT_GB,
            "passes_target": peak_ram_gb <= ComplianceTargets.RAM_TARGET_GB
        })

        results["speed"].append({
            "file": filename,
            "ratio": processing_ratio,
            "passes_acceptable": processing_ratio <= ComplianceTargets.SPEED_ACCEPTABLE_RATIO,
            "passes_ideal": processing_ratio <= ComplianceTargets.SPEED_IDEAL_RATIO
        })

        results["accuracy"].append({
            "file": filename,
            "detected": detected_speakers,
            "expected": metadata["expected_speakers"],
            "passes": abs(detected_speakers - metadata["expected_speakers"]) <= ComplianceTargets.SPEAKER_ACCURACY_TOLERANCE
        })

    # --- Report Generation (this part remains largely the same) ---
    print("\n" + "="*80)
    print("TRANSCREVAI COMPLIANCE SUMMARY REPORT")
    print("="*80)

    # RAM Summary
    print("\n📊 RAM COMPLIANCE")
    print("-" * 80)
    for r in results["ram"]:
        status = "✅ PASS" if r["passes_hard_limit"] else "❌ FAIL"
        print(f"{r['file']:20} | {r['ram_gb']:.2f} GB | {status}")

    avg_ram = sum(r["ram_gb"] for r in results["ram"]) / len(results["ram"])
    hard_limit_pass_rate = sum(r["passes_hard_limit"] for r in results["ram"]) / len(results["ram"]) * 100
    print(f"\nAverage RAM: {avg_ram:.2f} GB")
    print(f"Hard Limit Pass Rate: {hard_limit_pass_rate:.1f}%")

    # Speed Summary
    print("\n⚡ SPEED COMPLIANCE")
    print("-" * 80)
    for r in results["speed"]:
        status = "✅ PASS" if r["passes_acceptable"] else "❌ FAIL"
        print(f"{r['file']:20} | {r['ratio']:.2f}x | {status}")

    avg_speed = sum(r["ratio"] for r in results["speed"]) / len(results["speed"])
    acceptable_pass_rate = sum(r["passes_acceptable"] for r in results["speed"]) / len(results["speed"]) * 100
    print(f"\nAverage Ratio: {avg_speed:.2f}x")
    print(f"Acceptable Pass Rate: {acceptable_pass_rate:.1f}%")

    # Accuracy Summary
    print("\n🎯 ACCURACY COMPLIANCE")
    print("-" * 80)
    for r in results["accuracy"]:
        status = "✅ PASS" if r["passes"] else "❌ FAIL"
        print(f"{r['file']:20} | Expected: {r['expected']:<2} | Detected: {r['detected']:<2} | {status}")

    accuracy_pass_rate = sum(r["passes"] for r in results["accuracy"]) / len(results["accuracy"]) * 100
    print(f"\nAccuracy Pass Rate: {accuracy_pass_rate:.1f}%")
    print("\n" + "="*80)

    # Overall Assessment
    overall_pass = (
        hard_limit_pass_rate == 100.0 and
        acceptable_pass_rate == 100.0 and
        accuracy_pass_rate == 100.0
    )

    if overall_pass:
        print("✅ OVERALL: PRODUCTION READY (All compliance tests passed)")
    else:
        print("⚠️  OVERALL: NEEDS OPTIMIZATION (Some compliance tests failed)")

    print("="*80 + "\n")


# ==================== PERFORMANCE BENCHMARKS ====================

@pytest.mark.benchmark
def test_benchmark_initialization_time():
    """
    Benchmark: Service initialization time
    Measures time to load transcription and diarization models directly.
    """
    from src.transcription import TranscriptionService
    from src.diarization import PyannoteDiarizer
    from config.app_config import get_config

    config = get_config()
    device = "cpu"

    start_time = time.time()
    # Initialize services directly to measure model loading time
    _ = TranscriptionService(model_name=config.model_name, device=device)
    _ = PyannoteDiarizer(device=device)
    init_time = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"Initialization Benchmark")
    print(f"Time to load models: {init_time:.2f}s")
    print(f"{ '='*60}")

    # Initialization should be reasonably fast (< 30s)
    assert init_time < 30.0, f"Initialization took {init_time:.2f}s, expected < 30s"


@pytest.mark.parametrize("audio_path,filename,metadata", _get_available_benchmark_files())
def test_benchmark_component_timing(audio_path, filename, metadata):
    """
    Benchmark: Break down processing time by component
    Helps identify bottlenecks in the pipeline.
    """
    from src.transcription import TranscriptionService
    from src.diarization import PyannoteDiarizer
    from config.app_config import get_config
    import asyncio

    config = get_config()
    device = "cpu"  # Force CPU for consistent benchmarking

    # Initialize services
    transcription_service = TranscriptionService(model_name=config.model_name, device=device)
    diarization_service = PyannoteDiarizer(device=device)

    import librosa
    audio_duration = librosa.get_duration(path=audio_path)

    async def run_components():
        # Benchmark Transcription
        start_trans = time.time()
        transcription_result = await transcription_service.transcribe_with_enhancements(
            audio_path, word_timestamps=True
        )
        transcription_time = time.time() - start_trans

        # Benchmark Diarization
        start_diar = time.time()
        await diarization_service.diarize(
            audio_path, transcription_result.segments
        )
        diarization_time = time.time() - start_diar
        
        return transcription_time, diarization_time

    # Run the async inner function
    transcription_time, diarization_time = asyncio.run(run_components())

    total_time = transcription_time + diarization_time

    # Calculate percentages
    transcription_pct = (transcription_time / total_time) * 100 if total_time > 0 else 0
    diarization_pct = (diarization_time / total_time) * 100 if total_time > 0 else 0

    print(f"\n{'='*60}")
    print(f"Component Timing Benchmark: {filename}")
    print(f"Audio Duration: {audio_duration:.2f}s")
    print(f"-" * 60)
    print(f"Transcription: {transcription_time:.2f}s ({transcription_pct:.1f}%)")
    print(f"Diarization:   {diarization_time:.2f}s ({diarization_pct:.1f}%)")
    print(f"Total:         {total_time:.2f}s")
    print(f"Ratio:         {total_time/audio_duration:.2f}x")
    print(f"{ '='*60}")

    # Store for later analysis
    return {
        "file": filename,
        "audio_duration": audio_duration,
        "transcription_time": transcription_time,
        "diarization_time": diarization_time,
        "total_time": total_time,
        "ratio": total_time / audio_duration
    }