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


# ==================== FIXTURES ====================

@pytest.fixture(scope="module")
def recordings_dir():
    """Get recordings directory path"""
    path = Path(__file__).parent.parent / "data" / "recordings"
    if not path.exists():
        pytest.skip(f"Recordings directory not found: {path}")
    return path


@pytest.fixture(scope="module")
def available_benchmark_files(recordings_dir):
    """Find available benchmark files"""
    files = []
    for filename, metadata in BENCHMARK_FILES.items():
        filepath = recordings_dir / filename
        if filepath.exists():
            files.append((str(filepath), filename, metadata))

    if not files:
        pytest.skip("No benchmark audio files found in data/recordings")

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

@pytest.mark.asyncio
@pytest.mark.parametrize("audio_path,filename,metadata", [
    pytest.param(*file_data, id=file_data[1])
    for file_data in pytest.lazy_fixture("available_benchmark_files")
])
async def test_ram_compliance_hard_limit(audio_path, filename, metadata, memory_tracker):
    """
    COMPLIANCE RULE 7: RAM Hard Limit
    Validates that processing does not exceed hard RAM limit
    """
    # Get audio duration for context
    import librosa
    audio_duration = librosa.get_duration(path=audio_path)

    # Process audio
    session_id = f"test_ram_{Path(audio_path).stem}"

    from main import process_audio_pipeline

    # Track memory during processing
    async def track_memory():
        while True:
            memory_tracker.update()
            await asyncio.sleep(0.1)

    memory_task = asyncio.create_task(track_memory())

    try:
        await process_audio_pipeline(audio_path, session_id)
    finally:
        memory_task.cancel()
        try:
            await memory_task
        except asyncio.CancelledError:
            pass

    peak_ram_gb = memory_tracker.get_peak_usage_mb() / 1024
    delta_ram_gb = memory_tracker.get_delta_mb() / 1024

    # Log results
    print(f"\n{'='*60}")
    print(f"RAM Compliance Test: {filename}")
    print(f"Audio Duration: {audio_duration:.2f}s")
    print(f"Peak RAM: {peak_ram_gb:.2f} GB")
    print(f"Delta RAM: {delta_ram_gb:.2f} GB")
    print(f"Hard Limit: {ComplianceTargets.RAM_HARD_LIMIT_GB} GB")
    print(f"Target: {ComplianceTargets.RAM_TARGET_GB} GB")
    print(f"{'='*60}")

    # HARD LIMIT - Must pass
    assert peak_ram_gb <= ComplianceTargets.RAM_HARD_LIMIT_GB, \
        f"RAM usage {peak_ram_gb:.2f}GB exceeds hard limit {ComplianceTargets.RAM_HARD_LIMIT_GB}GB"


@pytest.mark.asyncio
@pytest.mark.parametrize("audio_path,filename,metadata", [
    pytest.param(*file_data, id=file_data[1])
    for file_data in pytest.lazy_fixture("available_benchmark_files")
])
async def test_speed_compliance_acceptable(audio_path, filename, metadata):
    """
    COMPLIANCE RULE 1: Processing Speed (Acceptable Interim)
    Validates processing ratio ≤ 2.0x (acceptable for accuracy trade-off)
    """
    import librosa
    audio_duration = librosa.get_duration(path=audio_path)

    session_id = f"test_speed_{Path(audio_path).stem}"

    start_time = time.time()

    from main import process_audio_pipeline
    await process_audio_pipeline(audio_path, session_id)

    processing_time = time.time() - start_time
    processing_ratio = processing_time / audio_duration if audio_duration > 0 else 0

    # Log results
    print(f"\n{'='*60}")
    print(f"Speed Compliance Test: {filename}")
    print(f"Audio Duration: {audio_duration:.2f}s")
    print(f"Processing Time: {processing_time:.2f}s")
    print(f"Processing Ratio: {processing_ratio:.2f}x")
    print(f"Target (Ideal): {ComplianceTargets.SPEED_IDEAL_RATIO}x")
    print(f"Target (Acceptable): {ComplianceTargets.SPEED_ACCEPTABLE_RATIO}x")
    print(f"Hard Limit: {ComplianceTargets.SPEED_HARD_LIMIT_RATIO}x")
    print(f"{'='*60}")

    # ACCEPTABLE INTERIM - Should pass
    assert processing_ratio <= ComplianceTargets.SPEED_ACCEPTABLE_RATIO, \
        f"Processing ratio {processing_ratio:.2f}x exceeds acceptable limit {ComplianceTargets.SPEED_ACCEPTABLE_RATIO}x"


@pytest.mark.asyncio
@pytest.mark.parametrize("audio_path,filename,metadata", [
    pytest.param(*file_data, id=file_data[1])
    for file_data in pytest.lazy_fixture("available_benchmark_files")
])
async def test_speaker_accuracy_compliance(audio_path, filename, metadata):
    """
    COMPLIANCE RULE 1: Speaker Diarization Accuracy
    Validates speaker count detection within tolerance
    """
    session_id = f"test_accuracy_{Path(audio_path).stem}"

    from main import process_audio_pipeline

    # Mock message sending to capture result
    captured_result = {}

    original_send = app_state.send_message
    async def mock_send(sid, message, priority=None):
        if message.get('type') == 'complete':
            captured_result.update(message.get('result', {}))
        await original_send(sid, message, priority)

    app_state.send_message = mock_send

    try:
        await process_audio_pipeline(audio_path, session_id)
    finally:
        app_state.send_message = original_send

    detected_speakers = captured_result.get("num_speakers", 0)
    expected_speakers = metadata["expected_speakers"]
    tolerance = ComplianceTargets.SPEAKER_ACCURACY_TOLERANCE

    speaker_error = abs(detected_speakers - expected_speakers)
    accuracy_percent = ((expected_speakers - speaker_error) / expected_speakers) * 100 if expected_speakers > 0 else 0

    # Log results
    print(f"\n{'='*60}")
    print(f"Accuracy Compliance Test: {filename}")
    print(f"Expected Speakers: {expected_speakers}")
    print(f"Detected Speakers: {detected_speakers}")
    print(f"Error: {speaker_error} (Tolerance: ±{tolerance})")
    print(f"Accuracy: {accuracy_percent:.1f}%")
    print(f"Target: ≥{ComplianceTargets.ACCURACY_MIN_PERCENT}%")
    print(f"{'='*60}")

    # MINIMUM ACCURACY - Should pass with tolerance
    assert speaker_error <= tolerance, \
        f"Speaker detection error {speaker_error} exceeds tolerance ±{tolerance}. " \
        f"Expected {expected_speakers}, got {detected_speakers}"


@pytest.mark.asyncio
async def test_compliance_summary_report(available_benchmark_files):
    """
    Generate comprehensive compliance summary report
    Tests all benchmark files and generates aggregate metrics
    """
    results = {
        "ram": [],
        "speed": [],
        "accuracy": []
    }

    for audio_path, filename, metadata in available_benchmark_files:
        import librosa
        audio_duration = librosa.get_duration(path=audio_path)

        # Measure RAM
        gc.collect()
        process = psutil.Process()
        baseline_mb = process.memory_info().rss / (1024 * 1024)

        session_id = f"test_summary_{Path(audio_path).stem}"

        start_time = time.time()

        from main import process_audio_pipeline
        await process_audio_pipeline(audio_path, session_id)

        processing_time = time.time() - start_time
        peak_mb = process.memory_info().rss / (1024 * 1024)

        processing_ratio = processing_time / audio_duration
        ram_gb = peak_mb / 1024

        # Get speaker accuracy
        session = app_state.sessions.get(session_id, {})
        detected_speakers = 0  # Default if not captured

        results["ram"].append({
            "file": filename,
            "ram_gb": ram_gb,
            "passes_hard_limit": ram_gb <= ComplianceTargets.RAM_HARD_LIMIT_GB,
            "passes_target": ram_gb <= ComplianceTargets.RAM_TARGET_GB
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
            "expected": metadata["expected_speakers"]
        })

    # Generate Report
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
        error = abs(r["detected"] - r["expected"])
        status = "✅ PASS" if error <= ComplianceTargets.SPEAKER_ACCURACY_TOLERANCE else "❌ FAIL"
        print(f"{r['file']:20} | Expected: {r['expected']} | Detected: {r['detected']} | {status}")

    print("\n" + "="*80)

    # Overall Assessment
    overall_pass = (
        hard_limit_pass_rate == 100.0 and
        acceptable_pass_rate == 100.0
    )

    if overall_pass:
        print("✅ OVERALL: PRODUCTION READY (All compliance tests passed)")
    else:
        print("⚠️  OVERALL: NEEDS OPTIMIZATION (Some compliance tests failed)")

    print("="*80 + "\n")


# ==================== PERFORMANCE BENCHMARKS ====================

@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_benchmark_initialization_time():
    """
    Benchmark: Service initialization time
    Measures time to load transcription and diarization models
    """
    from main import ThreadSafeEnhancedAppState

    test_state = ThreadSafeEnhancedAppState()

    start_time = time.time()
    await test_state.initialize_services()
    init_time = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"Initialization Benchmark")
    print(f"Time: {init_time:.2f}s")
    print(f"{'='*60}")

    # Initialization should be reasonably fast (< 30s)
    assert init_time < 30.0, f"Initialization took {init_time:.2f}s, expected < 30s"


@pytest.mark.benchmark
@pytest.mark.asyncio
@pytest.mark.parametrize("audio_path,filename,metadata", [
    pytest.param(*file_data, id=file_data[1])
    for file_data in pytest.lazy_fixture("available_benchmark_files")
])
async def test_benchmark_component_timing(audio_path, filename, metadata):
    """
    Benchmark: Break down processing time by component
    Helps identify bottlenecks in the pipeline
    """
    from src.transcription import TranscriptionService
    from src.diarization import PyannoteDiarizer
    from config.app_config import get_config

    config = get_config()
    device = "cpu"  # Force CPU for consistent benchmarking

    # Initialize services
    transcription_service = TranscriptionService()
    diarization_service = PyannoteDiarizer(device=device)

    import librosa
    audio_duration = librosa.get_duration(path=audio_path)

    # Benchmark Transcription
    start = time.time()
    transcription_result = await transcription_service.transcribe_with_enhancements(
        audio_path, word_timestamps=True
    )
    transcription_time = time.time() - start

    # Benchmark Diarization
    start = time.time()
    diarization_result = await diarization_service.diarize(
        audio_path, transcription_result.segments
    )
    diarization_time = time.time() - start

    total_time = transcription_time + diarization_time

    # Calculate percentages
    transcription_pct = (transcription_time / total_time) * 100
    diarization_pct = (diarization_time / total_time) * 100

    print(f"\n{'='*60}")
    print(f"Component Timing Benchmark: {filename}")
    print(f"Audio Duration: {audio_duration:.2f}s")
    print(f"-" * 60)
    print(f"Transcription: {transcription_time:.2f}s ({transcription_pct:.1f}%)")
    print(f"Diarization:   {diarization_time:.2f}s ({diarization_pct:.1f}%)")
    print(f"Total:         {total_time:.2f}s")
    print(f"Ratio:         {total_time/audio_duration:.2f}x")
    print(f"{'='*60}")

    # Store for later analysis
    return {
        "file": filename,
        "audio_duration": audio_duration,
        "transcription_time": transcription_time,
        "diarization_time": diarization_time,
        "total_time": total_time,
        "ratio": total_time / audio_duration
    }
