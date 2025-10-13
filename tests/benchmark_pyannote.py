#!/usr/bin/env python3
# tests/benchmark_pyannote.py
"""
Standalone Benchmark Script for pyannote.audio Validation
Validates RAM, Speed, and Accuracy for production readiness

Usage:
    python tests/benchmark_pyannote.py
    python tests/benchmark_pyannote.py --file data/recordings/q.speakers.wav
    python tests/benchmark_pyannote.py --all
"""

import argparse
import asyncio
import time
import psutil
import gc
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Add root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Ensure environment variable is set
if not os.getenv("HUGGING_FACE_HUB_TOKEN"):
    print("⚠️  Warning: HUGGING_FACE_HUB_TOKEN not set. Pyannote.audio may fail.")
    print("   Get token from: https://hf.co/settings/tokens")


# ==================== CONFIGURATION ====================

class BenchmarkConfig:
    """Benchmark configuration and compliance targets"""

    # Compliance Targets
    RAM_HARD_LIMIT_GB = 5.0      # Maximum acceptable
    RAM_TARGET_GB = 4.0          # Desired target
    RAM_IDEAL_GB = 3.5           # Original ideal

    SPEED_IDEAL_RATIO = 0.75     # Ideal: 0.75s per 1s audio
    SPEED_ACCEPTABLE_RATIO = 2.0  # Acceptable for accuracy trade-off
    SPEED_HARD_LIMIT_RATIO = 3.5  # Current baseline

    ACCURACY_TARGET = 90.0       # 90%+ target
    ACCURACY_MIN = 85.0          # Minimum acceptable
    SPEAKER_TOLERANCE = 1        # ±1 speaker acceptable

    # Benchmark Files
    BENCHMARK_FILES = {
        "d.speakers.wav": {"expected_speakers": 2, "min_duration": 5.0},
        "q.speakers.wav": {"expected_speakers": 4, "min_duration": 10.0},
        "t.speakers.wav": {"expected_speakers": 3, "min_duration": 8.0},
        "t2.speakers.wav": {"expected_speakers": 3, "min_duration": 8.0}
    }


# ==================== MEMORY TRACKER ====================

class MemoryTracker:
    """Tracks memory usage throughout benchmark"""

    def __init__(self):
        gc.collect()
        self.process = psutil.Process()
        self.baseline_mb = self.process.memory_info().rss / (1024 * 1024)
        self.peak_mb = self.baseline_mb
        self.samples = []

    def update(self):
        """Update peak memory"""
        current_mb = self.process.memory_info().rss / (1024 * 1024)
        self.peak_mb = max(self.peak_mb, current_mb)
        self.samples.append(current_mb)
        return current_mb

    def get_peak_gb(self):
        """Get peak memory in GB"""
        return self.peak_mb / 1024

    def get_delta_gb(self):
        """Get delta from baseline in GB"""
        return (self.peak_mb - self.baseline_mb) / 1024

    def reset(self):
        """Reset for new measurement"""
        gc.collect()
        self.baseline_mb = self.process.memory_info().rss / (1024 * 1024)
        self.peak_mb = self.baseline_mb
        self.samples = []


# ==================== BENCHMARK RUNNER ====================

class PyannnoteBenchmarkRunner:
    """Runs comprehensive benchmarks on pyannote.audio"""

    def __init__(self):
        self.config = BenchmarkConfig()
        self.results = {
            "ram": [],
            "speed": [],
            "accuracy": [],
            "initialization": {}
        }

    async def benchmark_initialization(self) -> float:
        """Benchmark service initialization time"""
        print("\n" + "="*80)
        print("🔧 INITIALIZATION BENCHMARK")
        print("="*80)

        memory_tracker = MemoryTracker()

        start_time = time.time()

        # Initialize services
        from src.transcription import TranscriptionService
        from src.diarization import PyannoteDiarizer

        print("Loading transcription model...")
        transcription_service = TranscriptionService()

        print("Loading diarization model...")
        device = "cpu"  # Force CPU for consistent benchmarking
        diarization_service = PyannoteDiarizer(device=device)

        init_time = time.time() - start_time
        init_ram_gb = memory_tracker.get_delta_gb()

        print(f"\n✓ Initialization Complete")
        print(f"  Time: {init_time:.2f}s")
        print(f"  RAM Used: {init_ram_gb:.2f} GB")

        self.results["initialization"] = {
            "time_seconds": init_time,
            "ram_gb": init_ram_gb
        }

        return transcription_service, diarization_service

    async def benchmark_single_file(
        self,
        audio_path: str,
        filename: str,
        metadata: Dict,
        transcription_service,
        diarization_service
    ) -> Dict:
        """Benchmark single audio file"""
        print(f"\n{'='*80}")
        print(f"📊 BENCHMARKING: {filename}")
        print(f"{'='*80}")

        import librosa

        # Get audio info
        try:
            audio_duration = librosa.get_duration(path=audio_path)
        except Exception as e:
            print(f"❌ Error loading audio: {e}")
            return None

        print(f"Audio Duration: {audio_duration:.2f}s")
        print(f"Expected Speakers: {metadata['expected_speakers']}")

        # Setup memory tracking
        memory_tracker = MemoryTracker()

        # Track memory in background
        async def track_memory():
            while True:
                memory_tracker.update()
                await asyncio.sleep(0.1)

        memory_task = asyncio.create_task(track_memory())

        # Benchmark transcription
        print("\n🎤 Transcription Phase...")
        transcription_start = time.time()

        try:
            transcription_result = await transcription_service.transcribe_with_enhancements(
                audio_path, word_timestamps=True
            )
            transcription_time = time.time() - transcription_start
            print(f"  ✓ Completed in {transcription_time:.2f}s")
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            memory_task.cancel()
            return None

        # Benchmark diarization
        print("\n👥 Diarization Phase...")
        diarization_start = time.time()

        try:
            diarization_result = await diarization_service.diarize(
                audio_path, transcription_result.segments
            )
            diarization_time = time.time() - diarization_start
            print(f"  ✓ Completed in {diarization_time:.2f}s")
            print(f"  ✓ Detected {diarization_result['num_speakers']} speakers")
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            memory_task.cancel()
            return None
        finally:
            memory_task.cancel()
            try:
                await memory_task
            except asyncio.CancelledError:
                pass

        # Calculate metrics
        total_time = transcription_time + diarization_time
        processing_ratio = total_time / audio_duration if audio_duration > 0 else 0
        peak_ram_gb = memory_tracker.get_peak_gb()

        detected_speakers = diarization_result['num_speakers']
        expected_speakers = metadata['expected_speakers']
        speaker_error = abs(detected_speakers - expected_speakers)

        # Print results
        print(f"\n{'─'*80}")
        print(f"RESULTS:")
        print(f"{'─'*80}")
        print(f"⏱️  Processing Time:")
        print(f"   Transcription: {transcription_time:.2f}s ({transcription_time/total_time*100:.1f}%)")
        print(f"   Diarization:   {diarization_time:.2f}s ({diarization_time/total_time*100:.1f}%)")
        print(f"   Total:         {total_time:.2f}s")
        print(f"   Ratio:         {processing_ratio:.2f}x")

        print(f"\n💾 Memory Usage:")
        print(f"   Peak RAM:      {peak_ram_gb:.2f} GB")

        print(f"\n🎯 Accuracy:")
        print(f"   Expected:      {expected_speakers} speakers")
        print(f"   Detected:      {detected_speakers} speakers")
        print(f"   Error:         {speaker_error} (Tolerance: ±{self.config.SPEAKER_TOLERANCE})")

        # Evaluate against targets
        print(f"\n{'─'*80}")
        print(f"COMPLIANCE:")
        print(f"{'─'*80}")

        speed_status = "✅ PASS" if processing_ratio <= self.config.SPEED_ACCEPTABLE_RATIO else "❌ FAIL"
        ram_status = "✅ PASS" if peak_ram_gb <= self.config.RAM_HARD_LIMIT_GB else "❌ FAIL"
        accuracy_status = "✅ PASS" if speaker_error <= self.config.SPEAKER_TOLERANCE else "❌ FAIL"

        print(f"Speed (≤{self.config.SPEED_ACCEPTABLE_RATIO}x):  {processing_ratio:.2f}x  {speed_status}")
        print(f"RAM (≤{self.config.RAM_HARD_LIMIT_GB}GB):   {peak_ram_gb:.2f}GB {ram_status}")
        print(f"Accuracy (±{self.config.SPEAKER_TOLERANCE}):     {speaker_error}      {accuracy_status}")

        # Store results
        result = {
            "file": filename,
            "audio_duration": audio_duration,
            "transcription_time": transcription_time,
            "diarization_time": diarization_time,
            "total_time": total_time,
            "processing_ratio": processing_ratio,
            "peak_ram_gb": peak_ram_gb,
            "expected_speakers": expected_speakers,
            "detected_speakers": detected_speakers,
            "speaker_error": speaker_error,
            "speed_pass": processing_ratio <= self.config.SPEED_ACCEPTABLE_RATIO,
            "ram_pass": peak_ram_gb <= self.config.RAM_HARD_LIMIT_GB,
            "accuracy_pass": speaker_error <= self.config.SPEAKER_TOLERANCE
        }

        return result

    async def run_benchmarks(self, audio_files: List[Tuple[str, str, Dict]]):
        """Run benchmarks on all provided audio files"""

        # Initialize services once
        transcription_service, diarization_service = await self.benchmark_initialization()

        # Benchmark each file
        for audio_path, filename, metadata in audio_files:
            result = await self.benchmark_single_file(
                audio_path, filename, metadata,
                transcription_service, diarization_service
            )

            if result:
                self.results["speed"].append(result)
                self.results["ram"].append(result)
                self.results["accuracy"].append(result)

        # Generate summary report
        self.generate_summary_report()

    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "="*80)
        print("📋 COMPREHENSIVE BENCHMARK SUMMARY")
        print("="*80)

        if not self.results["speed"]:
            print("\n⚠️  No benchmark results available")
            return

        # Speed Summary
        print(f"\n{'⚡ SPEED PERFORMANCE':^80}")
        print("─"*80)
        print(f"{'File':<20} {'Duration':<10} {'Time':<10} {'Ratio':<10} {'Status':<10}")
        print("─"*80)

        for r in self.results["speed"]:
            status = "✅" if r["speed_pass"] else "❌"
            print(f"{r['file']:<20} {r['audio_duration']:>7.1f}s  {r['total_time']:>7.1f}s  {r['processing_ratio']:>7.2f}x  {status}")

        avg_ratio = sum(r["processing_ratio"] for r in self.results["speed"]) / len(self.results["speed"])
        speed_pass_rate = sum(r["speed_pass"] for r in self.results["speed"]) / len(self.results["speed"]) * 100

        print("─"*80)
        print(f"Average Ratio: {avg_ratio:.2f}x")
        print(f"Pass Rate:     {speed_pass_rate:.1f}% (Target: ≤{self.config.SPEED_ACCEPTABLE_RATIO}x)")

        # RAM Summary
        print(f"\n{'💾 MEMORY USAGE':^80}")
        print("─"*80)
        print(f"{'File':<20} {'Peak RAM':<15} {'Status':<10}")
        print("─"*80)

        for r in self.results["ram"]:
            status = "✅" if r["ram_pass"] else "❌"
            print(f"{r['file']:<20} {r['peak_ram_gb']:>10.2f} GB  {status}")

        avg_ram = sum(r["peak_ram_gb"] for r in self.results["ram"]) / len(self.results["ram"])
        ram_pass_rate = sum(r["ram_pass"] for r in self.results["ram"]) / len(self.results["ram"]) * 100

        print("─"*80)
        print(f"Average RAM:   {avg_ram:.2f} GB")
        print(f"Pass Rate:     {ram_pass_rate:.1f}% (Target: ≤{self.config.RAM_HARD_LIMIT_GB}GB)")

        # Accuracy Summary
        print(f"\n{'🎯 SPEAKER ACCURACY':^80}")
        print("─"*80)
        print(f"{'File':<20} {'Expected':<12} {'Detected':<12} {'Error':<10} {'Status':<10}")
        print("─"*80)

        for r in self.results["accuracy"]:
            status = "✅" if r["accuracy_pass"] else "❌"
            print(f"{r['file']:<20} {r['expected_speakers']:>8}     {r['detected_speakers']:>8}     {r['speaker_error']:>6}  {status}")

        accuracy_pass_rate = sum(r["accuracy_pass"] for r in self.results["accuracy"]) / len(self.results["accuracy"]) * 100

        print("─"*80)
        print(f"Pass Rate:     {accuracy_pass_rate:.1f}% (Tolerance: ±{self.config.SPEAKER_TOLERANCE})")

        # Overall Assessment
        print(f"\n{'═'*80}")
        print(f"{'🏆 OVERALL ASSESSMENT':^80}")
        print(f"{'═'*80}\n")

        all_pass = (speed_pass_rate == 100.0 and ram_pass_rate == 100.0 and accuracy_pass_rate == 100.0)

        if all_pass:
            print("✅ PRODUCTION READY")
            print("   All compliance targets met. System is ready for production deployment.")
        else:
            print("⚠️  OPTIMIZATION NEEDED")
            if speed_pass_rate < 100:
                print(f"   • Speed: {100-speed_pass_rate:.0f}% of tests failed (target: ≤{self.config.SPEED_ACCEPTABLE_RATIO}x)")
            if ram_pass_rate < 100:
                print(f"   • RAM: {100-ram_pass_rate:.0f}% of tests failed (target: ≤{self.config.RAM_HARD_LIMIT_GB}GB)")
            if accuracy_pass_rate < 100:
                print(f"   • Accuracy: {100-accuracy_pass_rate:.0f}% of tests failed (tolerance: ±{self.config.SPEAKER_TOLERANCE})")

        print(f"\n{'═'*80}\n")

        # Save results to JSON
        self.save_results()

    def save_results(self):
        """Save benchmark results to JSON file"""
        output_dir = Path(__file__).parent.parent / "benchmarks"
        output_dir.mkdir(exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"benchmark_{timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"📁 Results saved to: {output_file}")


# ==================== MAIN ====================

async def main():
    parser = argparse.ArgumentParser(description="Benchmark pyannote.audio for TranscrevAI")
    parser.add_argument("--file", help="Single audio file to benchmark")
    parser.add_argument("--all", action="store_true", help="Benchmark all files in recordings directory")
    args = parser.parse_args()

    recordings_dir = Path(__file__).parent.parent / "data" / "recordings"

    # Determine which files to benchmark
    audio_files = []

    if args.file:
        # Single file
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"❌ Error: File not found: {args.file}")
            return

        filename = file_path.name
        metadata = BenchmarkConfig.BENCHMARK_FILES.get(filename, {"expected_speakers": 0})
        audio_files.append((str(file_path), filename, metadata))

    else:
        # All benchmark files
        if not recordings_dir.exists():
            print(f"❌ Error: Recordings directory not found: {recordings_dir}")
            return

        for filename, metadata in BenchmarkConfig.BENCHMARK_FILES.items():
            file_path = recordings_dir / filename
            if file_path.exists():
                audio_files.append((str(file_path), filename, metadata))

    if not audio_files:
        print("❌ Error: No audio files found to benchmark")
        print(f"   Expected files in: {recordings_dir}")
        print(f"   Files: {', '.join(BenchmarkConfig.BENCHMARK_FILES.keys())}")
        return

    # Run benchmarks
    runner = PyannnoteBenchmarkRunner()
    await runner.run_benchmarks(audio_files)


if __name__ == "__main__":
    asyncio.run(main())
