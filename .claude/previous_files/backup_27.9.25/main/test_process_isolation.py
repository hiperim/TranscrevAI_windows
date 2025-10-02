#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Process Isolation - Force high memory usage to test isolation
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def force_memory_pressure():
    """Simulate high memory usage to trigger process isolation"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"Current memory: {memory.percent:.1f}%")

        if memory.percent < 85:
            print("Creating memory pressure to test process isolation...")
            # Allocate memory to reach 85%+ usage (for testing only)
            target_memory = 85.0
            current_memory = memory.percent

            if current_memory < target_memory:
                # Calculate how much memory to allocate
                available_gb = memory.available / (1024**3)
                total_gb = memory.total / (1024**3)

                # We need to reduce available memory
                needed_reduction = (target_memory - current_memory) / 100 * total_gb

                print(f"Allocating ~{needed_reduction:.1f}GB to test process isolation...")

                # Don't actually allocate - just test the isolation manually
                print("SIMULATING 87% memory usage for process isolation test...")
                return 87.0

        return memory.percent
    except Exception as e:
        print(f"Memory check failed: {e}")
        return 50.0

def test_process_isolation_directly():
    """Test process isolation directly"""
    print("="*60)
    print("TESTING PROCESS ISOLATION DIRECTLY")
    print("="*60)

    try:
        from transcription import OptimizedTranscriber
        print("SUCCESS: OptimizedTranscriber imported")

        # Test file
        test_file = "data/recordings/t2.speakers.wav"

        if not os.path.exists(test_file):
            print(f"SKIP: File not found: {test_file}")
            return

        print(f"\nTesting with: {os.path.basename(test_file)}")
        print("-" * 40)

        # Initialize transcriber
        transcriber = OptimizedTranscriber("medium")

        # Force memory check simulation
        memory_percent = force_memory_pressure()

        print(f"Memory usage: {memory_percent:.1f}%")

        if memory_percent > 85:
            print("Testing PROCESS ISOLATION mode...")

            # Test process isolation directly
            result = transcriber.transcribe_with_process_isolation(test_file)

            if result and result.get("segments"):
                processing_time = result.get("processing_time", 0)
                text = result.get("text", "")
                segments = result.get("segments", [])

                print(f"\nPROCESS ISOLATION RESULTS:")
                print(f"Processing time: {processing_time:.2f}s")
                print(f"Text length: {len(text)} characters")
                print(f"Segments: {len(segments)}")
                print(f"Sample text: \"{text[:100]}...\"")

                # Check if real transcription
                is_real = len(text) > 0 and "placeholder" not in text.lower()
                print(f"Real transcription: {is_real}")

                if is_real:
                    print("SUCCESS: Process isolation working with real transcription!")
                else:
                    print("ISSUE: Process isolation not generating real transcription")

            else:
                error = result.get("error", "Unknown error")
                print(f"ERROR: Process isolation failed: {error}")

        else:
            print("Memory usage too low for process isolation test")
            print("Testing regular transcription...")

            result = transcriber.transcribe_parallel(test_file)

            if result and result.get("segments"):
                print("SUCCESS: Regular transcription working")
            else:
                print("ERROR: Regular transcription failed")

    except Exception as e:
        print(f"Test error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_process_isolation_directly()