#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Production System - Simple test for real TranscrevAI
"""

import os
import sys
import time
import gc
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def force_memory_cleanup():
    """Force aggressive memory cleanup"""
    print("Forcing memory cleanup...")
    gc.collect()

    # Try to free some system memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"Memory: {memory.percent:.1f}% used, {memory.available / (1024**3):.1f}GB available")
    except:
        pass

def test_production_system():
    """Test the real production TranscrevAI system"""

    print("="*60)
    print("TESTING PRODUCTION TRANSCREVAI SYSTEM")
    print("="*60)

    # Force cleanup first
    force_memory_cleanup()

    try:
        from transcription import OptimizedTranscriber
        print("SUCCESS: Using OptimizedTranscriber (production system)")

        # Test file
        test_file = "data/recordings/t2.speakers.wav"

        if not os.path.exists(test_file):
            print(f"SKIP: File not found: {test_file}")
            return

        print(f"\nTesting: {os.path.basename(test_file)}")
        print("-" * 40)

        # Force cleanup before loading
        force_memory_cleanup()

        # Initialize production transcriber
        transcriber = OptimizedTranscriber("medium")

        start_time = time.time()

        # Test with real audio using parallel transcription (with process isolation if needed)
        result = transcriber.transcribe_parallel(test_file)

        end_time = time.time()
        processing_time = end_time - start_time

        print(f"\nRESULTS:")
        print(f"Processing time: {processing_time:.2f}s")

        if result and result.get("segments"):
            # Calculate duration
            audio_duration = 10.6  # Known duration
            processing_ratio = processing_time / audio_duration

            segments = result.get("segments", [])
            result_text = " ".join([seg.get("text", "") for seg in segments])

            print(f"Audio duration: {audio_duration:.1f}s")
            print(f"Processing ratio: {processing_ratio:.3f}x")
            print(f"Segments: {len(segments)}")
            print(f"Text length: {len(result_text)} characters")

            # Show sample text
            if result_text:
                sample_text = result_text[:100]
                print(f"Sample text: \"{sample_text}...\"")

            # Check if realistic
            is_realistic = 1.0 <= processing_ratio <= 3.0
            print(f"\nVALIDATION:")
            print(f"Realistic performance (1.0-3.0x): {is_realistic}")
            print(f"Real transcription: {len(result_text) > 0}")

            if is_realistic and len(result_text) > 0:
                print("SUCCESS: Production system working correctly!")
            else:
                print("ISSUES: Performance or output problems detected")

        else:
            print("ERROR: No transcription result")

    except ImportError as e:
        print(f"Import error: {e}")
        print("Trying TranscriptionService instead...")

        try:
            # Fallback to TranscriptionService
            from transcription import TranscriptionService
            print("Using TranscriptionService as fallback")

            # But this requires async, so we'll just report the import success
            print("SUCCESS: TranscriptionService imported (async needed)")

        except ImportError as e2:
            print(f"All imports failed: {e2}")

    except Exception as e:
        print(f"Test error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_production_system()