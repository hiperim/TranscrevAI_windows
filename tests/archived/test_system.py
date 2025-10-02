#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test for TranscrevAI improvements
"""

import os
import sys
import time
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_transcription():
    """Test transcription with the improved system"""

    try:
        from transcription import TranscriptionService
        print("SUCCESS: Imports working correctly")

        # Initialize service
        transcription_service = TranscriptionService()
        print("SUCCESS: TranscriptionService initialized")

        # Test with a small file
        test_file = "data/recordings/d.speakers.wav"

        if not os.path.exists(test_file):
            print(f"File not found: {test_file}")
            return

        print(f"\nTesting: {os.path.basename(test_file)}")

        start_time = time.time()

        # Transcribe async
        result = await transcription_service.transcribe_audio_file(test_file, "pt")

        end_time = time.time()
        processing_time = end_time - start_time

        if result and result.get("segments"):
            audio_duration = result.get("total_duration", 0)
            processing_ratio = result.get("processing_ratio", float('inf'))
            target_achieved = result.get("target_achieved", False)

            print(f"\nRESULTS:")
            print(f"Duration: {audio_duration:.1f}s")
            print(f"Processing Time: {processing_time:.1f}s")
            print(f"Ratio: {processing_ratio:.3f}x")
            print(f"Target Achieved: {target_achieved}")
            print(f"Segments: {len(result['segments'])}")

            # Show validation working
            expected_target = processing_ratio < 0.5
            print(f"\nVALIDATION CHECK:")
            print(f"Expected target achieved: {expected_target}")
            print(f"Actual target achieved: {target_achieved}")

            if expected_target == target_achieved:
                print("VALIDATION FIX: Working correctly!")
            else:
                print("VALIDATION ERROR: Logic still broken!")

            # Show sample text
            if result["segments"]:
                sample_text = result["segments"][0].get("text", "").strip()[:80]
                print(f"\nSample text: \"{sample_text}...\"")

            print(f"\nIMPROVEMENTS VERIFIED:")
            print(f"1. Validation logic: {'FIXED' if expected_target == target_achieved else 'BROKEN'}")
            print(f"2. Processing ratio: {processing_ratio:.3f}x ({'GOOD' if processing_ratio < 0.5 else 'NEEDS_WORK'})")
            print(f"3. System functional: {'YES' if result.get('segments') else 'NO'}")

        else:
            print("ERROR: No transcription result")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("TranscrevAI System Test")
    print("=" * 40)

    # Run async test
    asyncio.run(test_transcription())

if __name__ == "__main__":
    main()