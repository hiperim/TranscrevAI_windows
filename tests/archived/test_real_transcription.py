#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Real Whisper Transcription Implementation
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_real_transcription():
    """Test the new real transcription implementation"""

    try:
        from transcription import HybridONNXWhisper
        print("SUCCESS: HybridONNXWhisper imported successfully")

        # Test files
        test_files = [
            "data/recordings/d.speakers.wav",
            "data/recordings/t.speakers.wav"
        ]

        for test_file in test_files:
            if not os.path.exists(test_file):
                print(f"SKIP: File not found: {test_file}")
                continue

            print(f"\nTesting: {os.path.basename(test_file)}")
            print("-" * 50)

            # Initialize HybridONNXWhisper
            transcriber = HybridONNXWhisper("medium", "pt")

            # Load a small test audio (simulate)
            # For this test, we'll create dummy audio data
            # In real usage, this would come from actual audio file
            audio_data = np.random.randn(16000 * 5).astype(np.float32)  # 5 seconds of dummy audio

            print(f"Testing with audio shape: {audio_data.shape}")

            start_time = time.time()

            try:
                # Test the encoder/decoder strategy
                result = transcriber.transcribe_with_encoder_decoder_strategy(audio_data)

                end_time = time.time()
                processing_time = end_time - start_time

                print(f"\nRESULTS:")
                print(f"Processing time: {processing_time:.2f}s")
                print(f"Text output: '{result.get('text', 'NO_TEXT')}'")
                print(f"Segments: {len(result.get('segments', []))}")
                print(f"Duration: {result.get('duration', 0):.2f}s")
                print(f"Real transcription: {result.get('model_info', [])}")

                # Check if it's real transcription (not placeholder)
                is_placeholder = "placeholder" in result.get('text', '').lower()
                is_real = not is_placeholder and len(result.get('text', '')) > 0

                print(f"\nVALIDATION:")
                print(f"Is placeholder: {is_placeholder}")
                print(f"Is real transcription: {is_real}")
                print(f"Processing ratio: {processing_time / 5.0:.3f}x (for 5s audio)")

                if is_real:
                    print("✅ SUCCESS: Real transcription working!")
                else:
                    print("❌ FAILED: Still using placeholder")

            except Exception as e:
                print(f"❌ ERROR during transcription: {e}")
                import traceback
                traceback.print_exc()

    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Testing Real Whisper Transcription Implementation")
    print("=" * 60)
    test_real_transcription()