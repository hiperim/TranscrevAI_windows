#!/usr/bin/env python3
"""Test script to validate offline Pyannote model loading"""
import sys
sys.path.insert(0, '/app/src')

from diarization import PyannoteDiarizer

print("=" * 60)
print("Testing Pyannote Pipeline Loading (Offline Mode)")
print("=" * 60)

try:
    # Initialize diarizer without HF token (should use local cache)
    diarizer = PyannoteDiarizer(device="cpu")

    if diarizer.pipeline is not None:
        print("\n✅ SUCCESS: Pipeline loaded successfully from local cache!")
        print(f"Pipeline type: {type(diarizer.pipeline)}")
        print("\nTest PASSED: Docker image can run diarization without HF token")
        sys.exit(0)
    else:
        print("\n❌ FAILED: Pipeline is None")
        sys.exit(1)

except Exception as e:
    print(f"\n❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
