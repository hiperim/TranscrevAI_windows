#!/usr/bin/env python3
"""
Docker Compatibility Test Script for TranscrevAI
Tests that the application works correctly after Docker modifications
"""

import sys
import importlib
import traceback

def test_import(module_name, friendly_name):
    """Test importing a module"""
    try:
        importlib.import_module(module_name)
        print(f"OK {friendly_name}")
        return True
    except Exception as e:
        print(f"ERROR {friendly_name}: {e}")
        return False

def test_config():
    """Test configuration loading"""
    try:
        from config.app_config import DATA_DIR, WHISPER_MODEL_DIR, TEMP_DIR
        print(f"OK Configuration loaded")
        print(f"  - DATA_DIR: {DATA_DIR}")
        print(f"  - WHISPER_MODEL_DIR: {WHISPER_MODEL_DIR}")
        print(f"  - TEMP_DIR: {TEMP_DIR}")
        return True
    except Exception as e:
        print(f"ERROR Configuration failed: {e}")
        return False

def test_fastapi():
    """Test FastAPI application"""
    try:
        from main import app
        print("OK FastAPI application loaded")
        return True
    except Exception as e:
        print(f"ERROR FastAPI application failed: {e}")
        return False

def main():
    """Run all tests"""
    print("TranscrevAI Docker Compatibility Test")
    print("=" * 40)
    
    tests = [
        ("src.audio_processing", "Audio Processing"),
        ("src.speaker_diarization", "Speaker Diarization"),
        ("src.transcription", "Transcription"),
        ("src.file_manager", "File Manager"),
        ("src.subtitle_generator", "Subtitle Generator"),
    ]
    
    results = []
    
    # Test core modules
    for module, name in tests:
        results.append(test_import(module, name))
    
    # Test configuration
    results.append(test_config())
    
    # Test FastAPI
    results.append(test_fastapi())
    
    # Summary
    print("\n" + "=" * 40)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"OK All tests passed ({passed}/{total})")
        print("\nSUCCESS: TranscrevAI is ready for Docker!")
        return 0
    else:
        print(f"ERROR Some tests failed ({passed}/{total})")
        print("\nWARNING: Please check the errors above")
        return 1

if __name__ == "__main__":
    sys.exit(main())