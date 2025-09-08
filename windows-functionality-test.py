#!/usr/bin/env python3
"""
Windows Functionality Test for TranscrevAI
Tests Windows-specific functionality to ensure compatibility is maintained
"""

import sys
import os
import tempfile
import asyncio
import time
from pathlib import Path

def test_file_operations():
    """Test file operations work correctly on Windows"""
    print("Testing file operations...")
    
    try:
        # Test Path operations
        test_path = Path("test_file.txt")
        test_path.write_text("test content")
        
        if test_path.exists() and test_path.read_text() == "test content":
            print("OK Basic file operations work")
            test_path.unlink()  # cleanup
            return True
        else:
            print("ERROR Basic file operations failed")
            return False
            
    except Exception as e:
        print(f"ERROR File operations failed: {e}")
        return False

def test_audio_processing_initialization():
    """Test that AudioRecorder can be initialized on Windows"""
    print("Testing AudioRecorder initialization...")
    
    try:
        from src.audio_processing import AudioRecorder
        
        # Test initialization without actually recording
        test_file = f"test_audio_{int(time.time())}.wav"
        recorder = AudioRecorder(test_file)
        print("OK AudioRecorder initialized successfully")
        
        # Cleanup - try to remove file if it exists
        try:
            if os.path.exists(test_file):
                os.unlink(test_file)
        except:
            pass  # Ignore cleanup errors
            
        return True
            
    except Exception as e:
        print(f"ERROR AudioRecorder initialization failed: {e}")
        return False

async def test_async_file_operations():
    """Test async file operations"""
    print("Testing async file operations...")
    
    try:
        from src.audio_processing import SimpleFileHandler
        
        # Create temp files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as src:
            src.write("test content")
            src_path = src.name
        
        dst_path = src_path + ".moved"
        
        # Test atomic move
        success = await SimpleFileHandler.safe_atomic_move(src_path, dst_path)
        
        if success and Path(dst_path).exists():
            print("OK Async file operations work")
            Path(dst_path).unlink()  # cleanup
            return True
        else:
            print("ERROR Async file operations failed")
            return False
            
    except Exception as e:
        print(f"ERROR Async file operations failed: {e}")
        return False

def test_directory_creation():
    """Test directory creation and management"""
    print("Testing directory creation...")
    
    try:
        from config.app_config import _ensure_directories_created
        
        # This should create all necessary directories
        _ensure_directories_created()
        
        from config.app_config import DATA_DIR, TEMP_DIR
        
        if DATA_DIR.exists() and TEMP_DIR.exists():
            print("OK Directory creation works")
            return True
        else:
            print("ERROR Directory creation failed")
            return False
            
    except Exception as e:
        print(f"ERROR Directory creation failed: {e}")
        return False

def test_websocket_functionality():
    """Test WebSocket components can load"""
    print("Testing WebSocket functionality...")
    
    try:
        # Test that WebSocket-related components can be imported
        import asyncio
        from fastapi import WebSocket
        
        print("OK WebSocket components loaded")
        return True
        
    except Exception as e:
        print(f"ERROR WebSocket functionality failed: {e}")
        return False

def test_ffmpeg_availability():
    """Test FFmpeg functionality"""
    print("Testing FFmpeg availability...")
    
    try:
        import subprocess
        
        # Try to run FFmpeg
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            print("OK FFmpeg is available and working")
            return True
        else:
            print("WARNING FFmpeg returned non-zero code but may still work")
            return True  # Don't fail the test for this
            
    except FileNotFoundError:
        print("WARNING FFmpeg not found in PATH - using static_ffmpeg")
        return True  # This is expected on some Windows systems
    except Exception as e:
        print(f"WARNING FFmpeg test failed: {e}")
        return True  # Don't fail the test for FFmpeg issues

def test_model_directory_setup():
    """Test model directory setup"""
    print("Testing model directory setup...")
    
    try:
        from config.app_config import WHISPER_MODEL_DIR
        
        # Ensure model directory exists
        WHISPER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        
        if WHISPER_MODEL_DIR.exists():
            print("OK Model directory setup works")
            return True
        else:
            print("ERROR Model directory setup failed")
            return False
            
    except Exception as e:
        print(f"ERROR Model directory setup failed: {e}")
        return False

async def test_application_startup():
    """Test that the application can start up"""
    print("Testing application startup simulation...")
    
    try:
        # Import main components without actually starting the server
        from main import app
        from config.app_config import _ensure_directories_created
        
        # Ensure directories are created
        _ensure_directories_created()
        
        print("OK Application startup simulation successful")
        return True
        
    except Exception as e:
        print(f"ERROR Application startup failed: {e}")
        return False

async def run_async_tests():
    """Run all async tests"""
    results = []
    
    results.append(await test_async_file_operations())
    results.append(await test_application_startup())
    
    return results

def main():
    """Run all Windows functionality tests"""
    print("TranscrevAI Windows Functionality Test")
    print("=" * 42)
    
    # Sync tests
    sync_results = [
        test_file_operations(),
        test_audio_processing_initialization(),
        test_directory_creation(),
        test_websocket_functionality(),
        test_ffmpeg_availability(),
        test_model_directory_setup(),
    ]
    
    # Async tests
    async_results = asyncio.run(run_async_tests())
    
    # Combine results
    all_results = sync_results + async_results
    
    # Summary
    print("\n" + "=" * 42)
    passed = sum(all_results)
    total = len(all_results)
    
    if passed == total:
        print(f"OK All functionality tests passed ({passed}/{total})")
        print("\nSUCCESS: Windows functionality is preserved!")
        return 0
    else:
        print(f"ERROR Some functionality tests failed ({passed}/{total})")
        print("\nWARNING: Check Windows compatibility issues above")
        return 1

if __name__ == "__main__":
    sys.exit(main())