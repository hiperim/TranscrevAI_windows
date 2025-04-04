import pytest
import os
import numpy as np
import sys
import soundfile as sf
import tempfile
from pathlib import Path
import shutil
import time
import stat
import subprocess
from ctypes import windll
import psutil
from unittest.mock import Mock, patch
from src.logging_setup import setup_app_logging
from src.file_manager import FileManager

logger = setup_app_logging()

@pytest.fixture
def generate_test_audio():
    def _generate(duration=5.0, speakers=2, sample_rate=16000):
        samples = int(duration * sample_rate)
        data = np.zeros(samples, dtype=np.float32)
        for i in range(speakers):
            freq = 440 * (i + 1)
            t = np.linspace(0, duration, samples)
            data += 0.5 * np.sin(2 * np.pi * freq * t)
        data = data / np.max(np.abs(data))
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp:
            temp_file = temp.name
            sf.write(temp_file, data, sample_rate)
            logger.debug(f"Generated test audio: {temp_file}")
            return temp_file
        for attempt in range(10):
            try:
                with open(temp_file, "rb") as f:
                    f.read(1024)
                break
            except PermissionError:
                    time.sleep(0.5 * (attempt + 1))     
    return _generate

@pytest.fixture
def temp_path(tmp_path_factory):
    """Windows temporary directory fixture with enhanced cleaning"""
    base_temp = tmp_path_factory.getbasetemp()
    test_temp = base_temp / "test_audio"
        # Remove existing directory first with Windows handling
    if test_temp.exists():
        if sys.platform == "win32":
            try:
                # Terminate all processes that might have handles to our test directory
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        # Check if process has any relation to our test directory
                        proc_info = proc.as_dict(attrs=['pid', 'name', 'cmdline', 'open_files'])
                        proc_name = proc_info['name'].lower() if proc_info['name'] else ""
                        audio_related = any(x in proc_name for x in ['ffmpeg', 'ffprobe', 'python', 'soundfile', 'audio'])
                        if audio_related:
                            try:
                                open_files = proc.open_files()
                                if any(str(test_temp) in f.path for f in open_files):
                                    logger.info(f"Terminating process {proc.pid} ({proc_name}) holding test files")
                                    proc.kill()
                            except (psutil.AccessDenied, psutil.NoSuchProcess):
                                pass
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                # Use robust Windows API for permission reset
                try:
                    import win32security
                    import win32con
                    import win32file
                    # Set directory as not read-only
                    for root, dirs, files in os.walk(str(test_temp)):
                        for dir in dirs:
                            dir_path = os.path.join(root, dir)
                            try:
                                # Remove read-only attribute
                                win32file.SetFileAttributes(dir_path, win32file.FILE_ATTRIBUTE_NORMAL)
                            except Exception:
                                pass
                        for file in files:
                            file_path = os.path.join(root, file)
                            try:
                                # Remove read-only attribute
                                win32file.SetFileAttributes(file_path, win32file.FILE_ATTRIBUTE_NORMAL)
                            except Exception:
                                pass
                    # Give everyone full control of the directory
                    everyone_sid = win32security.ConvertStringSidToSid("S-1-1-0")  # Everyone SID
                    security_descriptor = win32security.SECURITY_DESCRIPTOR()
                    dacl = win32security.ACL()
                    dacl.AddAccessAllowedAce(win32security.ACL_REVISION, win32con.GENERIC_ALL, everyone_sid)
                    security_descriptor.SetSecurityDescriptorDacl(1, dacl, 0)
                    # Apply to directory and all subdirectories
                    win32security.SetFileSecurity(str(test_temp), win32security.DACL_SECURITY_INFORMATION, security_descriptor)
                except ImportError:
                    logger.warning("Win32 security modules not available")
                # Wait for windows to release handles
                time.sleep(2.0)
                # Force flush windows filesystem cache
                try:
                    subprocess.run(["cmd", "/c", "echo > NUL"], shell=True, check=False)
                except Exception:
                    pass
            except Exception as e:
                logger.warning(f"Windows directory prep failed: {e}")
        # Multiple attempts for directory removal
        for attempt in range(5):
            try:
                shutil.rmtree(test_temp, ignore_errors=True)
                break
            except Exception as e:
                wait_time = 0.5 * (2 ** attempt)  # Exponential backoff
                logger.debug(f"Directory removal attempt {attempt+1} failed: {e}, waiting {wait_time}s")
                time.sleep(wait_time)
    # Ensure directory exists
    test_temp.mkdir(parents=True, exist_ok=True)
    # Add special handling for windows permissions for new directory
    if sys.platform == "win32":
        try:
            # Ensure exists directory with known good permissions
            os.chmod(test_temp, 0o777)
        except Exception as e:
            logger.debug(f"Permission setting error: {e}")
    yield test_temp
    # Enhanced cleanup after yield, ensure all audio procecess are terminated
    try:
        subprocess.run(["taskkill", "/F", "/IM", "ffmpeg.exe", "/IM", "ffprobe.exe", "/T"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except Exception:
        pass
    # Wait for windows to release handles
    time.sleep(2.0)
    # Multiple strategies for cleanup
    try:
        # Force directory handle closure with robocopy empty trick
        temp_empty = tmp_path_factory.getbasetemp() / f"empty_{int(time.time())}"
        temp_empty.mkdir(exist_ok=True)
        try:
            # Use robocopy to mirror an empty directory (deletes content)
            subprocess.run(["robocopy", str(temp_empty), str(test_temp), "/MIR", "/NFL", "/NDL", "/NJH", "/NJS", "/NC", "/NS", "/NP"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        except Exception:
            pass
        # Remove empty directory
        try:
            temp_empty.rmdir()
        except Exception:
            pass
    except Exception as e:
        logger.debug(f"Advanced cleanup error: {e}")
    # Recursive error handler on final cleanup
    def on_rm_error(func, path, exc_info):
        # Make read-only files writable and retry
        os.chmod(path, stat.S_IWRITE)
        try:
            func(path)
        except Exception:
            # Last resort: try to use the shell to delete
            try:
                subprocess.run(["cmd", "/c", f"rd /s /q \"{path}\""], shell=True, check=False)
            except Exception:
                logger.debug(f"Failed to remove {path}")
    # Final cleanup
    try:
        shutil.rmtree(test_temp, onerror=on_rm_error)
    except Exception as e:
        logger.debug(f"Final cleanup failed: {e}")

def pytest_configure(config):
    config.addinivalue_line("markers", "asyncio: mark test as async")
    config.addinivalue_line("markers", "timeout: mark test to timeout")
    config.option.asyncio_mode = "strict"