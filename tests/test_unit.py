from pathlib import Path
import unittest
import os
import sys
import tempfile
import pytest
import asyncio
import aiofiles
import subprocess
import time
import soundfile as sf
import shutil
import logging
import numpy as np
import psutil
from ctypes import windll
from unittest.mock import Mock, patch, MagicMock
from src.audio_processing import AudioRecorder, AudioProcessingError
from src.file_manager import FileManager, ANDROID_ENABLED
from src.speaker_diarization import SpeakerDiarization
from src.subtitle_generator import generate_srt
from src.transcription import transcribe_audio_with_progress
from config.app_config import MODEL_DIR
from src.logging_setup import setup_app_logging
from tests.conftest import generate_test_audio

logger = setup_app_logging()

@pytest.mark.timeout(30)
def pytest_configure(config):
    config.addinivalue_line("markers", "android: mark test as Android-only")

def pytest_runtest_setup(item):
    android_marker = item.get_closest_marker("android")
    if android_marker and not FileManager.is_mobile():
        pytest.skip("Requires Android environment")

class AsyncTestCase(unittest.IsolatedAsyncioTestCase):
    @pytest.mark.timeout(30)
    async def asyncSetUp(self):
        await super().asyncSetUp()
        logger.info(f"Starting test: {self._testMethodName}")
        self.test_audio_dir = Path(__file__).parent / "test_audio"
        self.test_audio_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir = self.test_audio_dir / "audio_capture"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
    
    @pytest.mark.timeout(30)
    async def asyncTearDown(self):
        await super().asyncTearDown()
        temp_dir = FileManager.get_data_path("temp")
        if sys.platform == "win32":
            for proc in psutil.process_iter(["pid", "name"]):
                try:
                    if "ffmpeg" in proc.info["name"].lower():
                        proc.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            await asyncio.sleep(1)
        cutoff = time.time() - 300 # only remove files older than current test   # 5min. safety window
        try:
            for f in os.listdir(temp_dir):
                file_path = Path(temp_dir) / f
                try:
                    if file_path.stat().st_mtime < cutoff:
                        await TestPerformance.safe_remove(file_path)
                except (PermissionError, OSError):
                    # Skip inaccessible files 
                    continue
        except Exception as e:
            logger.warning(f"Temp cleanup error: {e}")

class TestAudioRecorder:
    def cleanup_audio_processes(self):
    # Ensure all audio procs. are terminated
        if sys.platform == "win32":
            try:
                # Kill ffmpeg procs.
                for proc in psutil.process_iter(["pid", "name"]):
                    try:
                        if any(x in proc.name().lower() for x in ["ffmpeg", "ffprobe"]):
                            logger.debug(f"Terminating audio process {proc.pid}")
                            proc.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
            except Exception as e:
                logger.debug(f"Audio process cleanup error: {e}")

    @pytest.fixture(scope="function", autouse=True)
    def setup_method(self, temp_path):
        self.audio_dir = temp_path / "audio"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self._recorder = None
        with patch("src.file_manager.FileManager.get_data_path") as mock_path:
            mock_path.return_value = str(self.audio_dir)
            yield
        if self._recorder and self._recorder.is_recording:
            try:
                asyncio.run(self._recorder.stop_recording())
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
        shutil.rmtree(self.audio_dir, ignore_errors=True)

    @pytest.fixture
    def mock_android(self):
        with patch("src.file_manager.FileManager.is_mobile", return_value=True), \
             patch("src.file_manager.FileManager.get_data_path", return_value="/mock/path"):
            yield Mock()

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_audio_capture(self):
        output_file = str(self.audio_dir / "test.wav")
        output_path = Path(output_file)
        self._recorder = AudioRecorder(output_file=output_file)
        try:
            if FileManager.is_mobile() and ANDROID_ENABLED:
                with patch("jnius.autoclass") as mock_autoclass:
                    mock_media = Mock()
                    mock_media.AudioSource.MIC = 1
                    mock_autoclass.return_value = mock_media
                    await self._recorder.start_recording()
            else:
                with patch("sounddevice.InputStream") as mock_stream:
                    mock_stream_instance = MagicMock()

                    def mock_start():
                        # Simulate audio data callback after recording starts
                        fake_data = np.random.rand(1024, 1).astype(np.float32)
                        for _ in range(10):  # Simulate multiple callbacks
                            self._recorder._audio_callback(fake_data, 1024, None, None)
                    mock_stream_instance.start.side_effect = mock_start
                    mock_stream.return_value = mock_stream_instance
                    await self._recorder.start_recording()
                    await asyncio.sleep(1.5)
            frames_copy = self._recorder._frames.copy() 
            await self._recorder.stop_recording()
            if sys.platform == "win32":
                validated = False
                output_path = Path(output_file)
                for _ in range(40):  # 0.5s delays = 20s total
                    try:
                        if output_path.exists():
                            # FFprobe-based validation for audio capture
                            cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(output_path)]
                            result = subprocess.run(cmd, capture_output=True, text=True)
                            if result.returncode == 0:
                                converted_duration = float(result.stdout.strip())
                                source_duration = sum(frame.shape[0] for frame in frames_copy) / self._recorder.sample_rate
                                if abs(source_duration - converted_duration) <= 0.15:
                                    validated = True
                                    break
                                else:
                                    logger.warning(f"Duration mismatch: {source_duration:.2f} vs {converted_duration:.2f}")
                            else:
                                logger.warning(f"FFprobe error: {result.stderr[:200]}")
                        await asyncio.sleep(0.5)
                    except (Exception, PermissionError, OSError) as e:
                        logger.warning(f".mp4 validation attempt {_} failed: {str(e)}")
                        subprocess.run(["powershell", f"Get-Process *ffmpeg*,*ffprobe*" 
                                                      f"| Where-Object {{$_.Path -like '*{output_path.name}*'}} "
                                                      f"| Stop-Process -Force -ErrorAction SilentlyContinue"], shell=True, check=False)
                if not validated:
                    pytest.fail(f".mp4 validation failed after 40 attempts. Final state: Exists={output_path.exists()}, Size={output_path.stat().st_size if output_path.exists() else 0}")
            assert Path(output_file).exists(), "Output file not created"
            assert Path(output_file).stat().st_size > 1024, "Output file too small"
            with sf.SoundFile(output_file) as sf_file:
                assert sf_file.samplerate in [16000, 44100], "Invalid sample rate"
                assert sf_file.channels in [1, 2], "Invalid channel count"
        except RuntimeError as e:
            if "Android components unavailable" in str(e):
                pytest.skip("Android test requires mobile environment")
            raise
        finally:
            if sys.platform == "win32":
                await self._windows_process_cleanup(Path(output_file))
            await asyncio.sleep(0.5)  # Cleanup delay

    async def _windows_file_validation(self, path):
        # Windows-specific file checks with handle verification
        for _ in range(10):
            if not Path(path).exists():
                logger.warning(f"File {path} no longer exists")
                return
            try:
                with open(path, 'rb') as f:
                    f.read(1024)
                assert Path(path).stat().st_size > 1024
                return
            except FileNotFoundError:
                await asyncio.sleep(0.3)
            except PermissionError:
                await self._windows_process_cleanup()
                await asyncio.sleep(0.5)
        pytest.fail("File validation failed after 10 attempts")

    async def _windows_process_cleanup(self, output_path):
        # Enhanced Windows cleanup
        if output_path is None:
            logger.warning("No output path provided for cleanup")
            return
        if not isinstance(output_path, Path):
            try: 
                output_path = Path(str(output_path))
            except Exception as e:
                logger.error(f"Failed to convert output path: {e}")
                return
        try:
            subprocess.run(["powershell", f"Get-Process *ffmpeg*,*ffprobe* | " +
                                          f"Where-Object {{(Get-Process $_.Id -ErrorAction SilentlyContinue).Modules.FileName -match '{output_path.name}' -or " +
                                          f"  $_.MainWindowTitle -match '{output_path.name}'}} | " +
                                          f"Stop-Process -Force -ErrorAction SilentlyContinue"], shell=True, check=False)
        except Exception as e:
            logger.debug(f"Process cleanup error: {e}")
            
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_mp4_conversion_validation(self, generate_test_audio):
        source_file = generate_test_audio(duration=5.0)
        output_file = self.audio_dir / "test.mp4"
        self._recorder = AudioRecorder(output_file=str(output_file))
        try:
            if sys.platform == "win32":
                if not os.access(str(source_file), os.R_OK):
                    pytest.skip("Windows file access issues")
            for _ in range(15):
                if Path(source_file).exists():
                    try:
                        with open(source_file, "rb") as f:
                            f.read(1024)
                        break
                    except PermissionError:
                        await asyncio.sleep(0.5)
            else:
                pytest.fail("Source file never appeared after 15 attempts")
            test_data, sr = sf.read(source_file)  # read source audio data
            if len(test_data.shape) == 1:
                test_data = test_data.reshape(-1, 1)
            test_data = test_data.astype(np.float32)
            with patch("sounddevice.InputStream") as mock_stream:
                mock_stream_instance = MagicMock()  
                def mock_start():
                    frame_length = int(sr * 0.1)
                    total_frames = len(test_data)   
                    current_pos = 0
                    while current_pos < total_frames:
                        end_pos = min(current_pos + frame_length, total_frames)
                        chunk = test_data[current_pos:end_pos]
                        self._recorder._audio_callback(chunk, len(chunk), None, None)
                        current_pos = end_pos
                        time.sleep(0.1)  # simulate real-time delay
                mock_stream_instance.start.side_effect = mock_start
                mock_stream.return_value = mock_stream_instance
                await self._recorder.start_recording()
                source_duration = AudioRecorder.get_audio_duration(str(source_file))
                await asyncio.sleep(source_duration + 0.5)  # Buffer to ensure data is processed
                frames_copy = self._recorder._frames.copy()
                await self._recorder.stop_recording()
                if sys.platform == "win32":
                    validated = False
                    output_path = Path(output_file)
                    for _ in range(60):  # 0.5s delays: 30s total
                        try:
                            if output_path.exists():
                                # FFprobe-based validation for .mp4
                                cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(output_path)]
                                result = subprocess.run(cmd, capture_output=True, text=True)
                                if result.returncode == 0:
                                    converted_duration = float(result.stdout.strip())
                                    source_duration = sum(frame.shape[0] for frame in frames_copy) / self._recorder.sample_rate
                                    if abs(source_duration - converted_duration) <= 0.15:
                                        validated = True
                                        break
                                else:
                                    logger.warning(f"FFprobe MP4 error: {result.stderr[:200]}")
                            await asyncio.sleep(0.25)
                        except Exception as e:
                            logger.warning(f".mp4 validation attempt {_} failed: {str(e)}")
                            # Before:
                            # subprocess.run(["powershell", "Get-Process *ffmpeg* | Stop-Process -Force"], shell=True, check=False)
                            subprocess.run(["powershell", f"Get-Process *ffmpeg* | Where-Object {{$_.Path -like '*{output_path.name}*'}} | Stop-Process -Force"], shell=True, check=False)
                    if not validated:
                        pytest.fail(f"MP4 validation failed after 60 attempts. Duration: {converted_duration if 'converted_duration' in locals() else 'N/A'}")
            converted_duration = float(result.stdout.strip())
            source_duration = sum(frame.shape[0] for frame in frames_copy) / self._recorder.sample_rate
            assert abs(source_duration - converted_duration) <= 0.15, \
                f"Duration mismatch: {source_duration:.1f}s vs. {converted_duration:.1f}s"
        finally:
            if Path(output_file).exists():
                await TestPerformance.safe_remove(output_file)
            if Path(source_file).exists():
                await TestPerformance.safe_remove(source_file)

class TestFileOperations(AsyncTestCase):
    @pytest.mark.timeout(30)
    async def test_model_lifecycle(self):
        test_url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
        try:
            # Removes existing model directory
            model_dir = os.path.join(MODEL_DIR, "en")
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir, ignore_errors=True)
                logger.info(f"Removed existing EN model directory for clean test: {model_dir}")
            model_path = await FileManager.download_and_extract_model(test_url, "en", MODEL_DIR)
            assert os.path.isdir(model_path), "Model directory does not exist"
            required_files = ["am/final.mdl", "conf/model.conf", "graph/phones/word_boundary.int", "graph/Gr.fst", "graph/HCLr.fst", "ivector/final.ie"]
            missing_files = [] 
            for file in required_files:
                full_path = os.path.join(model_path, file)
                if not os.path.exists(full_path):
                    missing_files.append(file)
                    logger.error(f"Missing required file: {full_path}")
            assert len(missing_files) == 0, f"Missing files: {missing_files}"
        except Exception as e:
            pytest.fail(f"Model lifecycle test failed: {str(e)}")

class TestDiarization():
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_speaker_identification(self, generate_test_audio):
        test_file = generate_test_audio(duration=10.0, speakers=2)
        try:
            with patch("src.speaker_diarization.SpeakerDiarization._diarize") as mock_diarize:
                mock_diarize.return_value = mock_diarize.return_value = [{"start": 0.0, "end": 1.0, "speaker": "Speaker_1"}, {"start": 1.0, "end": 2.0, "speaker": "Speaker_2"}]
                diarizer = SpeakerDiarization()
                segments = await diarizer.diarize_audio(str(test_file))
                unique_speakers = {s["speaker"] for s in segments}
                assert 1 <= len(unique_speakers) <= 2
        finally:
            # Clean-up
            try:
                if os.path.exists(test_file):
                    os.unlink(test_file)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file: {e}")


class TestSubtitles():
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_srt_integrity(self):
        temp_dir = FileManager.get_data_path("temp/test_srt")
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(temp_dir) / "test.srt"
        try:
            test_segments = [{"start": 0.0, "end": 1.0, "speaker": "Speaker_1"}, {"start": 1.5, "end": 2.5, "speaker": "Speaker_2"}]
            test_transcription = [{"start": 0.2, "end": 0.8, "text": "Test_1"}, {"start": 1.6, "end": 2.2, "text": "Test_2"}]
            await generate_srt(test_transcription, test_segments, str(output_path))
            assert output_path.exists()
            async with aiofiles.open(output_path, "r") as f:
                content = await f.read()
                assert "Speaker_1" in content
                assert "Test_1" in content
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

class TestFullPipeline():
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    @patch("src.transcription.Model")
    @patch("src.transcription.transcribe_audio_with_progress")
    async def test_recording_to_subtitles(self, mock_transcribe, mock_model, generate_test_audio): 

        async def mock_generator(audio_file, model_path, language_code):
            yield 100, [{"text": "Test transcription", "start": 0.0, "end": 2.0}]
        
        # Set the mock to return async mock_generator()
        mock_transcribe.side_effect = mock_generator
        temp_dir = tempfile.mkdtemp()
        try:
            audio_file = generate_test_audio(duration=2.0, speakers=2)
            srt_file = Path(temp_dir) / "output.srt"
            with patch("src.speaker_diarization.SpeakerDiarization.diarize_audio") as mock_diarize:
                mock_diarize.return_value = [{"start": 0.0, "end": 1.0, "speaker": "Speaker_1"},
                                             {"start": 1.0, "end": 2.0, "speaker": "Speaker_2"}]
                diarizer = SpeakerDiarization()
                segments = await diarizer.diarize_audio(str(audio_file))
                transcription = []
                # This will call func. mock_generator
                async for _, result in mock_transcribe(str(audio_file), "mock_model_path", "en"):
                    transcription.extend(result)
                await generate_srt(transcription, segments, str(srt_file))
                assert srt_file.exists()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

class TestPerformance():
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_diarization_latency(self, generate_test_audio):
        test_file = generate_test_audio(duration=10, speakers=2)
        try:
            with patch("src.speaker_diarization.SpeakerDiarization._diarize") as mock_diarize:
                mock_diarize.return_value = [{"start": 0, "end": 5, "speaker": "Speaker_1"}]
                diarizer = SpeakerDiarization()
                start_time = time.monotonic()
                await diarizer.diarize_audio(str(test_file))
                processing_time = time.monotonic() - start_time
                assert processing_time < 15.0
        finally:
            # Clean-up
            try:
                if os.path.exists(test_file):
                    os.unlink(test_file)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file: {e}")

    async def safe_remove(path: str, max_retries: int = 10) -> bool:
        path = Path(path)
        for i in range(max_retries):
            try:
                if sys.platform == "win32": 
                    if not path.exists():
                        return True
                    # add process tree termination
                    for proc in psutil.process_iter():
                        try:
                            open_files = proc.open_files()
                            if any(str(path) in f.path for f in open_files):
                                proc.kill()
                        except (psutil.AccessDenied, psutil.NoSuchProcess):
                            continue
                    await asyncio.sleep(0.5)
                path.unlink(missing_ok=True)
                return True
            except PermissionError:
                await asyncio.sleep(1 * (i + 1))
        return False

if __name__ == "__main__":
    pytest.main(["-v", "--cov=src", "--cov-report=html:cov_html", "-p", "no:warnings"])
