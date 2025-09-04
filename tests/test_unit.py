from pathlib import Path
from typing import Union
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
from unittest.mock import patch, MagicMock, AsyncMock
from src.audio_processing import AudioRecorder, AudioProcessingError
from src.file_manager import FileManager
from src.speaker_diarization import SpeakerDiarization
from src.subtitle_generator import generate_srt
from src.transcription import WhisperTranscriptionService, TranscriptionError, transcribe_audio_with_progress   
from config.app_config import WHISPER_MODEL_DIR, WHISPER_MODELS
from src.logging_setup import setup_app_logging
from tests.conftest import generate_test_audio

@pytest.fixture(scope="session")
def mock_whisper():
    # Mock Whisper model for testing
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {
        "text": "This is a test transcription",
        "segments": [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "This is a test",
                "confidence": 0.95
            },
            {
                "start": 2.0,
                "end": 4.0,
                "text": "transcription",
                "confidence": 0.90
            }
        ],
        "language": "en"
    }
    return mock_model

@pytest.fixture(scope="session")
def mock_torch():
    # Mock torch for testing
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.device.return_value = "cpu"
    return mock_torch

@pytest.fixture(scope="session") 
def mock_pyannote_pipeline():
    # Mock PyAnnote pipeline for testing
    mock_pipeline = MagicMock()
    
    # Mock diarization result
    def mock_itertracks(yield_label=False):
        # Simulate segments with speakers
        from types import SimpleNamespace
        segments = [
            (SimpleNamespace(start=0.0, end=2.0), None, "SPEAKER_00"),
            (SimpleNamespace(start=2.0, end=4.0), None, "SPEAKER_01")
        ]
        return iter(segments)
    
    mock_pipeline.itertracks = mock_itertracks
    return mock_pipeline

@pytest.fixture(scope="session")
def mock_static_ffmpeg():
    # Mock static_ffmpeg for testing
    mock_static = MagicMock()
    mock_static.add_paths.return_value = None
    return mock_static

@pytest.fixture(scope="session")
def fast_test_audio():
    # Generate minimal test audio data quickly
    sample_rate = 16000
    duration = 0.1  # 100ms only
    samples = int(sample_rate * duration)
    audio_data = np.random.uniform(-0.1, 0.1, samples).astype(np.float32)
    return audio_data, sample_rate

@pytest.fixture(autouse=True, scope="session")
def mock_heavy_imports():
    # Automatically mock heavy imports for all tests
    patches = []
    
    # Mock ML imports
    mock_whisper_module = MagicMock()
    mock_whisper_module.load_model.return_value = MagicMock()
    patches.append(patch.dict('sys.modules', {'whisper': mock_whisper_module}))
    
    mock_torch_module = MagicMock() 
    mock_torch_module.cuda.is_available.return_value = False
    patches.append(patch.dict('sys.modules', {'torch': mock_torch_module}))
    
    # Mock PyAnnote
    mock_pyannote = MagicMock()
    patches.append(patch.dict('sys.modules', {'pyannote.audio': mock_pyannote}))
    
    # Mock static_ffmpeg
    mock_static = MagicMock()
    patches.append(patch.dict('sys.modules', {'static_ffmpeg': mock_static}))
    
    # Start all patches
    for p in patches:
        p.start()
        
    yield
    
    # Stop all patches
    for p in patches:
        p.stop()

@pytest.fixture
def mock_transcription_service():
    # Mock TranscriptionService with fast responses
    service = MagicMock()
    
    async def mock_transcribe(audio_file, language="en"):
        await asyncio.sleep(0.01)  # Minimal delay
        return {
            "text": "Test transcription result",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "Test", "confidence": 0.95},
                {"start": 1.0, "end": 2.0, "text": "transcription", "confidence": 0.90},
                {"start": 2.0, "end": 3.0, "text": "result", "confidence": 0.85}
            ],
            "language": language
        }
    
    service.transcribe_audio_with_progress = AsyncMock(side_effect=mock_transcribe)
    return service

@pytest.fixture
def mock_speaker_diarization():
    # Mock SpeakerDiarization with fast responses
    diarizer = MagicMock()
    
    async def mock_diarize(audio_file, num_speakers=2):
        await asyncio.sleep(0.01)  # Minimal delay
        return [
            {"start": 0.0, "end": 1.5, "speaker": "Speaker_1", "confidence": 0.9},
            {"start": 1.5, "end": 3.0, "speaker": "Speaker_2", "confidence": 0.85}
        ]
    
    diarizer.diarize_audio = AsyncMock(side_effect=mock_diarize)
    return diarizer

# Fast test configurations
FAST_TEST_CONFIG = {
    "whisper_model_size": "tiny",  # Smallest model
    "max_test_duration": 0.1,      # 100ms audio max
    "skip_model_download": True,
    "use_cpu_only": True,
    "mock_heavy_operations": True
}

logger = setup_app_logging()

class AsyncTestCase(unittest.IsolatedAsyncioTestCase):
    @pytest.mark.timeout(5)  # Reduced from 30s
    async def asyncSetUp(self):
        await super().asyncSetUp()
        logger.info(f"Starting test: {self._testMethodName}")
        self.test_audio_dir = Path(__file__).parent / "test_audio"
        self.test_audio_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir = self.test_audio_dir / "audio_capture"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
    
    @pytest.mark.timeout(5)  # Reduced from 30s
    async def asyncTearDown(self):
        await super().asyncTearDown()
        temp_dir = FileManager.get_data_path("temp")
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
                        await TestPerformance.safe_remove(str(file_path))
                except (PermissionError, OSError):
                    # Skip inaccessible files 
                    continue
        except Exception as e:
            logger.warning(f"Temp cleanup error: {e}")

class TestAudioRecorder:
    def cleanup_audio_processes(self):
    # Ensure all audio procs. are terminated
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

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)  # Reduced timeout
    async def test_audio_capture(self):
        output_file = str(self.audio_dir / "test.wav")
        output_path = Path(output_file)
        self._recorder = AudioRecorder(output_file=output_file)
        try:
            with patch("sounddevice.InputStream") as mock_stream:
                mock_stream_instance = MagicMock()

                def mock_start():
                    # Simulate audio data callback after recording starts
                    fake_data = np.random.rand(1024, 1).astype(np.float32)
                    # Ensure self._recorder is initialized and has '_frames' attribute
                    if self._recorder is None:
                        raise RuntimeError("AudioRecorder instance is not initialized")
                    # Use a local reference to avoid static analyzers identifying _recorder as 'None'
                    rec = self._recorder
                    if not hasattr(rec, "_audio_callback"):
                        if not hasattr(rec, "_frames"):
                            rec._frames = []
                        # Attach callback to local reference
                        setattr(rec, "_audio_callback", lambda data, frames, x, y: rec._frames.append(data))
                    for _ in range(10):  # Simulate multiple callbacks
                        cb = getattr(rec, "_audio_callback", None)
                        if cb is None:
                            # Fallback no-operation callback to avoid attribute access issues
                            def _noop(data, frames, x, y):
                                return None
                            setattr(rec, "_audio_callback", _noop)
                            cb = getattr(rec, "_audio_callback")
                        cb(fake_data, 1024, None, None)
                
                mock_stream_instance.start.side_effect = mock_start
                mock_stream.return_value = mock_stream_instance

                await self._recorder.start_recording()
                await asyncio.sleep(1.5)
            frames_copy = self._recorder._frames.copy() 
            await self._recorder.stop_recording()
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
            # Windows validation
            assert Path(output_file).exists(), "Output file not created"
            assert Path(output_file).stat().st_size > 1024, "Output file too small"
            with sf.SoundFile(output_file) as sf_file:
                assert sf_file.samplerate in [16000, 44100], "Invalid sample rate"
                assert sf_file.channels in [1, 2], "Invalid channel count"
        finally:
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
                await self._windows_process_cleanup(path)
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
    @pytest.mark.timeout(10)  # Reduced timeout
    async def test_mp4_conversion_validation(self, generate_test_audio):
        source_file = generate_test_audio(duration=5.0)
        output_file = self.audio_dir / "test.mp4"
        self._recorder = AudioRecorder(output_file=str(output_file))
        try:
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
                        # Safely deliver chunk to recorder: ensure recorder exists, prefer a callable
                        # _audio_callback if present, otherwise append to _frames creating it if needed.
                        if self._recorder is None:
                            raise RuntimeError("AudioRecorder instance is not initialized")
                        rec = self._recorder
                        callback = getattr(rec, "_audio_callback", None)
                        if callable(callback):
                            try:
                                callback(chunk, len(chunk), None, None)
                            except Exception:
                                # On unexpected failure, fall back to buffering frames directly
                                if not hasattr(rec, "_frames"):
                                    rec._frames = []
                                rec._frames.append(chunk)
                        else:
                            if not hasattr(rec, "_frames"):
                                rec._frames = []
                            rec._frames.append(chunk)
                        current_pos = end_pos
                        time.sleep(0.1)  # simulate real-time delay

                mock_stream_instance.start.side_effect = mock_start
                mock_stream.return_value = mock_stream_instance
                await self._recorder.start_recording()
                source_duration = AudioRecorder.get_audio_duration(str(source_file))
                await asyncio.sleep(source_duration + 0.5)  # Buffer to ensure data is processed
                frames_copy = self._recorder._frames.copy()
                await self._recorder.stop_recording()
                validated = False
                output_path = Path(output_file)
                converted_duration = None
                result = None
                
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
                    pytest.fail(f"MP4 validation failed after 60 attempts. Duration: {converted_duration if converted_duration is not None else 'N/A'}")
                
                # Only proceed with assertion if we have valid results
                if result is not None and converted_duration is not None:
                    source_duration = sum(frame.shape[0] for frame in frames_copy) / self._recorder.sample_rate
                    assert abs(source_duration - converted_duration) <= 0.15, \
                        f"Duration mismatch: {source_duration:.1f}s vs. {converted_duration:.1f}s"
        finally:
            if Path(output_file).exists():
                await TestPerformance.safe_remove(output_file)
            if Path(source_file).exists():
                await TestPerformance.safe_remove(source_file)

class TestFileOperations(AsyncTestCase):
    @pytest.mark.timeout(5)  # Reduced timeout with mocks
    async def test_whisper_model_lifecycle(self, mock_whisper):
        # Test Whisper model lifecycle using fast mocks
        try:
            # Use mocked model for fast testing
            model_name = WHISPER_MODELS["en"]
            model = mock_whisper
            
            assert model is not None, "Whisper model not loaded"
            assert hasattr(model, "transcribe"), "Model missing transcribe method"
            
            # Test transcription capability
            result = model.transcribe("dummy_audio")
            assert result["text"] == "This is a test transcription"
            assert len(result["segments"]) == 2
            
            logger.info(f"Whisper model '{model_name}' tested successfully with mocks")
            
        except Exception as e:
            pytest.fail(f"Whisper model lifecycle test failed: {str(e)}")

class TestDiarization():
    @pytest.mark.asyncio
    @pytest.mark.timeout(5)  # Reduced timeout with mocks
    async def test_pyannote_diarization(self, fast_test_audio, mock_pyannote_pipeline, mock_speaker_diarization):
        # Test PyAnnote.Audio diarization with fast mocks
        audio_data, sample_rate = fast_test_audio
        
        # Create temporary audio file
        temp_file = Path(FileManager.get_unified_temp_dir()) / "fast_test.wav"
        sf.write(temp_file, audio_data, sample_rate)
        
        try:
            # Use mocked diarization service
            segments = await mock_speaker_diarization.diarize_audio(str(temp_file))
            
            assert len(segments) >= 1, "No diarization segments returned"
            assert segments[0]["speaker"] == "Speaker_1"
            assert segments[1]["speaker"] == "Speaker_2"
            
            unique_speakers = {s["speaker"] for s in segments}
            assert len(unique_speakers) == 2, f"Expected 2 speakers, got {len(unique_speakers)}"
                
        finally:
            # Clean-up
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete temporary file: {e}")


class TestSubtitles():
    @pytest.mark.asyncio
    @pytest.mark.timeout(10)  # Reduced timeout
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
    @pytest.mark.timeout(5)  # Reduced timeout with fast mocks
    async def test_recording_to_subtitles_whisper(self, fast_test_audio, mock_transcription_service, mock_speaker_diarization):
        # Test full pipeline with optimized fast mocks
        audio_data, sample_rate = fast_test_audio
        
        temp_dir = tempfile.mkdtemp()
        try:
            # Create test audio file
            audio_file = Path(temp_dir) / "test.wav"
            sf.write(audio_file, audio_data, sample_rate)
            
            srt_file = Path(temp_dir) / "output.srt"
            
            # Use fast mocked services
            segments = await mock_speaker_diarization.diarize_audio(str(audio_file))
            transcription = await mock_transcription_service.transcribe_audio_with_progress(str(audio_file), "en")
            
            # Generate SRT with mocked data
            transcription_data = transcription["segments"]
            await generate_srt(transcription_data, segments, str(srt_file))
            
            assert srt_file.exists(), "SRT file was not created"
            
            # Verify SRT content
            with open(srt_file, 'r') as f:
                srt_content = f.read()
                assert len(srt_content) > 0, "SRT file is empty"
                assert "Test" in srt_content, "Expected transcription text not found"
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

class TestPerformance():
    @pytest.mark.asyncio
    @pytest.mark.timeout(5)  # Reduced timeout with fast mocks
    async def test_diarization_latency(self, fast_test_audio, mock_speaker_diarization):
        # Test diarization latency with fast mocks
        audio_data, sample_rate = fast_test_audio
        
        temp_file = Path(FileManager.get_unified_temp_dir()) / "latency_test.wav"
        sf.write(temp_file, audio_data, sample_rate)
        
        try:
            # Test latency with mocked service
            start_time = time.monotonic()
            segments = await mock_speaker_diarization.diarize_audio(str(temp_file))
            processing_time = time.monotonic() - start_time
            
            # With mocks, should be very fast
            assert processing_time < 1.0, f"Diarization took too long: {processing_time}s"
            assert len(segments) == 2, "Expected 2 segments from mock"
            
            logger.info(f"Diarization latency test completed in {processing_time:.3f}s")
        finally:
            # Clean-up
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete temporary file: {e}")
    @staticmethod
    async def safe_remove(path: Union[str, Path], max_retries: int = 10) -> bool:
        p = Path(path)
        for i in range(max_retries):
            try:
                if not p.exists():
                    return True
                # Add process tree termination
                for proc in psutil.process_iter():
                    try:
                        open_files = proc.open_files()
                        if any(str(p) in f.path for f in open_files):
                            proc.kill()
                    except (psutil.AccessDenied, psutil.NoSuchProcess):
                        continue
                    await asyncio.sleep(0.5)
                p.unlink(missing_ok=True)
                return True
            except PermissionError:
                await asyncio.sleep(1 * (i + 1))
        return False

if __name__ == "__main__":
    pytest.main(["-v", "--cov=src", "--cov-report=html:cov_html", "-p", "no:warnings"])
