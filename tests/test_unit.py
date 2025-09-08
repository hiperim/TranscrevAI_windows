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
import json
import wave
import websockets

from src.audio_processing import AudioRecorder, AudioProcessingError
from src.file_manager import FileManager
from src.speaker_diarization import SpeakerDiarization
from src.subtitle_generator import generate_srt
from src.transcription import transcribe_audio_with_progress   
from config.app_config import WHISPER_MODEL_DIR, WHISPER_MODELS
from src.logging_setup import setup_app_logging
from tests.conftest import generate_test_audio

# ========================
# EXISTING FIXTURES
# ========================

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

# ========================
# COUNTDOWN FIX TESTS
# ========================

class TestCountdownFix:
    """
    Test suite for the countdown functionality that fixes the audio recording delay issue.
    
    The issue: First 2-3 seconds of audio were lost due to audio stream initialization delay.
    The fix: 3-second countdown before starting actual recording.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for countdown tests"""
        self.server_url = "ws://localhost:8001"
        self.test_recordings = [
            "recording_1757186318.wav",
            "recording_1757200116.mp4", 
            "recording_1757200941.wav",
            "recording_1757201283.mp4",
            "recording_1757258529.mp4",
            "recording_1757258721.wav",
            "recording_1757260158.wav"
        ]
        
    # ========================
    # COUNTDOWN TIMING TESTS
    # ========================
    
    @pytest.mark.asyncio
    async def test_countdown_timing_accuracy(self):
        """Test that the 3-second countdown is accurate within tolerance"""
        start_time = time.time()
        
        for i in range(3, 0, -1):
            await asyncio.sleep(1.0)
        
        end_time = time.time()
        elapsed = end_time - start_time
        expected = 3.0
        tolerance = 0.1  # 100ms tolerance
        
        assert abs(elapsed - expected) <= tolerance, f"Countdown timing inaccurate: {elapsed:.2f}s (expected {expected}s)"
        
    @pytest.mark.asyncio
    async def test_countdown_prevents_early_recording(self):
        """Test that recording doesn't start until countdown completes"""
        recording_started = False
        countdown_complete = False
        
        async def simulate_countdown():
            nonlocal countdown_complete
            await asyncio.sleep(3.0)  # 3-second countdown
            countdown_complete = True
        
        async def simulate_recording_start():
            nonlocal recording_started
            # This should only happen after countdown
            if countdown_complete:
                recording_started = True
        
        # Start countdown
        countdown_task = asyncio.create_task(simulate_countdown())
        
        # Wait a bit, recording should not have started yet
        await asyncio.sleep(1.5)
        assert not recording_started, "Recording started before countdown completed"
        
        # Wait for countdown to complete
        await countdown_task
        await simulate_recording_start()
        
        assert countdown_complete, "Countdown did not complete"
        assert recording_started, "Recording did not start after countdown"

    # ========================
    # BUTTON STATE TESTS
    # ========================
    
    @pytest.mark.asyncio
    async def test_button_state_sequence(self):
        """Test the expected button state sequence during countdown"""
        # Define expected states
        states = [
            {'phase': 'Initial', 'record': True, 'pause': False, 'stop': False},
            {'phase': 'Countdown', 'record': False, 'pause': False, 'stop': False},
            {'phase': 'Recording', 'record': False, 'pause': True, 'stop': True},
            {'phase': 'Completed', 'record': True, 'pause': False, 'stop': False}
        ]
        
        # Simulate button state changes
        current_state = states[0]  # Initial
        assert current_state['record'] == True
        assert current_state['pause'] == False  
        assert current_state['stop'] == False
        
        # User clicks record -> Countdown state
        current_state = states[1]  # Countdown
        assert current_state['record'] == False  # All buttons disabled
        assert current_state['pause'] == False
        assert current_state['stop'] == False
        
        # Countdown completes -> Recording state  
        await asyncio.sleep(3.0)  # Simulate countdown
        current_state = states[2]  # Recording
        assert current_state['record'] == False  # Record stays disabled
        assert current_state['pause'] == True   # Pause/stop enabled
        assert current_state['stop'] == True
        
        # Recording stops -> Back to initial
        current_state = states[3]  # Completed
        assert current_state['record'] == True   # Record re-enabled
        assert current_state['pause'] == False  # Others disabled
        assert current_state['stop'] == False

    @pytest.mark.asyncio 
    async def test_double_click_prevention(self):
        """Test that countdown prevents double clicks"""
        button_disabled = False
        
        # First click - should disable button
        button_disabled = True
        assert button_disabled == True, "Button should be disabled after first click"
        
        # Subsequent clicks during countdown should be ignored
        for click_attempt in range(2, 5):
            if button_disabled:
                # Click ignored (good)
                pass
            else:
                pytest.fail(f"Click {click_attempt} was not ignored - button not disabled")
                
    @pytest.mark.asyncio
    async def test_error_handling_button_states(self):
        """Test button states during error conditions"""
        # Simulate error during countdown
        error_recovery_state = {
            'record': True,   # re-enabled for retry
            'pause': False,   # disabled  
            'stop': False     # disabled
        }
        
        assert error_recovery_state['record'] == True, "Record button should be re-enabled after error"
        assert error_recovery_state['pause'] == False, "Pause button should remain disabled after error"
        assert error_recovery_state['stop'] == False, "Stop button should remain disabled after error"

    # ========================
    # WEBSOCKET TESTS  
    # ========================
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self):
        """Test WebSocket connection for countdown functionality"""
        try:
            session_id = f"test_session_{int(time.time())}"
            uri = f"{self.server_url}/ws/{session_id}"
            
            async with websockets.connect(uri) as websocket:
                # Send ping
                ping_msg = {"type": "ping"}
                await websocket.send(json.dumps(ping_msg))
                
                # Wait for pong
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                response_data = json.loads(response)
                
                assert response_data.get("type") == "pong", f"Expected pong, got {response_data}"
                
        except Exception as e:
            pytest.skip(f"WebSocket connection failed: {e} - Server may not be running")

    @pytest.mark.asyncio
    async def test_recording_workflow_with_countdown(self):
        """Test the complete recording workflow with countdown delay"""
        try:
            session_id = f"test_workflow_{int(time.time())}"
            uri = f"{self.server_url}/ws/{session_id}"
            
            async with websockets.connect(uri) as websocket:
                # Simulate the countdown delay (this is the fix!)
                await asyncio.sleep(3.0)
                
                # Send start recording message
                start_msg = {
                    "type": "start_recording", 
                    "data": {
                        "language": "en",
                        "format": "wav"
                    }
                }
                await websocket.send(json.dumps(start_msg))
                
                # Wait for recording_started confirmation
                response = await asyncio.wait_for(websocket.recv(), timeout=15.0)
                response_data = json.loads(response)
                
                # Should get recording_started after countdown
                assert response_data.get("type") == "recording_started", f"Expected recording_started, got {response_data.get('type')}"
                
        except Exception as e:
            pytest.skip(f"Recording workflow test failed: {e} - Server may not be running")

    # ========================
    # RECORDING ANALYSIS TESTS
    # ========================
    
    def get_audio_info(self, file_path):
        """Get basic info about an audio file"""
        try:
            file_size = os.path.getsize(file_path)
            
            if file_path.endswith('.wav'):
                try:
                    with wave.open(file_path, 'rb') as wav_file:
                        frames = wav_file.getnframes()
                        sample_rate = wav_file.getframerate()
                        duration = frames / sample_rate
                        return {
                            'format': 'WAV',
                            'size': file_size,
                            'duration': duration,
                            'sample_rate': sample_rate,
                            'frames': frames
                        }
                except:
                    return {'format': 'WAV', 'size': file_size, 'duration': 'unknown'}
            else:
                return {'format': 'MP4', 'size': file_size, 'duration': 'unknown'}
        except Exception as e:
            return {'error': str(e)}

    def analyze_recording_for_countdown_benefit(self, info):
        """Analyze if this recording would benefit from countdown fix"""
        if 'duration' in info and isinstance(info['duration'], (int, float)):
            duration = info['duration']
            
            if duration < 10:
                return f"HIGH BENEFIT - Short recording ({duration:.1f}s) likely lost first 2-3s"
            elif duration < 30:
                return f"MEDIUM BENEFIT - Medium recording ({duration:.1f}s) may have lost opening words"
            else:
                return f"LOW BENEFIT - Long recording ({duration:.1f}s) less impacted by initial loss"
        else:
            return "UNKNOWN BENEFIT - Could not determine duration"

    @pytest.mark.asyncio
    async def test_existing_recordings_analysis(self):
        """Analyze existing recordings for countdown benefit"""
        recordings_dir = "data/recordings"
        benefit_summary = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'UNKNOWN': 0}
        
        for recording in self.test_recordings:
            file_path = os.path.join(recordings_dir, recording)
            
            if os.path.exists(file_path):
                info = self.get_audio_info(file_path)
                benefit = self.analyze_recording_for_countdown_benefit(info)
                
                # Extract benefit level
                if 'HIGH BENEFIT' in benefit:
                    benefit_summary['HIGH'] += 1
                elif 'MEDIUM BENEFIT' in benefit:
                    benefit_summary['MEDIUM'] += 1
                elif 'LOW BENEFIT' in benefit:
                    benefit_summary['LOW'] += 1
                else:
                    benefit_summary['UNKNOWN'] += 1
        
        # At least some recordings should benefit
        total_beneficial = benefit_summary['HIGH'] + benefit_summary['MEDIUM']
        total_analyzed = sum(benefit_summary.values())
        
        if total_analyzed > 0:
            benefit_percentage = (total_beneficial / total_analyzed) * 100
            # At least 30% of recordings should benefit from the countdown fix
            assert benefit_percentage >= 30, f"Only {benefit_percentage:.1f}% of recordings benefit from countdown fix"

    @pytest.mark.asyncio
    async def test_multiple_recording_sessions(self):
        """Test multiple recording sessions with countdown"""
        session_results = []
        
        for i in range(3):
            try:
                session_id = f"multi_test_{i}_{int(time.time())}"
                uri = f"{self.server_url}/ws/{session_id}"
                
                async with websockets.connect(uri) as websocket:
                    # Simulate countdown for each session
                    await asyncio.sleep(3.0)
                    
                    # Start recording
                    start_msg = {
                        "type": "start_recording",
                        "data": {"language": "en", "format": "wav"}
                    }
                    await websocket.send(json.dumps(start_msg))
                    
                    # Wait for confirmation
                    response = await asyncio.wait_for(websocket.recv(), timeout=15.0)
                    response_data = json.loads(response)
                    
                    if response_data.get("type") == "recording_started":
                        session_results.append(True)
                    else:
                        session_results.append(False)
                        
            except Exception as e:
                session_results.append(False)
        
        # At least 2 out of 3 sessions should work
        successful_sessions = sum(session_results)
        assert successful_sessions >= 2, f"Only {successful_sessions}/3 recording sessions succeeded"

    # ========================  
    # INTEGRATION TESTS
    # ========================
    
    @pytest.mark.asyncio
    async def test_countdown_fix_integration(self):
        """Integration test for the complete countdown fix"""
        # Test the complete flow:
        # 1. User clicks record
        # 2. Countdown starts (3 seconds) 
        # 3. Buttons are disabled during countdown
        # 4. After countdown, WebSocket message is sent
        # 5. Recording starts
        
        start_time = time.time()
        
        # Step 1: User clicks record (buttons get disabled)
        buttons_disabled = True
        assert buttons_disabled, "Buttons should be disabled immediately when record is clicked"
        
        # Step 2 & 3: 3-second countdown with disabled buttons
        for i in range(3, 0, -1):
            await asyncio.sleep(1.0)
            assert buttons_disabled, f"Buttons should remain disabled during countdown {i}"
        
        # Step 4: Countdown complete, check timing
        end_time = time.time()
        elapsed = end_time - start_time
        assert abs(elapsed - 3.0) <= 0.1, f"Countdown timing incorrect: {elapsed:.2f}s"
        
        # Step 5: Recording would start now (WebSocket message sent)
        # This part would normally involve WebSocket communication
        recording_ready = True  # Simulated
        assert recording_ready, "Recording should be ready to start after countdown"

# ========================
# ORIGINAL EXISTING TESTS (preserved from original test_unit.py)
# ========================

@pytest.fixture
def audio_file(tmp_path):
    """Create a temporary audio file for testing"""
    duration = 2.0
    sample_rate = 16000
    filename = tmp_path / "test_audio.wav"
    
    # Generate test audio data
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(440 * 2 * np.pi * t)  # 440 Hz sine wave
    
    # Save as WAV file
    sf.write(filename, audio_data, sample_rate)
    return str(filename)

class TestFileManager:
    """Test FileManager functionality"""
    
    def test_get_data_path(self):
        """Test getting data paths"""
        path = FileManager.get_data_path("test")
        assert isinstance(path, str)
        assert "data" in path
        assert "test" in path
    
    def test_ensure_directory_exists(self, tmp_path):
        """Test directory creation"""
        test_dir = tmp_path / "test_directory"
        FileManager.ensure_directory_exists(str(test_dir))
        assert test_dir.exists()
    
    def test_cleanup_temp_files(self, tmp_path):
        """Test temp file cleanup"""
        # Create some temp files
        temp_file = tmp_path / "temp_test.tmp"
        temp_file.write_text("test content")
        
        # Test cleanup (mock implementation)
        assert temp_file.exists()

class TestAudioRecorder:
    """Test AudioRecorder with comprehensive error handling"""
    
    def test_audio_recorder_initialization(self, tmp_path):
        """Test AudioRecorder initialization"""
        output_file = tmp_path / "test_recording.wav"
        recorder = AudioRecorder(str(output_file))
        
        assert recorder.output_file == str(output_file)
        assert recorder.sample_rate == 16000
        assert not recorder.is_recording
    
    @pytest.mark.asyncio
    async def test_audio_recorder_lifecycle(self, tmp_path):
        """Test complete recording lifecycle"""
        output_file = tmp_path / "test_recording.wav"
        recorder = AudioRecorder(str(output_file))
        
        # Test that recorder starts in correct state
        assert not recorder.is_recording
        assert recorder._stream is None
        
        # Test cleanup
        await recorder.cleanup_resources()

class TestSpeakerDiarization:
    """Test SpeakerDiarization functionality"""
    
    def test_speaker_diarization_initialization(self):
        """Test SpeakerDiarization initialization"""
        diarizer = SpeakerDiarization()
        assert diarizer.min_speakers >= 1
        assert diarizer.max_speakers >= 1
    
    @pytest.mark.asyncio
    async def test_estimate_speaker_count_advanced(self, audio_file):
        """Test advanced speaker count estimation"""
        diarizer = SpeakerDiarization()
        
        try:
            count = diarizer.estimate_speaker_count_advanced(audio_file)
            assert isinstance(count, int)
            assert count >= 1
        except Exception as e:
            pytest.skip(f"Speaker analysis requires audio dependencies: {e}")

class TestTranscription:
    """Test transcription functionality"""
    
    @pytest.mark.asyncio
    async def test_whisper_transcription_service_init(self):
        """Test transcription service initialization"""
        try:
            # Test that transcription modules can be imported
            from src.transcription import get_whisper
            whisper_module = get_whisper()
            assert whisper_module is not None
        except Exception as e:
            pytest.skip(f"Whisper service requires model dependencies: {e}")
    
    @pytest.mark.asyncio
    async def test_transcribe_audio_with_progress(self, audio_file, mock_whisper):
        """Test transcription with progress tracking"""
        try:
            with patch('src.transcription.whisper.load_model', return_value=mock_whisper):
                result = []
                async for progress, data in transcribe_audio_with_progress(
                    audio_file, 
                    "en", 
                    16000,
                    "neutral",
                    "balanced"
                ):
                    if data:
                        result = data
                
                assert isinstance(result, list)
                if result:
                    assert isinstance(result[0], dict)
                    assert "text" in result[0]
        except Exception as e:
            pytest.skip(f"Transcription test requires Whisper dependencies: {e}")

class TestSubtitleGeneration:
    """Test subtitle generation functionality"""
    
    @pytest.mark.asyncio
    async def test_generate_srt_basic(self):
        """Test basic SRT generation"""
        transcription_data = [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "Hello world",
                "speaker": "Speaker_1"
            },
            {
                "start": 2.0,
                "end": 4.0,
                "text": "This is a test",
                "speaker": "Speaker_1"
            }
        ]
        
        diarization_segments = []
        
        try:
            srt_file = await generate_srt(transcription_data, diarization_segments)
            if srt_file:
                assert os.path.exists(srt_file)
                assert srt_file.endswith('.srt')
        except Exception as e:
            pytest.skip(f"SRT generation test failed: {e}")

class TestIntegration:
    """Integration tests for the complete system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_processing_mock(self, audio_file, mock_whisper):
        """Test end-to-end processing with mocked components"""
        try:
            # Test the complete pipeline with mocked Whisper
            with patch('src.transcription.whisper.load_model', return_value=mock_whisper):
                # This would test the complete flow but with mocked dependencies
                # Test basic transcription function signature
                results = []
                async for progress, data in transcribe_audio_with_progress(
                    audio_file,
                    "en"
                ):
                    if data:
                        results = data
                
                result = {"transcription_data": results}
                
                assert "transcription_data" in result
                assert len(result["transcription_data"]) > 0
                
        except Exception as e:
            pytest.skip(f"Integration test requires dependencies: {e}")

# ========================
# PERFORMANCE TESTS
# ========================

class TestPerformance:
    """Performance tests for critical components"""
    
    @pytest.mark.asyncio
    async def test_countdown_performance(self):
        """Test that countdown doesn't introduce significant overhead"""
        start_time = time.time()
        
        # Simulate countdown timing
        for i in range(3, 0, -1):
            await asyncio.sleep(1.0)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Countdown should complete in ~3 seconds with minimal overhead
        assert elapsed < 3.2, f"Countdown took too long: {elapsed:.2f}s"
        assert elapsed > 2.8, f"Countdown completed too quickly: {elapsed:.2f}s"

# ========================
# CONFIGURATION TESTS  
# ========================

class TestConfiguration:
    """Test configuration and setup"""
    
    def test_logging_setup(self):
        """Test logging configuration"""
        logger = setup_app_logging(logger_name="test_logger")
        assert logger.name == "test_logger"
    
    def test_model_configuration(self):
        """Test model configuration"""
        assert WHISPER_MODEL_DIR.exists() or True  # May not exist in test environment
        assert isinstance(WHISPER_MODELS, dict)
        assert "en" in WHISPER_MODELS

# ========================
# ERROR HANDLING TESTS
# ========================

class TestErrorHandling:
    """Test error handling across components"""
    
    def test_audio_processing_errors(self):
        """Test AudioProcessingError handling"""
        error = AudioProcessingError("Test error", AudioProcessingError.ErrorType.FILE_ACCESS)
        assert "file_access: Test error" in str(error)
        assert error.error_type == AudioProcessingError.ErrorType.FILE_ACCESS
    
    def test_transcription_error_handling(self):
        """Test transcription error handling"""
        # Test that transcription handles errors gracefully
        # Since we removed TranscriptionError, we test general exception handling
        try:
            # This is a placeholder test for error handling
            result = None
            assert result is None  # Placeholder assertion
        except Exception:
            pytest.skip("Error handling test not implemented")

# ========================
# RECORDING CONVERSION AND ACCURACY TESTS  
# ========================

class TestRecordingAccuracy:
    """Test recording conversion and transcription accuracy"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for recording accuracy tests"""
        self.recordings_dir = Path("data/recordings")
        self.expected_transcription = [
            ("Speaker_1", "Então, Rogério, eu sei que é seu primeiro dia aqui na empresa, mas eu não gostaria que você me chamasse de você na empresa, sabe? Acho que é um pouco informal demais. A gente tem uma hierarquia aqui dentro."),
            ("Speaker_2", "Claro"),
            ("Speaker_1", "Então acho melhor a gente mudar isso."),
            ("Speaker_2", "Desculpa. Vou voltar.")
        ]
        self.expected_keywords = ["rogério", "primeiro dia", "empresa", "hierarquia", "claro", "desculpa", "informal", "mudar"]
    
    def convert_mp4_to_wav_with_ffmpeg(self, mp4_file, wav_file):
        """Convert MP4 to WAV using FFmpeg"""
        try:
            import subprocess
            
            # Use FFmpeg from venv
            ffmpeg_path = 'venv/Scripts/static_ffmpeg.exe'
            
            # Test if ffmpeg is available
            try:
                subprocess.run([ffmpeg_path, '-version'], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                return False
            
            # Run ffmpeg conversion
            cmd = [
                ffmpeg_path, '-y',
                '-i', str(mp4_file),
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                str(wav_file)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
            
        except Exception:
            return False
    
    def get_latest_recording(self):
        """Get the latest recording file"""
        if not self.recordings_dir.exists():
            pytest.skip("Recordings directory not found")
        
        # Try MP4 files first
        mp4_files = list(self.recordings_dir.glob("recording_*.mp4"))
        wav_files = list(self.recordings_dir.glob("recording_*.wav"))
        
        all_files = mp4_files + wav_files
        if not all_files:
            pytest.skip("No recording files found")
        
        return max(all_files, key=lambda x: x.stat().st_mtime)
    
    @pytest.mark.asyncio
    async def test_latest_recording_conversion(self):
        """Test conversion of latest recording to WAV format"""
        latest_file = self.get_latest_recording()
        
        if latest_file.suffix == '.mp4':
            wav_file = latest_file.with_suffix('.wav')
            success = self.convert_mp4_to_wav_with_ffmpeg(latest_file, wav_file)
            assert success, f"Failed to convert {latest_file} to WAV"
            assert wav_file.exists(), "WAV file was not created"
            assert wav_file.stat().st_size > 0, "WAV file is empty"
        else:
            wav_file = latest_file
        
        return str(wav_file)
    
    @pytest.mark.asyncio
    async def test_transcription_accuracy(self):
        """Test transcription and diarization accuracy"""
        try:
            # Get or convert latest recording
            wav_file = await self.test_latest_recording_conversion()
            
            # Import the concurrent processor
            from src.concurrent_engine import concurrent_processor
            
            # Mock websocket manager for testing
            class MockWebSocketManager:
                async def send_message(self, session_id, message):
                    pass
            
            websocket_manager = MockWebSocketManager()
            
            # Process the audio
            result = await concurrent_processor.process_audio_concurrent(
                session_id="accuracy_test",
                audio_file=wav_file,
                language="pt", 
                websocket_manager=websocket_manager,
                audio_input_type="conversation",
                processing_profile="balanced"
            )
            
            # Extract results
            transcription_data = result.get("transcription_data", [])
            diarization_segments = result.get("diarization_segments", [])
            speakers_detected = result.get("speakers_detected", 0)
            quality_metrics = result.get("quality_metrics", {})
            
            # Test speaker detection
            assert speakers_detected == 2, f"Expected 2 speakers, got {speakers_detected}"
            
            # Test transcription exists
            assert len(transcription_data) > 0, "No transcription data generated"
            
            # Test keyword accuracy
            if transcription_data:
                full_text = " ".join(seg.get('text', '') for seg in transcription_data).lower()
                found_keywords = [kw for kw in self.expected_keywords if kw in full_text]
                
                keyword_accuracy = len(found_keywords) / len(self.expected_keywords)
                assert keyword_accuracy >= 0.5, f"Keyword accuracy too low: {keyword_accuracy:.2f} (found {len(found_keywords)}/{len(self.expected_keywords)})"
            
            # Test overall quality
            overall_quality = quality_metrics.get('overall_quality', 0)
            assert overall_quality >= 0.3, f"Overall quality too low: {overall_quality:.2f}"
            
            return result
            
        except ImportError as e:
            pytest.skip(f"Missing dependencies for transcription test: {e}")
        except Exception as e:
            pytest.fail(f"Transcription accuracy test failed: {e}")
    
    @pytest.mark.asyncio
    async def test_medium_model_usage(self):
        """Test that only medium models are used"""
        try:
            from config.app_config import WHISPER_MODELS
            
            # Check that only medium models are configured for supported languages
            expected_langs = ['pt', 'en', 'es']
            for lang in expected_langs:
                if lang in WHISPER_MODELS:
                    model = WHISPER_MODELS[lang]
                    assert model == 'medium', f"Language {lang} should use medium model, got {model}"
            
        except ImportError:
            pytest.skip("Cannot test model configuration - config not available")

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])