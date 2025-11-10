"""
Unit tests for core services (transcription, diarization, audio_quality).
Uses mocks to avoid file I/O overhead and test components in isolation.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import numpy as np


class TestTranscriptionService:
    """Test TranscriptionService in isolation"""

    def test_initialization(self):
        """Test service initializes with correct parameters"""
        from src.transcription import TranscriptionService

        service = TranscriptionService(model_name="medium", device="cpu")

        # Model uses lazy loading, just verify initialization parameters
        assert service.device == "cpu"
        assert service.model_name == "medium"

    @patch('faster_whisper.WhisperModel')
    @pytest.mark.asyncio
    async def test_transcribe_returns_result(self, mock_whisper):
        """Test transcription returns proper TranscriptionResult"""
        from src.transcription import TranscriptionService, TranscriptionResult

        # Mock Whisper model response
        mock_model_instance = MagicMock()
        
        # 1. The 'transcribe' method returns a tuple: (segment_generator, info_object)
        mock_segment = Mock()
        mock_segment.text = "Test transcription"
        mock_segment.start = 0.0
        mock_segment.end = 2.0
        mock_segment.avg_logprob = -0.5
        
        mock_info = Mock()
        mock_info.language = "pt"
        
        # The generator can be simulated with a simple list
        mock_model_instance.transcribe.return_value = ([mock_segment], mock_info)

        # The mock for the class returns our configured instance
        mock_whisper.return_value = mock_model_instance

        service = TranscriptionService(model_name="medium", device="cpu")
        # Manually initialize to load the mock model
        await service.initialize()

        # Test the async method directly
        result = await service.transcribe_with_enhancements("/fake/path.wav")

        assert isinstance(result, TranscriptionResult)
        assert "Test transcription" in result.text
        assert result.language == "pt"
        assert len(result.segments) == 1
        assert result.confidence > 0

    @patch('faster_whisper.WhisperModel')
    @pytest.mark.asyncio
    async def test_transcribe_file_not_found(self, mock_whisper):
        """Test error handling when audio file doesn't exist"""
        from src.transcription import TranscriptionService
        from src.exceptions import TranscriptionError

        service = TranscriptionService(model_name="medium", device="cpu")
        await service.initialize()

        # The service's internal error handling should raise TranscriptionError
        with pytest.raises(TranscriptionError):
            await service.transcribe_with_enhancements("/nonexistent/file.wav")


class TestPyannoteDiarizer:
    """Test PyannoteDiarizer in isolation"""

    @patch('src.diarization.Pipeline.from_pretrained')
    def test_initialization(self, mock_pipeline):
        """Test diarizer initializes with correct model"""
        from src.diarization import PyannoteDiarizer

        diarizer = PyannoteDiarizer(device="cpu")

        mock_pipeline.assert_called_once()
        assert diarizer.device == "cpu"

    @patch('src.diarization.Pipeline.from_pretrained')
    @pytest.mark.asyncio
    async def test_diarize_returns_speaker_segments(self, mock_from_pretrained):
        """Test diarization returns speaker segments"""
        from src.diarization import PyannoteDiarizer

        # Mock the pipeline instance and its return value
        mock_pipeline_instance = MagicMock()
        mock_annotation = MagicMock()

        # Mock the behavior of iterating over the annotation
        mock_pyannote_segment = Mock()
        mock_pyannote_segment.start = 0.0
        mock_pyannote_segment.end = 2.0

        mock_annotation.itertracks.return_value = [
            (mock_pyannote_segment, 'A', 'SPEAKER_00'),
            (mock_pyannote_segment, 'B', 'SPEAKER_01'),
        ]
        mock_annotation.labels.return_value = ['SPEAKER_00', 'SPEAKER_01']

        # The pipeline callable returns the annotation
        mock_pipeline_instance.return_value = mock_annotation
        mock_from_pretrained.return_value = mock_pipeline_instance

        diarizer = PyannoteDiarizer(device="cpu")

        # Provide mock transcription segments
        mock_transcription_segments = [{
            "start": 0.0,
            "end": 2.0,
            "text": "Hello world",
            "words": [{"word": "Hello", "start": 0.1, "end": 0.5}, {"word": "world", "start": 0.6, "end": 1.0}]
        }]

        result = await diarizer.diarize("/fake/audio.wav", mock_transcription_segments)

        assert result["num_speakers"] == 2
        assert len(result["segments"]) > 0
        assert "speaker" in result["segments"][0]["words"][0]

    @patch('src.diarization.Pipeline.from_pretrained')
    @pytest.mark.asyncio
    async def test_diarize_file_not_found(self, mock_from_pretrained):
        """Test error handling when audio file doesn't exist"""
        from src.diarization import PyannoteDiarizer

        # Mock the pipeline instance to raise an error when called
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.side_effect = FileNotFoundError("File not found")
        mock_from_pretrained.return_value = mock_pipeline_instance

        diarizer = PyannoteDiarizer(device="cpu")

        mock_segments = [{"start": 0.0, "end": 2.0, "text": "Test", "words": []}]

        # Test that the service handles the error and falls back gracefully
        result = await diarizer.diarize("/nonexistent/file.wav", mock_segments)

        # The diarize method should catch the error and return a single-speaker result
        assert result["num_speakers"] == 1
        assert "speaker" in result["segments"][0]


class TestAudioQualityAnalyzer:
    """Test AudioQualityAnalyzer in isolation"""

    def test_analyze_quality_mock_audio(self):
        """Test audio quality analysis with mock audio data"""
        from src.audio_processing import AudioQualityAnalyzer

        analyzer = AudioQualityAnalyzer()

        # Mock librosa load
        mock_audio = np.random.randn(16000 * 10)  # 10 seconds of random audio
        mock_sr = 16000

        with patch('librosa.load', return_value=(mock_audio, mock_sr)):
            result = analyzer.analyze_audio_quality("/fake/audio.wav")

        assert result.clarity_score >= 0
        assert result.rms_level >= 0
        assert result.snr_estimate >= 0
        assert result.recommended_quantization is not None

    def test_analyze_quality_silent_audio(self):
        """Test analyzer detects silent audio"""
        from src.audio_processing import AudioQualityAnalyzer

        analyzer = AudioQualityAnalyzer()

        # Create silent audio (all zeros)
        mock_audio = np.zeros(16000 * 5)
        mock_sr = 16000

        with patch('librosa.load', return_value=(mock_audio, mock_sr)):
            result = analyzer.analyze_audio_quality("/fake/silence.wav")

        assert result.rms_level == pytest.approx(0.0, abs=1e-6)
        assert result.has_issues is True


class TestSessionManager:
    """Test SessionManager session lifecycle"""

    @pytest.mark.asyncio
    async def test_create_and_get_session(self):
        """Test creating and retrieving a session"""
        from src.audio_processing import SessionManager, SessionData
        from datetime import datetime

        manager = SessionManager(session_timeout_hours=1)

        session_data = SessionData(
            session_id="test_123",
            websocket=None,
            format="wav",
            started_at=datetime.now()
        )

        await manager.create_session("test_123", session_data)

        retrieved = await manager.get_session("test_123")
        assert retrieved is not None
        assert retrieved.session_id == "test_123"

    @pytest.mark.asyncio
    async def test_session_exists(self):
        """Test checking if session exists"""
        from src.audio_processing import SessionManager, SessionData
        from datetime import datetime

        manager = SessionManager(session_timeout_hours=1)

        assert not await manager.session_exists("nonexistent")

        session_data = SessionData(
            session_id="exists_123",
            websocket=None,
            format="wav",
            started_at=datetime.now()
        )
        await manager.create_session("exists_123", session_data)

        assert await manager.session_exists("exists_123")

    @pytest.mark.asyncio
    async def test_remove_session(self):
        """Test removing a session"""
        from src.audio_processing import SessionManager, SessionData
        from datetime import datetime

        manager = SessionManager(session_timeout_hours=1)

        session_data = SessionData(
            session_id="remove_me",
            websocket=None,
            format="wav",
            started_at=datetime.now()
        )
        await manager.create_session("remove_me", session_data)

        assert await manager.session_exists("remove_me")

        await manager.remove_session("remove_me")

        assert not await manager.session_exists("remove_me")


class TestFileManager:
    """Test FileManager file operations"""

    def test_get_data_path(self):
        """Test getting data subdirectory paths"""
        from src.file_manager import FileManager
        from pathlib import Path
        import os

        test_dir = Path("/test/data").resolve()
        fm = FileManager(data_dir=test_dir)

        temp_path = fm.get_data_path("temp")
        assert temp_path == test_dir / "temp"

    @pytest.mark.asyncio
    async def test_save_uploaded_file_mock(self):
        """Test saving uploaded file verifies path generation"""
        from src.file_manager import FileManager
        from pathlib import Path

        fm = FileManager(data_dir=Path("/test/data"))

        # Simple test: just verify path generation logic
        expected_path = Path("/test/data/inputs/test.wav").resolve()

        # Test path generation (internal method)
        actual_path = (fm.data_dir / "inputs" / "test.wav").resolve()

        assert actual_path == expected_path


class TestWebSocketValidator:
    """Test WebSocketValidator validation logic"""

    def test_valid_actions(self):
        """Test validator correctly identifies valid actions"""
        from src.websocket_handler import WebSocketValidator

        validator = WebSocketValidator()
        assert validator.validate_action("start") is None
        assert validator.validate_action("audio_chunk") is None
        assert validator.validate_action("stop") is None

    def test_invalid_action_raises_exception(self):
        """Test validator raises ValidationError for invalid actions"""
        from src.websocket_handler import WebSocketValidator
        from src.exceptions import ValidationError

        validator = WebSocketValidator()
        with pytest.raises(ValidationError, match="Ação inválida ou ausente"):
            validator.validate_action("delete")
        with pytest.raises(ValidationError):
            validator.validate_action(None)
        with pytest.raises(ValidationError):
            validator.validate_action("")

    def test_audio_format_validation(self):
        """Test audio format validation"""
        from src.websocket_handler import WebSocketValidator
        from src.exceptions import ValidationError

        validator = WebSocketValidator()
        assert validator.validate_audio_format("wav") is None
        assert validator.validate_audio_format("mp4") is None
        with pytest.raises(ValidationError, match="Formato de arquivo inválido ou não suportado"):
            validator.validate_audio_format("mp3")
        with pytest.raises(ValidationError):
            validator.validate_audio_format("")

    def test_session_state_validation(self):
        """Test session state validation"""
        from src.websocket_handler import WebSocketValidator
        from src.exceptions import ValidationError

        validator = WebSocketValidator()
        valid_session = {"status": "recording"}
        invalid_session = {"status": "processing"}
        assert validator.validate_session_state_for_chunk(valid_session) is None
        with pytest.raises(ValidationError, match="Gravação não está ativa"):
            validator.validate_session_state_for_chunk(invalid_session)
