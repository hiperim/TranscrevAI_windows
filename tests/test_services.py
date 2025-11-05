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

    @patch('src.transcription.WhisperModel')
    def test_initialization(self, mock_whisper):
        """Test service initializes with correct model"""
        from src.transcription import TranscriptionService

        service = TranscriptionService(model_name="medium", device="cpu")

        mock_whisper.assert_called_once_with(
            "medium",
            device="cpu",
            compute_type="int8"
        )
        assert service.device == "cpu"

    @patch('src.transcription.WhisperModel')
    def test_transcribe_returns_result(self, mock_whisper):
        """Test transcription returns proper TranscriptionResult"""
        from src.transcription import TranscriptionService

        # Mock Whisper model response
        mock_model_instance = MagicMock()
        mock_segment = Mock()
        mock_segment.text = "Test transcription"
        mock_segment.start = 0.0
        mock_segment.end = 2.0
        mock_model_instance.transcribe.return_value = ([mock_segment], {"language": "pt"})

        mock_whisper.return_value = mock_model_instance

        service = TranscriptionService(model_name="medium", device="cpu")

        # Mock file existence
        with patch.object(Path, 'exists', return_value=True):
            result = service.transcribe("/fake/path.wav")

        assert result.text == "Test transcription"
        assert result.language == "pt"
        assert len(result.segments) == 1

    @patch('src.transcription.WhisperModel')
    def test_transcribe_file_not_found(self, mock_whisper):
        """Test error handling when audio file doesn't exist"""
        from src.transcription import TranscriptionService
        from src.exceptions import TranscriptionError

        service = TranscriptionService(model_name="medium", device="cpu")

        with patch.object(Path, 'exists', return_value=False):
            with pytest.raises(TranscriptionError):
                service.transcribe("/nonexistent/file.wav")


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
    def test_diarize_returns_speaker_segments(self, mock_pipeline):
        """Test diarization returns speaker segments"""
        from src.diarization import PyannoteDiarizer

        # Mock pipeline response
        mock_pipeline_instance = MagicMock()
        mock_annotation = Mock()
        mock_timeline = Mock()
        mock_segment = Mock()
        mock_segment.start = 0.0
        mock_segment.end = 2.0

        mock_timeline.__iter__ = Mock(return_value=iter([mock_segment]))
        mock_annotation.itertracks.return_value = [(mock_segment, None, "SPEAKER_00")]

        mock_pipeline_instance.return_value = mock_annotation
        mock_pipeline.return_value = mock_pipeline_instance

        diarizer = PyannoteDiarizer(device="cpu")

        with patch.object(Path, 'exists', return_value=True):
            result = diarizer.diarize("/fake/audio.wav")

        assert result.num_speakers >= 1
        assert len(result.speaker_segments) >= 1

    @patch('src.diarization.Pipeline.from_pretrained')
    def test_diarize_file_not_found(self, mock_pipeline):
        """Test error handling when audio file doesn't exist"""
        from src.diarization import PyannoteDiarizer
        from src.exceptions import DiarizationError

        diarizer = PyannoteDiarizer(device="cpu")

        with patch.object(Path, 'exists', return_value=False):
            with pytest.raises(DiarizationError):
                diarizer.diarize("/nonexistent/file.wav")


class TestAudioQualityAnalyzer:
    """Test AudioQualityAnalyzer in isolation"""

    def test_analyze_quality_mock_audio(self):
        """Test audio quality analysis with mock audio data"""
        from src.audio_processing import AudioQualityAnalyzer

        analyzer = AudioQualityAnalyzer()

        # Mock librosa load
        mock_audio = np.random.randn(16000 * 10)  # 10 seconds of random audio
        mock_sr = 16000

        with patch('src.audio_processing.librosa.load', return_value=(mock_audio, mock_sr)):
            result = analyzer.analyze_quality("/fake/audio.wav")

        assert result.sample_rate == mock_sr
        assert result.duration > 0
        assert result.rms_energy >= 0
        assert result.snr is not None

    def test_analyze_quality_silent_audio(self):
        """Test analyzer detects silent audio"""
        from src.audio_processing import AudioQualityAnalyzer

        analyzer = AudioQualityAnalyzer()

        # Create silent audio (all zeros)
        mock_audio = np.zeros(16000 * 5)
        mock_sr = 16000

        with patch('src.audio_processing.librosa.load', return_value=(mock_audio, mock_sr)):
            result = analyzer.analyze_quality("/fake/silence.wav")

        assert result.rms_energy == pytest.approx(0.0, abs=1e-6)
        assert result.has_voice is False


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

        fm = FileManager(data_dir=Path("/test/data"))

        temp_path = fm.get_data_path("temp")
        assert temp_path == Path("/test/data/temp")

    @pytest.mark.asyncio
    async def test_save_uploaded_file_mock(self):
        """Test saving uploaded file with mock"""
        from src.file_manager import FileManager
        from pathlib import Path

        fm = FileManager(data_dir=Path("/test/data"))

        # Mock file object
        mock_file = Mock()
        mock_file.read = Mock(return_value=asyncio.coroutine(lambda: b"fake audio data")())

        with patch('builtins.open', create=True) as mock_open:
            with patch.object(Path, 'mkdir'):
                result = await fm.save_uploaded_file(mock_file, "test.wav")

        assert "test.wav" in str(result)


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
