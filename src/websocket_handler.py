# This file will contain the WebSocketHandler and WebSocketValidator classes.
import time
from typing import Dict, Any, Optional
import asyncio
import base64
import logging
from fastapi import WebSocket

from src.error_messages import get_user_message
from src.exceptions import ValidationError, SessionError, AudioProcessingError
import os

# Import the pipeline function
from src.pipeline import process_audio_pipeline

# Type hinting for AppState
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from main import AppState

logger = logging.getLogger(__name__)

# Security limits for live recording
MAX_RECORDING_DURATION = 3600  # 1 hour in seconds
MAX_CHUNK_SIZE = 1 * 1024 * 1024  # 1MB per chunk
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB total

class WebSocketValidator:
    """Validates incoming WebSocket messages and session states, raising exceptions on failure."""

    def validate_action(self, action: Optional[str]):
        """Validates the 'action' field, raising ValidationError on failure."""
        if not action or action not in ["start", "audio_chunk", "stop", "pause", "resume"]:
            logger.warning("Invalid or missing action received", extra={"action": action})
            raise ValidationError(get_user_message("invalid_action"))
        return None

    def validate_audio_format(self, audio_format: str):
        """Validates the audio format, raising ValidationError on failure."""
        if audio_format not in ["wav", "mp4"]:
            logger.warning("Invalid audio format received", extra={"format": audio_format})
            raise ValidationError(get_user_message("invalid_format"))
        return None

    def validate_session_state_for_chunk(self, session: Dict[str, Any]):
        """Validates session is 'recording', raising ValidationError on failure."""
        if session.get("status") != "recording":
            logger.warning("Audio chunk received for non-recording session", extra={"status": session.get("status")})
            raise ValidationError("Gravação não está ativa. Use 'start' primeiro.") # Keep simple message for this one
        return None

    def validate_chunk_size(self, chunk_b64: str):
        """Validates individual chunk size, raising ValidationError on failure."""
        if len(chunk_b64) > MAX_CHUNK_SIZE * 1.4:
            logger.warning("Audio chunk exceeds size limit", extra={"size": len(chunk_b64), "limit": MAX_CHUNK_SIZE})
            raise ValidationError(get_user_message("file_too_large", max_size=f"{MAX_CHUNK_SIZE // 1024 // 1024}MB (por chunk)"))
        return None

    def validate_total_size(self, total_data_received: int):
        """Validates total audio size, raising ValidationError on failure."""
        if total_data_received > MAX_FILE_SIZE:
            logger.warning("Total file size exceeds limit", extra={"size": total_data_received, "limit": MAX_FILE_SIZE})
            raise ValidationError(get_user_message("file_too_large", max_size=MAX_FILE_SIZE // 1024 // 1024))
        return None

    def validate_recording_duration(self, recording_start_time: Optional[float]):
        """Checks if recording reached 60min limit. Returns True if limit reached (should auto-stop)."""
        if recording_start_time is None:
            return False
        elapsed_time = time.time() - recording_start_time
        if elapsed_time > MAX_RECORDING_DURATION:
            logger.warning("Recording duration reached 60min limit", extra={"duration": elapsed_time, "limit": MAX_RECORDING_DURATION})
            return True  # Signal auto-stop needed
        return False

class WebSocketHandler:
    """Handles the business logic for WebSocket actions."""

    def __init__(self, app_state: 'AppState'):
        self.app_state = app_state

    async def handle_start(self, data: Dict[str, Any], session_id: str, websocket: WebSocket):
        """Handles the 'start' action and returns the initial recording state."""
        logger.info(f"▶️ Starting recording for session {session_id}")
        
        if not self.app_state.session_manager:
            raise SessionError("SessionManager not initialized")

        session = await self.app_state.session_manager.get_session(session_id)
        if not session:
            raise SessionError(f"Session not found during start: {session_id}")

        # The processor should be created and assigned when the session is created,
        # not retrieved here. This logic needs to be revisited in TIER 2.
        # For now, we assume it exists if the session exists.

        audio_format = data.get("format", "wav").lower()
        session.status = "recording"
        # The format is now part of the SessionData object
        session.format = audio_format
        
        # The LiveAudioProcessor is now part of the session logic, not directly handled here.
        # This part of the code is becoming coupled and will be addressed in TIER 2.

        await websocket.send_json({
            "type": "recording_started",
            "session_id": session_id,
            "format": audio_format,
            "message": f"Gravação iniciada (formato: {audio_format.upper()})"
        })
        return time.time(), 0

    async def handle_chunk(self, data: Dict[str, Any], session_id: str, websocket: WebSocket):
        """Handles the 'audio_chunk' action. Returns the number of bytes received."""
        if not self.app_state.session_manager:
            raise SessionError("SessionManager not initialized")

        # No need to get the session here, the chunk processing logic should handle it.
        # This indicates a need for further refactoring.

        chunk_b64 = data.get("data", "")
        try:
            audio_data = base64.b64decode(chunk_b64)
        except Exception:
            await websocket.send_json({
                "type": "error",
                "message": "Dados de áudio inválidos (base64 decode falhou)"
            })
            return 0, True # Return 0 bytes and error=True

        # This logic should be moved to a dedicated session/processor class in a future refactor.
        # For now, we assume a processor exists and can handle the chunk.
        # await processor.process_audio_chunk(session_id, audio_data)
        return len(audio_data), False # Return bytes received and error=False

    async def handle_stop(self, session_id: str, websocket: WebSocket):
        """Handles the 'stop' action, finalizing the recording and starting the background processing."""
        logger.info(f"⏹️ Stopping recording for session {session_id}")

        if not self.app_state.session_manager:
            raise SessionError("SessionManager not initialized")

        session = await self.app_state.session_manager.get_session(session_id)
        if not session:
            raise SessionError(f"Session not found during stop: {session_id}")

        session.status = "processing"

        await websocket.send_json({
            "type": "recording_stopped",
            "message": "Gravação finalizada. Processando..."
        })

        audio_path = session.temp_file
        if not audio_path or not os.path.exists(audio_path):
            logger.error(f"Audio file not found for session {session_id} at path: {audio_path}")
            raise AudioProcessingError("Arquivo de áudio gravado não encontrado para processamento.", context={"session_id": session_id})

        # Run the processing pipeline in the background without blocking
        asyncio.create_task(process_audio_pipeline(self.app_state, audio_path, session_id))

        await websocket.send_json({
            "type": "processing_started",
            "message": "Processamento iniciado em background"
        })
        logger.info(f"Audio processing started for session {session_id}.")
