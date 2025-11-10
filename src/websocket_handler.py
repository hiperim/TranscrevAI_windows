import time
from typing import Dict, Any, Optional
import asyncio
import logging
from fastapi import WebSocket
from src.error_messages import get_user_message
from src.exceptions import ValidationError, SessionError, AudioProcessingError
import os

# Import pipeline function
from src.pipeline import process_audio_pipeline

logger = logging.getLogger(__name__)

# Limits for live recording
MAX_RECORDING_DURATION = 3600  # 1 hr in sec
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500mb total

class WebSocketValidator:

    def validate_action(self, action: Optional[str]):
        """Validates the 'action' field, raising ValidationError"""
        if not action or action not in ["start", "audio_chunk", "stop", "pause", "resume"]:
            logger.warning("Invalid or missing action received", extra={"action": action})
            raise ValidationError(get_user_message("invalid_action"))
        return None

    def validate_audio_format(self, audio_format: str):
        """Validates the audio format, raising ValidationError"""
        if audio_format not in ["wav", "mp4"]:
            logger.warning("Invalid audio format received", extra={"format": audio_format})
            raise ValidationError(get_user_message("invalid_format"))
        return None

    def validate_recording_duration(self, recording_start_time: Optional[float]):
        """Checks if recording reached 60min limit and auto-stops"""
        if recording_start_time is None:
            return False
        elapsed_time = time.time() - recording_start_time
        if elapsed_time > MAX_RECORDING_DURATION:
            logger.warning("Recording duration reached 60min limit", extra={"duration": elapsed_time, "limit": MAX_RECORDING_DURATION})
            return True  # Signal for auto-stop
        return False

class WebSocketHandler:

    def __init__(self, session_manager, live_audio_processor):
        self.session_manager = session_manager
        self.live_audio_processor = live_audio_processor

    async def handle_start(self, data: Dict[str, Any], session_id: str, websocket: WebSocket):
        """Handles the 'start' action and returns the initial recording state"""
        logger.info(f"▶️ Starting recording for session {session_id}")

        session = await self.session_manager.get_session(session_id)
        if not session:
            raise SessionError(f"Session not found during start: {session_id}")

        audio_format = data.get("format", "wav").lower()
        session.status = "recording"
        session.format = audio_format

        await websocket.send_json({
            "type": "recording_started",
            "session_id": session_id,
            "format": audio_format,
            "message": f"Gravação iniciada (formato: {audio_format.upper()})"
        })
        return time.time(), 0


    async def handle_stop(self, session_id: str, websocket: WebSocket):
        """Handles the 'stop' action, finalizing the recording and starting the background processing"""
        logger.info(f"⏹️ Stopping recording for session {session_id}")

        session = await self.session_manager.get_session(session_id)
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

        # Run the processing pipeline in the background
        asyncio.create_task(process_audio_pipeline(audio_path, session_id))

        await websocket.send_json({
            "type": "processing_started",
            "message": "Processamento iniciado em background"
        })
        logger.info(f"Audio processing started for session {session_id}.")
