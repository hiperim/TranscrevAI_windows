# This file will contain the WebSocketHandler and WebSocketValidator classes.
import time
from typing import Dict, Any, Optional
import asyncio
import base64
import logging
from fastapi import WebSocket

# Import the pipeline function
from src.pipeline import run_pipeline_sync

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
    """Validates incoming WebSocket messages and session states."""

    def validate_action(self, action: Optional[str]) -> Optional[str]:
        """Validates the 'action' field."""
        if not action or action not in ["start", "audio_chunk", "stop"]:
            return "Ação inválida ou ausente. Use 'start', 'audio_chunk' ou 'stop'."
        return None

    def validate_audio_format(self, audio_format: str) -> Optional[str]:
        """Validates the audio format for the 'start' action."""
        if audio_format not in ["wav", "mp4"]:
            return "Formato de áudio inválido. Use 'wav' ou 'mp4'."
        return None

    def validate_session_state_for_chunk(self, session: Dict[str, Any]) -> Optional[str]:
        """Validates if the session is in a 'recording' state to receive chunks."""
        if session.get("status") != "recording":
            return "Gravação não está ativa. Use 'start' primeiro."
        return None

    def validate_chunk_size(self, chunk_b64: str) -> Optional[str]:
        """Validates the size of an individual audio chunk."""
        # Base64 adds ~33% overhead, so we check against a slightly larger limit.
        if len(chunk_b64) > MAX_CHUNK_SIZE * 1.4:
            return f"Chunk de áudio muito grande (tamanho máximo: {MAX_CHUNK_SIZE // 1024 // 1024}MB)."
        return None

    def validate_total_size(self, total_data_received: int) -> Optional[str]:
        """Validates the total accumulated size of audio data."""
        if total_data_received > MAX_FILE_SIZE:
            return f"Tamanho total do arquivo excedido (máximo: {MAX_FILE_SIZE // 1024 // 1024}MB)."
        return None

    def validate_recording_duration(self, recording_start_time: Optional[float]) -> Optional[str]:
        """Validates the total duration of the recording."""
        if recording_start_time is None:
            return None # Cannot validate if start time is not set
        elapsed_time = time.time() - recording_start_time
        if elapsed_time > MAX_RECORDING_DURATION:
            return f"Duração máxima da gravação excedida (máximo: {MAX_RECORDING_DURATION // 60} minutos)."
        return None

class WebSocketHandler:
    """Handles the business logic for WebSocket actions."""

    def __init__(self, app_state: 'AppState'):
        self.app_state = app_state

    async def handle_start(self, data: Dict[str, Any], session_id: str, websocket: WebSocket):
        """Handles the 'start' action and returns the initial recording state."""
        logger.info(f"▶️ Starting recording for session {session_id}")
        
        session = self.app_state.session_manager.get_session(session_id)
        processor = session.get("processor")
        
        audio_format = data.get("format", "wav").lower()
        session["status"] = "recording"
        session["audio_format"] = audio_format
        
        await processor.start_recording(session_id)
        
        await websocket.send_json({
            "type": "recording_started",
            "session_id": session_id,
            "format": audio_format,
            "message": f"Gravação iniciada (formato: {audio_format.upper()})"
        })
        # Return state for the main loop to manage
        return time.time(), 0

    async def handle_chunk(self, data: Dict[str, Any], session_id: str, websocket: WebSocket):
        """Handles the 'audio_chunk' action. Returns the number of bytes received."""
        session = self.app_state.session_manager.get_session(session_id)
        processor = session.get("processor")

        chunk_b64 = data.get("data", "")
        try:
            audio_data = base64.b64decode(chunk_b64)
        except Exception:
            await websocket.send_json({
                "type": "error",
                "message": "Dados de áudio inválidos (base64 decode falhou)"
            })
            return 0, True # Return 0 bytes and error=True

        await processor.process_audio_chunk(session_id, audio_data)
        return len(audio_data), False # Return bytes received and error=False

    async def handle_stop(self, session_id: str, websocket: WebSocket):
        """Handles the 'stop' action."""
        logger.info(f"⏹️ Stopping recording for session {session_id}")

        session = self.app_state.session_manager.get_session(session_id)
        processor = session.get("processor")

        wav_path = await processor.stop_recording(session_id)
        session["status"] = "processing"

        await websocket.send_json({
            "type": "recording_stopped",
            "message": "Gravação finalizada. Processando..."
        })

        audio_path = wav_path
        session["files"]["audio"] = str(audio_path)

        await websocket.send_json({
            "type": "progress",
            "stage": "transcription",
            "percentage": 10,
            "message": "Iniciando transcrição..."
        })
        
        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, run_pipeline_sync, self.app_state, str(audio_path), session_id)

        await websocket.send_json({
            "type": "processing_started",
            "message": "Processamento iniciado em background"
        })
        logger.info(f"Audio processing started for session {session_id}.")
