# src/worker.py

import logging
import queue
import time
import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Any, List
from concurrent.futures import Future

from src.transcription import TranscriptionResult

if TYPE_CHECKING:
    from main import AppState

logger = logging.getLogger(__name__)

# --- Constants ---
CHUNK_SECONDS = 5  # We will test with 5 and 15 seconds later
# Assuming 16kHz, 16-bit mono audio, so 32,000 bytes per second.
CHUNK_BYTES_THRESHOLD = 16000 * 2 * CHUNK_SECONDS

class WorkerSession:
    """Manages the state of a single session within the worker."""
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.audio_buffer: List[bytes] = []
        self.total_bytes: int = 0
        self.batch_count: int = 0

def transcription_worker(app_state: "AppState", loop: asyncio.AbstractEventLoop):
    sessions: Dict[str, WorkerSession] = {}

    if not app_state.transcription_queue:
        logger.error("Transcription queue is not initialized. Worker thread exiting.")
        return

    while True:
        try:
            job = app_state.transcription_queue.get()
            if job is None:  # Poison pill to exit
                logger.info("Poison pill received. Transcription worker shutting down.")
                break

            session_id = job.get("session_id")
            if not session_id:
                logger.warning("Invalid job received (no session_id), skipping.")
                continue

            # Get or create a session state within the worker
            if session_id not in sessions:
                sessions[session_id] = WorkerSession(session_id)
            worker_session = sessions[session_id]

            job_type = job.get("type", "audio_chunk")

            if job_type == "audio_chunk":
                audio_chunk = job.get("audio_chunk_bytes")
                if not audio_chunk:
                    continue

                worker_session.audio_buffer.append(audio_chunk)
                worker_session.total_bytes += len(audio_chunk)

                if worker_session.total_bytes >= CHUNK_BYTES_THRESHOLD:
                    _process_audio_batch(app_state, loop, worker_session)

            elif job_type == "stop":
                logger.info(f"Stop signal received for session {session_id}. Processing final batch.")
                if worker_session.audio_buffer:
                    _process_audio_batch(app_state, loop, worker_session, is_final=True)
                # Clean up the session from the worker's memory
                del sessions[session_id]

            elif job_type == "echo":
                payload = job.get("payload")
                message = {"type": "echo_response", "payload": payload}
                asyncio.run_coroutine_threadsafe(
                    app_state.send_message(session_id, message),
                    loop
                )

            app_state.transcription_queue.task_done()

        except Exception as e:
            logger.error(f"Error in transcription worker: {e}", exc_info=True)

def _process_audio_batch(app_state: "AppState", loop: asyncio.AbstractEventLoop, worker_session: WorkerSession, is_final: bool = False):
    """Prepares and schedules a batch of audio for transcription, but does not block."""
    worker_session.batch_count += 1
    batch_num = worker_session.batch_count
    session_id = worker_session.session_id

    audio_data = b''.join(worker_session.audio_buffer)
    worker_session.audio_buffer.clear()
    worker_session.total_bytes = 0

    if not audio_data:
        logger.warning(f"Attempted to process an empty batch for session {session_id}")
        return

    logger.info(f"Processing batch {batch_num} for session {session_id} ({len(audio_data)} bytes).")

    if not app_state.file_manager or not app_state.live_audio_processor or not app_state.transcription_service:
        logger.error("Core services (FileManager, LiveAudioProcessor, or TranscriptionService) not initialized in worker.")
        return

    temp_dir = app_state.file_manager.get_data_path("temp")

    # Detect if data is already WAV format (starts with 'RIFF' and contains 'WAVE')
    is_wav = audio_data[:4] == b'RIFF' and b'WAVE' in audio_data[:20]

    if is_wav:
        # Data is already WAV, save directly
        wav_path = temp_dir / f"{session_id}_batch_{batch_num}.wav"
        webm_path = None
        try:
            with open(wav_path, 'wb') as f:
                f.write(audio_data)
        except Exception as e:
            logger.error(f"Failed to save WAV batch {batch_num} for session {session_id}: {e}", exc_info=True)
            return
    else:
        # Data is WebM, needs conversion
        webm_path = temp_dir / f"{session_id}_batch_{batch_num}.webm"
        wav_path = temp_dir / f"{session_id}_batch_{batch_num}.wav"

        try:
            with open(webm_path, 'wb') as f:
                f.write(audio_data)

            app_state.live_audio_processor._convert_webm_to_wav(str(webm_path), str(wav_path), sample_rate=16000)

        except Exception as e:
            logger.error(f"Failed to prepare audio batch {batch_num} for session {session_id}: {e}", exc_info=True)
            # Cleanup failed files
            try:
                if webm_path and webm_path.exists():
                    webm_path.unlink()
            except Exception as cleanup_err:
                logger.warning(f"Failed to cleanup WebM after error: {cleanup_err}")

            try:
                if wav_path.exists():
                    wav_path.unlink()
            except Exception as cleanup_err:
                logger.warning(f"Failed to cleanup WAV after error: {cleanup_err}")
            return

    def on_transcription_complete(future: Future[TranscriptionResult]):
        """Callback executed in the main event loop thread once transcription is done."""
        try:
            transcription_result = future.result()

            # --- Send Result ---
            message = {
                "type": "transcription_chunk",
                "data": {
                    "text": transcription_result.text,
                    "is_final_batch": is_final
                }
            }
            asyncio.run_coroutine_threadsafe(app_state.send_message(session_id, message), loop)
            logger.info(f"Sent transcription for batch {batch_num} of session {session_id}.")

        except Exception as e:
            logger.error(f"Error in transcription callback for batch {batch_num} of session {session_id}: {e}", exc_info=True)
            # Notify user via WebSocket
            error_message = {
                "type": "processing_error",
                "message": f"Erro ao processar áudio: {str(e)}",
                "batch": batch_num
            }
            asyncio.run_coroutine_threadsafe(app_state.send_message(session_id, error_message), loop)
        finally:
            # --- Cleanup ---
            try:
                if wav_path.exists():
                    wav_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to cleanup WAV file {wav_path}: {e}")

            try:
                if webm_path and webm_path.exists():
                    webm_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to cleanup WebM file {webm_path}: {e}")

    # --- Schedule Transcription ---
    future = asyncio.run_coroutine_threadsafe(
        app_state.transcription_service.transcribe_with_enhancements(str(wav_path)),
        loop
    )
    future.add_done_callback(on_transcription_complete)
