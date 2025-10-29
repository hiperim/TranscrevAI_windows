import asyncio
import logging
import time
import librosa

from src.subtitle_generator import generate_srt
from src.websocket_enhancements import MessagePriority

# Type hinting for AppState without causing circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from main import AppState

logger = logging.getLogger(__name__)

async def process_audio_pipeline(app_state: 'AppState', audio_path: str, session_id: str):
    """The complete, non-blocking pipeline for processing a single audio file."""
    try:
        if not all([app_state.transcription_service, app_state.diarization_service, app_state.audio_quality_analyzer]):
            raise RuntimeError("Application services are not initialized.")

        audio_duration = librosa.get_duration(path=audio_path)
        pipeline_start_time = time.time()

        await app_state.send_message(session_id, {'type': 'progress', 'stage': 'start', 'percentage': 5, 'message': 'Analisando qualidade do áudio...'})
        
        assert app_state.transcription_service is not None
        transcription_result = await app_state.transcription_service.transcribe_with_enhancements(audio_path)

        await app_state.send_message(session_id, {'type': 'progress', 'stage': 'diarization', 'percentage': 50, 'message': 'Transcrição concluída. Identificando falantes...'})

        assert app_state.diarization_service is not None
        diarization_result = await app_state.diarization_service.diarize(audio_path, transcription_result.segments)

        await app_state.send_message(session_id, {'type': 'progress', 'stage': 'srt', 'percentage': 80, 'message': 'Gerando legendas...'})

        srt_path = await generate_srt(diarization_result["segments"], output_path=app_state.file_manager.get_data_path("temp"), filename=f"{session_id}.srt")

        pipeline_end_time = time.time()
        actual_processing_time = pipeline_end_time - pipeline_start_time
        processing_ratio = actual_processing_time / audio_duration if audio_duration > 0 else 0

        logger.info(f"Processing complete for {session_id}: {actual_processing_time:.2f}s for {audio_duration:.2f}s audio (ratio: {processing_ratio:.2f}x)")
    except Exception as e:
        logger.error(f"Error in audio processing pipeline: {str(e)}", exc_info=True)
        await app_state.send_message(session_id, {'type': 'error', 'message': f"Erro no pipeline: {str(e)}"}, MessagePriority.CRITICAL)
        return

    final_result = {
        "segments": diarization_result["segments"],
        "num_speakers": diarization_result["num_speakers"],
        "processing_time": round(actual_processing_time, 2),
        "processing_ratio": round(processing_ratio, 2),
        "audio_duration": round(audio_duration, 2)
    }

    try:
        if app_state.session_manager:
            session = app_state.session_manager.get_session(session_id)
            if session:
                if "files" not in session:
                    session["files"] = {}
                session["files"]["audio"] = audio_path
                session["files"]["subtitles"] = str(srt_path)
                logger.info(f"File paths stored in SessionManager for {session_id}")

        with app_state._lock:
            if session_id not in app_state.sessions: app_state.sessions[session_id] = {}
            app_state.sessions[session_id]['srt_file_path'] = srt_path

        await app_state.send_message(session_id, {'type': 'complete', 'result': final_result}, MessagePriority.CRITICAL)

    except Exception as e:
        logger.error(f"Audio pipeline failed during finalization for session {session_id}: {e}", exc_info=True)
        await app_state.send_message(session_id, {'type': 'error', 'message': str(e)}, MessagePriority.CRITICAL)

def run_pipeline_sync(app_state: 'AppState', audio_path: str, session_id: str):
    """Synchronous wrapper to run the async pipeline in a separate thread."""
    try:
        asyncio.run(process_audio_pipeline(app_state, audio_path, session_id))
    except Exception as e:
        logger.error(f"Exception in synchronous pipeline runner for session {session_id}: {e}", exc_info=True)
