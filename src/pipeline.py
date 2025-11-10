import logging
import time
import librosa

from src.subtitle_generator import generate_srt
from src.dependencies import (
    get_transcription_service,
    get_diarization_service,
    get_file_manager,
    get_session_manager
)

logger = logging.getLogger(__name__)

async def process_audio_pipeline(audio_path: str, session_id: str) -> None:
    """Non-blocking pipeline via directio inhections for processing single audio file"""
    transcription_service = get_transcription_service()
    diarization_service = get_diarization_service()
    file_manager = get_file_manager()
    session_manager = get_session_manager()

    try:
        audio_duration = librosa.get_duration(path=audio_path)
        pipeline_start_time = time.time()

        session = await session_manager.get_session(session_id)
        if session and session.websocket:
            try:
                await session.websocket.send_json({'type': 'progress', 'stage': 'start', 'percentage': 5, 'message': 'Analisando qualidade do áudio...'})
            except Exception:
                logger.warning(f"Could not send progress update to {session_id} (WebSocket closed)")

        transcription_result = await transcription_service.transcribe_with_enhancements(audio_path)

        if session and session.websocket:
            try:
                await session.websocket.send_json({'type': 'progress', 'stage': 'diarization', 'percentage': 50, 'message': 'Transcrição concluída. Identificando falantes...'})
            except Exception:
                logger.warning(f"Could not send progress update to {session_id} (WebSocket closed)")

        diarization_result = await diarization_service.diarize(audio_path, transcription_result.segments)

        if session and session.websocket:
            try:
                await session.websocket.send_json({'type': 'progress', 'stage': 'srt', 'percentage': 80, 'message': 'Gerando legendas...'})
            except Exception:
                logger.warning(f"Could not send progress update to {session_id} (WebSocket closed)")

        srt_path = await generate_srt(diarization_result["segments"], file_manager=file_manager, filename=f"{session_id}.srt")

        pipeline_end_time = time.time()
        actual_processing_time = pipeline_end_time - pipeline_start_time
        processing_ratio = actual_processing_time / audio_duration if audio_duration > 0 else 0

        logger.info(f"Processing complete for {session_id}: {actual_processing_time:.2f}s for {audio_duration:.2f}s audio (ratio: {processing_ratio:.2f}x)")
    except Exception as e:
        logger.error(f"Error in audio processing pipeline: {str(e)}", exc_info=True)
        session = await session_manager.get_session(session_id)
        if session and session.websocket:
            try:
                await session.websocket.send_json({'type': 'error', 'message': f"Erro no pipeline: {str(e)}"})
            except Exception:
                logger.warning(f"Could not send error to {session_id} (WebSocket closed)")
        return

    final_result = {
        "segments": diarization_result["segments"],
        "num_speakers": diarization_result["num_speakers"],
        "processing_time": round(actual_processing_time, 2),
        "processing_ratio": round(processing_ratio, 2),
        "audio_duration": round(audio_duration, 2)
    }

    try:
        session = await session_manager.get_session(session_id)
        if session:
            session.files["audio"] = audio_path
            session.files["subtitles"] = str(srt_path)
            session.status = "completed"  # Mark session as completed
            logger.info(f"File paths stored in SessionManager for {session_id}")

            if session.websocket:
                try:
                    await session.websocket.send_json({'type': 'complete', 'result': final_result})
                except Exception:
                    logger.warning(f"Could not send completion to {session_id} (WebSocket closed)")

    except Exception as e:
        logger.error(f"Audio pipeline failed during finalization for session {session_id}: {e}", exc_info=True)
        session = await session_manager.get_session(session_id)
        if session and session.websocket:
            try:
                await session.websocket.send_json({'type': 'error', 'message': str(e)})
            except Exception:
                logger.warning(f"Could not send error to {session_id} (WebSocket closed)")


