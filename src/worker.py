# src/worker.py

import logging
import queue
import time
import asyncio

logger = logging.getLogger(__name__)

def transcription_worker(
    transcription_queue: queue.Queue,
    file_manager,
    live_audio_processor,
    transcription_service,
    session_manager,
    loop: asyncio.AbstractEventLoop
):
    while True:
        try:
            job = transcription_queue.get()
            if job is None:  # Poison pill to exit
                logger.info("Poison pill received. Transcription worker shutting down.")
                break

            session_id = job.get("session_id")
            if not session_id:
                logger.warning("Invalid job received (no session_id), skipping.")
                continue

            job_type = job.get("type", "audio_chunk")

            if job_type == "audio_chunk":
                # Audio chunks are buffered by LiveAudioProcessor
                # No real-time transcription - all processing happens at stop
                continue

            elif job_type == "stop":
                logger.info(f"Stop signal received for session {session_id}. Starting final processing.")
                wav_path = job.get("wav_path")

                # Validate session still exists before processing
                try:
                    session_check_future = asyncio.run_coroutine_threadsafe(
                        session_manager.get_session(session_id), loop
                    )
                    session_exists = session_check_future.result(timeout=1)

                    if not session_exists:
                        logger.warning(f"Session {session_id} no longer exists. Skipping final processing.")
                        continue
                except Exception as e:
                    logger.warning(f"Failed to validate session {session_id}: {e}. Skipping final processing.")
                    continue

                # Run complete transcription + diarization + SRT generation
                if wav_path:
                    # Import diarization service
                    from src.dependencies import get_diarization_service
                    diarization_service = get_diarization_service()

                    # Schedule finalization in event loop
                    asyncio.run_coroutine_threadsafe(
                        _finalize_live_recording(
                            session_manager, session_id,
                            wav_path, file_manager, diarization_service, transcription_service
                        ),
                        loop
                    )
                    logger.info(f"Scheduled final processing for session {session_id}")
                else:
                    logger.warning(f"No WAV path provided for session {session_id}, skipping processing")

            elif job_type == "echo":
                payload = job.get("payload")
                message = {"type": "echo_response", "payload": payload}
                asyncio.run_coroutine_threadsafe(
                    _send_message(session_manager, session_id, message),
                    loop
                )

            transcription_queue.task_done()

        except Exception as e:
            logger.error(f"Error in transcription worker: {e}", exc_info=True)

async def _send_message(session_manager, session_id: str, message: dict):
    """Helper to send message via websocket"""
    session = await session_manager.get_session(session_id)
    if session and session.websocket:
        await session.websocket.send_json(message)
    else:
        logger.warning(f"Cannot send message to {session_id}: session removed or WebSocket closed")

async def _finalize_live_recording(
    session_manager, session_id: str,
    wav_path: str, file_manager, diarization_service, transcription_service
):
    """
    Final processing step for live recordings:
    - Transcribes complete audio with word timestamps
    - Diarizes audio using transcription segments
    - Generates SRT file
    - Stores file paths in session for downloads
    - Sends complete result to UI
    """
    try:
        from src.subtitle_generator import generate_srt
        import librosa
        import time

        logger.info(f"Finalizing live recording for session {session_id}...")

        # Get session data
        session = await session_manager.get_session(session_id)
        if not session:
            logger.warning(f"Session {session_id} not found during finalization")
            return

        # Get audio duration for metrics
        audio_duration = librosa.get_duration(path=wav_path)

        # Transcribe complete audio WITH word timestamps for SRT generation
        logger.info(f"Transcribing complete audio with word timestamps for session {session_id}...")
        processing_start = time.time()

        transcription_result = await transcription_service.transcribe_with_enhancements(
            wav_path,
            word_timestamps=True  # Enable for precise diarization alignment
        )

        processing_time = time.time() - processing_start
        processing_ratio = processing_time / audio_duration if audio_duration > 0 else 0.0

        # Debug: Log transcription segments count
        logger.info(f"Whisper returned {len(transcription_result.segments) if transcription_result.segments else 0} segments for session {session_id}")

        # Run diarization on complete audio with detailed segments (containing word timestamps)
        logger.info(f"Running diarization on complete audio: {wav_path}")
        diarization_result = await diarization_service.diarize(wav_path, transcription_result.segments)

        # Debug: Log diarization segments count
        logger.info(f"Diarization returned {len(diarization_result['segments'])} segments for session {session_id}")

        # Fallback: if diarization returns empty segments, create synthetic segment
        if not diarization_result["segments"] and transcription_result.text:
            logger.warning(f"No segments from diarization, creating fallback segment for {session_id}")
            diarization_result["segments"] = [{
                "start": 0.0,
                "end": audio_duration,
                "text": transcription_result.text.strip(),
                "speaker": "SPEAKER_00"
            }]
            diarization_result["num_speakers"] = 1

        # Generate SRT file
        srt_path = await generate_srt(diarization_result["segments"], file_manager=file_manager, filename=f"{session_id}.srt")
        logger.info(f"SRT file generated: {srt_path}")

        # Store file paths in session for download endpoints
        session.files["audio"] = wav_path
        session.files["subtitles"] = str(srt_path)
        session.files["transcript"] = transcription_result.text.strip()
        logger.info(f"File paths stored in session.files for {session_id}")

        # Send complete result to UI
        final_result = {
            "segments": diarization_result["segments"],
            "num_speakers": diarization_result["num_speakers"],
            "transcription": transcription_result.text.strip(),
            "processing_time": round(processing_time, 2),
            "processing_ratio": round(processing_ratio, 2),
            "audio_duration": round(audio_duration, 2)
        }

        if session.websocket:
            await session.websocket.send_json({
                "type": "complete",
                "result": final_result
            })
            logger.info(f"Final result sent to UI for session {session_id}")

    except Exception as e:
        logger.error(f"Error finalizing live recording for {session_id}: {e}", exc_info=True)
        session = await session_manager.get_session(session_id)
        if session and session.websocket:
            await session.websocket.send_json({
                "type": "error",
                "message": f"Erro ao finalizar processamento: {str(e)}"
            })
