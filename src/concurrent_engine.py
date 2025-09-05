import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor
from src.transcription import transcribe_audio_with_progress
from src.speaker_diarization import SpeakerDiarization
from src.logging_setup import setup_app_logging

logger = setup_app_logging(logger_name="transcrevai.concurrent_engine")

class ConcurrentProcessor:
    """Concurrent processing engine for parallel transcription and diarization"""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2)  # One for transcription, one for diarization
        self.active_sessions = {}
        
    async def process_audio_concurrent(self, session_id: str, audio_file: str, language: str, websocket_manager, model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Process audio with concurrent transcription and diarization
        
        Args:
            session_id: Unique session identifier
            audio_file: Path to audio file
            language: Language code for transcription
            websocket_manager: WebSocket manager for real-time updates
            model_path: Optional model path (unused in Whisper implementation)
        
        Returns:
            Dict containing transcription data, diarization segments, and metadata
        """
        try:
            logger.info(f"Starting concurrent processing for session {session_id}")
            
            # Initialize session tracking
            self.active_sessions[session_id] = {
                "transcription_progress": 0,
                "diarization_progress": 0,
                "start_time": time.time(),
                "status": "processing"
            }
            
            # Create tasks for concurrent execution
            transcription_task = asyncio.create_task(
                self._transcription_with_progress(session_id, audio_file, language, websocket_manager)
            )
            
            diarization_task = asyncio.create_task(
                self._diarization_with_progress(session_id, audio_file, websocket_manager)
            )
            
            # Wait for both tasks to complete
            transcription_result, diarization_result = await asyncio.gather(
                transcription_task, 
                diarization_task,
                return_exceptions=True
            )
            
            # Handle potential exceptions
            if isinstance(transcription_result, Exception):
                logger.error(f"Transcription failed: {transcription_result}")
                transcription_result = []
            
            if isinstance(diarization_result, Exception):
                logger.error(f"Diarization failed: {diarization_result}")
                diarization_result = [{
                    "start": 0.0,
                    "end": 1.0,
                    "speaker": "Speaker_1"
                }]
            
            # Calculate processing statistics
            session_data = self.active_sessions.get(session_id, {})
            processing_time = time.time() - session_data.get("start_time", time.time())
            
            # Clean up session
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            # Ensure diarization_result is iterable before counting speakers
            if isinstance(diarization_result, Exception) or not isinstance(diarization_result, list):
                unique_speakers = 1
            else:
                unique_speakers = len(set(seg.get('speaker', 'Speaker_1') for seg in diarization_result))
            
            result = {
                "transcription_data": transcription_result,
                "diarization_segments": diarization_result,
                "speakers_detected": unique_speakers,
                "processing_time": processing_time,
                "method": "concurrent_whisper_pyannote"
            }
            
            logger.info(f"Concurrent processing completed for session {session_id}: "
                       f"{len(transcription_result) if isinstance(transcription_result, list) else 0} transcription segments, "
                       f"{len(diarization_result) if isinstance(diarization_result, list) else 0} diarization segments, "
                       f"{unique_speakers} speakers, "
                       f"processing time: {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Concurrent processing failed for session {session_id}: {e}")
            # Clean up session
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            raise
    
    async def _transcription_with_progress(self, session_id: str, audio_file: str, language: str, websocket_manager) -> List[Dict]:
        """Handle transcription with progress updates"""
        try:
            transcription_data = []
            
            async for progress, data in transcribe_audio_with_progress(
                audio_file, language, 16000
            ):
                # Update session progress
                if session_id in self.active_sessions:
                    self.active_sessions[session_id]["transcription_progress"] = progress
                    
                    # Send combined progress update
                    diarization_progress = self.active_sessions[session_id].get("diarization_progress", 0)
                    await websocket_manager.send_message(session_id, {
                        "type": "progress",
                        "transcription": progress,
                        "diarization": diarization_progress
                    })
                
                if data:
                    transcription_data = data
            
            return transcription_data
            
        except Exception as e:
            logger.error(f"Transcription with progress failed: {e}")
            raise
    
    async def _diarization_with_progress(self, session_id: str, audio_file: str, websocket_manager) -> List[Dict]:
        """Handle diarization with progress updates"""
        try:
            # Simulate progress updates for diarization (PyAnnote doesn't provide incremental progress)
            diarizer = SpeakerDiarization()
            
            # Send progress updates
            for progress in [10, 30, 60, 80]:
                if session_id in self.active_sessions:
                    self.active_sessions[session_id]["diarization_progress"] = progress
                    
                    transcription_progress = self.active_sessions[session_id].get("transcription_progress", 0)
                    await websocket_manager.send_message(session_id, {
                        "type": "progress", 
                        "transcription": transcription_progress,
                        "diarization": progress
                    })
                
                await asyncio.sleep(0.5)  # Small delay to simulate processing
            
            # Perform actual diarization
            segments = await diarizer.diarize_audio(audio_file)
            
            # Final progress update
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["diarization_progress"] = 100
                transcription_progress = self.active_sessions[session_id].get("transcription_progress", 0)
                await websocket_manager.send_message(session_id, {
                    "type": "progress",
                    "transcription": transcription_progress,
                    "diarization": 100
                })
            
            return segments
            
        except Exception as e:
            logger.error(f"Diarization with progress failed: {e}")
            raise
    
    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current processing status for a session"""
        return self.active_sessions.get(session_id)
    
    def cleanup_session(self, session_id: str):
        """Clean up session data"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(f"Cleaned up concurrent processing session: {session_id}")
    
    async def shutdown(self):
        """Shutdown the concurrent processor"""
        try:
            # Cancel all active sessions
            for session_id in list(self.active_sessions.keys()):
                self.cleanup_session(session_id)
            
            # Shutdown thread pool
            self.executor.shutdown(wait=True)
            logger.info("Concurrent processor shutdown complete")
            
        except Exception as e:
            logger.error(f"Concurrent processor shutdown failed: {e}")

# Global concurrent processor instance
concurrent_processor = ConcurrentProcessor()