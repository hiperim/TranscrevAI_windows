import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor
from src.transcription import transcribe_audio_with_progress
from src.speaker_diarization import SpeakerDiarization
from src.logging_setup import setup_app_logging

# Import new optimization components
from src.model_manager import get_model_manager
from src.audio_chunker import get_audio_chunker
from src.contextual_corrector import get_confidence_corrector
from src.performance_monitor import get_performance_monitor
from config.whisper_optimization import get_optimized_config, validate_real_time_performance

logger = setup_app_logging(logger_name="transcrevai.concurrent_engine")

class OptimizedConcurrentProcessor:
    """
    Enhanced concurrent processor with optimizations
    
    Features:
    - Optimized model management with caching
    - Smart audio chunking for real-time processing
    - Contextual corrections for better accuracy
    - Performance monitoring integration
    - Real-time ratio validation
    """
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2)  # One for transcription, one for diarization
        self.active_sessions = {}
        
        # Initialize optimization components
        self.model_manager = get_model_manager()
        self.audio_chunker = get_audio_chunker()
        self.corrector = get_confidence_corrector()
        self.performance_monitor = get_performance_monitor()
        
        # Start performance monitoring
        try:
            asyncio.create_task(self.performance_monitor.start_monitoring())
        except RuntimeError:
            logger.info("Performance monitoring will start when event loop is available")
        
        logger.info("OptimizedConcurrentProcessor initialized with all optimization components")
        
    async def process_audio_optimized(self, session_id: str, audio_file: str, language: str, websocket_manager, model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Process audio with optimized concurrent transcription and diarization
        
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
            logger.info(f"Starting optimized concurrent processing for session {session_id}")
            
            # Get audio duration for performance tracking
            import librosa
            try:
                audio_duration = librosa.get_duration(filename=audio_file)
            except Exception:
                audio_duration = 60.0  # Default estimate
            
            # Start performance monitoring
            await self.performance_monitor.track_processing_session(
                session_id, audio_duration, language, get_optimized_config(language).get("model_name", "small")
            )
            
            # Send audio analysis progress
            await websocket_manager.send_audio_analysis_progress(session_id, {
                "stage": "analyzing",
                "progress": 10,
                "duration": audio_duration
            })
            
            # Initialize session tracking with optimization data
            self.active_sessions[session_id] = {
                "transcription_progress": 0,
                "diarization_progress": 0,
                "start_time": time.time(),
                "status": "processing",
                "audio_duration": audio_duration,
                "language": language
            }
            
            # Create optimized audio chunks
            try:
                chunks = await self.audio_chunker.create_optimized_chunks(audio_file)
                await websocket_manager.send_audio_analysis_progress(session_id, {
                    "stage": "chunking",
                    "progress": 20,
                    "chunks": len(chunks)
                })
            except Exception as e:
                logger.warning(f"Audio chunking failed, using single chunk: {e}")
                chunks = [{"file_path": audio_file, "start_time": 0.0, "end_time": audio_duration, "index": 0}]
            
            # Preload optimized model if needed
            model_load_start = time.time()
            try:
                model = await self.model_manager.get_optimized_model(language)
                model_load_time = time.time() - model_load_start
                await self.performance_monitor.record_model_load_time(session_id, model_load_time)
            except Exception as e:
                logger.error(f"Failed to load optimized model: {e}")
                # Fall back to standard processing
            
            # Create tasks for concurrent execution with chunk processing
            transcription_task = asyncio.create_task(
                self._optimized_transcription_with_progress(session_id, chunks, language, websocket_manager)
            )
            
            diarization_task = asyncio.create_task(
                self._diarization_with_progress(session_id, audio_file, websocket_manager)
            )
            
            # Send detailed progress updates
            await websocket_manager.send_chunked_progress(session_id, {
                "transcription": 25,
                "diarization": 25,
                "stage": "processing",
                "chunk_number": 1,
                "total_chunks": len(chunks)
            })
            
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
                    "end": audio_duration,
                    "speaker": "Speaker_1"
                }]
            
            # Apply contextual corrections to transcription
            if isinstance(transcription_result, list) and transcription_result:
                try:
                    corrected_transcription = self.corrector.correct_low_confidence_words(
                        transcription_result, language
                    )
                    transcription_result = corrected_transcription
                except Exception as e:
                    logger.warning(f"Contextual correction failed: {e}")
            
            # Calculate processing statistics
            session_data = self.active_sessions.get(session_id, {})
            processing_time = time.time() - session_data.get("start_time", time.time())
            
            # Validate real-time performance
            real_time_ratio = processing_time / audio_duration if audio_duration > 0 else 1.0
            is_real_time = validate_real_time_performance(processing_time, audio_duration)
            
            # Complete performance tracking
            await self.performance_monitor.complete_processing_session(session_id)
            
            # Clean up chunk files
            if chunks and len(chunks) > 1:
                try:
                    await self.audio_chunker.cleanup_chunk_files(chunks)
                except Exception as e:
                    logger.warning(f"Chunk cleanup failed: {e}")
            
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
                "audio_duration": audio_duration,
                "real_time_ratio": real_time_ratio,
                "is_real_time": is_real_time,
                "chunks_processed": len(chunks),
                "method": "optimized_concurrent_whisper_pyaudioanalysis"
            }
            
            logger.info(f"Optimized concurrent processing completed for session {session_id}: "
                       f"{len(transcription_result) if isinstance(transcription_result, list) else 0} transcription segments, "
                       f"{len(diarization_result) if isinstance(diarization_result, list) else 0} diarization segments, "
                       f"{unique_speakers} speakers, "
                       f"processing time: {processing_time:.2f}s, "
                       f"real-time ratio: {real_time_ratio:.2f}x")
            
            return result
            
        except Exception as e:
            logger.error(f"Optimized concurrent processing failed for session {session_id}: {e}")
            # Clean up session
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            raise
    
    # Keep the old method for backward compatibility
    async def process_audio_concurrent(self, session_id: str, audio_file: str, language: str, websocket_manager, model_path: Optional[str] = None) -> Dict[str, Any]:
        """Legacy method - redirects to optimized version"""
        return await self.process_audio_optimized(session_id, audio_file, language, websocket_manager, model_path)
    
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
    
    async def _optimized_transcription_with_progress(self, session_id: str, chunks: List[Dict], language: str, websocket_manager) -> List[Dict]:
        """Handle optimized transcription with chunked progress updates"""
        try:
            transcription_data = []
            total_chunks = len(chunks)
            
            for i, chunk in enumerate(chunks):
                # Record chunk processing
                await self.performance_monitor.record_chunk_processed(session_id)
                
                # Process chunk with optimized Whisper
                chunk_start_time = time.time()
                chunk_data = None  # Initialize to avoid unbound variable
                
                async for progress, data in transcribe_audio_with_progress(
                    chunk["file_path"], language, 16000
                ):
                    chunk_data = data  # Store data from the generator
                    
                    # Calculate overall progress based on chunk progress
                    chunk_progress = (i / total_chunks) * 100 + (progress / total_chunks)
                    
                    # Update session progress
                    if session_id in self.active_sessions:
                        self.active_sessions[session_id]["transcription_progress"] = chunk_progress
                        
                        # Send detailed chunked progress
                        diarization_progress = self.active_sessions[session_id].get("diarization_progress", 0)
                        
                        # Calculate real-time ratio for this chunk
                        chunk_processing_time = time.time() - chunk_start_time
                        chunk_duration = chunk.get("duration", 30.0)
                        chunk_real_time_ratio = chunk_processing_time / chunk_duration if chunk_duration > 0 else 1.0
                        
                        await websocket_manager.send_chunked_progress(session_id, {
                            "transcription": chunk_progress,
                            "diarization": diarization_progress,
                            "chunk_number": i + 1,
                            "total_chunks": total_chunks,
                            "stage": f"transcribing_chunk_{i+1}",
                            "real_time_ratio": chunk_real_time_ratio,
                            "chunk_details": {
                                "start_time": chunk.get("start_time", 0),
                                "end_time": chunk.get("end_time", 0),
                                "duration": chunk_duration
                            }
                        })
                
                # Adjust timestamps for chunk offset using the stored chunk_data
                if chunk_data:
                    chunk_start = chunk.get("start_time", 0)
                    for segment in chunk_data:
                        if "start" in segment:
                            segment["start"] += chunk_start
                        if "end" in segment:
                            segment["end"] += chunk_start
                    
                    transcription_data.extend(chunk_data)
            
            return transcription_data
            
        except Exception as e:
            logger.error(f"Optimized transcription with progress failed: {e}")
            # Fallback to standard transcription
            return await self._transcription_with_progress(session_id, chunks[0]["file_path"], language, websocket_manager)
    
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
concurrent_processor = OptimizedConcurrentProcessor()