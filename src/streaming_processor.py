import asyncio
import logging
import time
import numpy as np
from typing import AsyncGenerator, Dict, Any, List, Tuple, Optional
from collections import deque
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

from src.transcription import TranscriptionService
from src.speaker_diarization import SpeakerDiarization
from src.advanced_audio_processing import AdvancedAudioProcessor
from src.logging_setup import setup_app_logging

logger = setup_app_logging(logger_name="transcrevai.streaming")

class StreamingProcessor:
    """
    Real-time streaming processor for audio transcription and speaker diarization.
    
    Implements parallel processing pipeline with:
    - Real-time audio chunking
    - Streaming transcription
    - Concurrent speaker diarization
    - Progressive result streaming
    """
    
    def __init__(self, chunk_duration: float = 2.0, overlap_duration: float = 0.5):
        """
        Initialize streaming processor.
        
        Args:
            chunk_duration: Duration of audio chunks in seconds
            overlap_duration: Overlap between chunks for better accuracy
        """
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.sample_rate = 16000
        
        # Processing queues
        self.audio_queue = asyncio.Queue(maxsize=50)
        self.transcription_queue = asyncio.Queue(maxsize=20)
        self.diarization_queue = asyncio.Queue(maxsize=20)
        
        # Results buffers
        self.transcription_buffer = deque(maxlen=100)
        self.diarization_buffer = deque(maxlen=100)
        
        # Processing state
        self.is_processing = False
        self.chunk_counter = 0
        self.total_duration = 0.0
        
        # Thread pool for CPU-intensive tasks
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Services
        self.transcription_service = None
        self.diarization_service = None
        self.audio_processor = AdvancedAudioProcessor(sample_rate=self.sample_rate)
        
    async def initialize_services(self, language: str = "en"):
        """Initialize transcription and diarization services."""
        try:
            self.transcription_service = TranscriptionService()
            await self.transcription_service.initialize(language)
            
            self.diarization_service = SpeakerDiarization()
            
            logger.info(f"Streaming services initialized for language: {language}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize streaming services: {e}")
            return False
    
    async def start_processing(self, websocket_manager, session_id: str) -> None:
        """Start the streaming processing pipeline."""
        if self.is_processing:
            return
        
        self.is_processing = True
        logger.info(f"Starting streaming processing for session {session_id}")
        
        # Start parallel processing tasks
        tasks = [
            asyncio.create_task(self._audio_chunking_worker()),
            asyncio.create_task(self._transcription_worker(websocket_manager, session_id)),
            asyncio.create_task(self._diarization_worker(websocket_manager, session_id)),
            asyncio.create_task(self._result_merger_worker(websocket_manager, session_id))
        ]
        
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Streaming processing error: {e}")
        finally:
            self.is_processing = False
    
    async def stop_processing(self):
        """Stop the streaming processing pipeline."""
        self.is_processing = False
        logger.info("Stopping streaming processing")
        
        # Clear queues
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        while not self.transcription_queue.empty():
            try:
                self.transcription_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        while not self.diarization_queue.empty():
            try:
                self.diarization_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
    
    async def add_audio_chunk(self, audio_data: np.ndarray, timestamp: float, 
                             enable_processing: bool = True) -> None:
        """Add audio chunk to processing queue with optional audio enhancement."""
        if not self.is_processing:
            return
        
        try:
            # Apply advanced audio processing if enabled
            processed_audio = audio_data
            if enable_processing:
                processed_audio = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self.audio_processor.process_audio_chunk,
                    audio_data,
                    True,  # enable_noise_reduction
                    True,  # enable_echo_cancellation
                    True   # enable_enhancement
                )
            
            chunk_info = {
                "data": processed_audio,
                "original_data": audio_data,
                "timestamp": timestamp,
                "chunk_id": self.chunk_counter,
                "duration": len(processed_audio) / self.sample_rate,
                "processed": enable_processing
            }
            
            await self.audio_queue.put(chunk_info)
            self.chunk_counter += 1
            self.total_duration = timestamp + chunk_info["duration"]
            
        except asyncio.QueueFull:
            logger.warning("Audio queue full, dropping chunk")
        except Exception as e:
            logger.error(f"Audio chunk processing error: {e}")
            # Add unprocessed chunk as fallback
            chunk_info = {
                "data": audio_data,
                "original_data": audio_data,
                "timestamp": timestamp,
                "chunk_id": self.chunk_counter,
                "duration": len(audio_data) / self.sample_rate,
                "processed": False
            }
            try:
                await self.audio_queue.put(chunk_info)
                self.chunk_counter += 1
                self.total_duration = timestamp + chunk_info["duration"]
            except asyncio.QueueFull:
                logger.warning("Audio queue full, dropping fallback chunk")
    
    async def _audio_chunking_worker(self):
        """Worker for processing audio chunks."""
        overlap_samples = int(self.overlap_duration * self.sample_rate)
        previous_chunk = None
        
        while self.is_processing:
            try:
                chunk_info = await asyncio.wait_for(
                    self.audio_queue.get(), timeout=1.0
                )
                
                audio_data = chunk_info["data"]
                
                # Add overlap from previous chunk
                if previous_chunk is not None and len(previous_chunk) >= overlap_samples:
                    overlap_data = previous_chunk[-overlap_samples:]
                    audio_data = np.concatenate([overlap_data, audio_data])
                    chunk_info["has_overlap"] = True
                else:
                    chunk_info["has_overlap"] = False
                
                # Queue for both transcription and diarization
                chunk_info["data"] = audio_data
                await self.transcription_queue.put(chunk_info.copy())
                await self.diarization_queue.put(chunk_info.copy())
                
                previous_chunk = chunk_info["data"]
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Audio chunking error: {e}")
    
    async def _transcription_worker(self, websocket_manager, session_id: str):
        """Worker for real-time transcription."""
        while self.is_processing:
            try:
                chunk_info = await asyncio.wait_for(
                    self.transcription_queue.get(), timeout=1.0
                )
                
                # Process transcription in thread pool
                transcription_result = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self._process_transcription_chunk,
                    chunk_info
                )
                
                if transcription_result:
                    # Add to buffer
                    self.transcription_buffer.append(transcription_result)
                    
                    # Send real-time update
                    await websocket_manager.send_message(session_id, {
                        "type": "streaming_transcription",
                        "chunk_id": chunk_info["chunk_id"],
                        "timestamp": chunk_info["timestamp"],
                        "text": transcription_result.get("text", ""),
                        "confidence": transcription_result.get("confidence", 0.0)
                    })
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Transcription worker error: {e}")
    
    async def _diarization_worker(self, websocket_manager, session_id: str):
        """Worker for real-time speaker diarization."""
        while self.is_processing:
            try:
                chunk_info = await asyncio.wait_for(
                    self.diarization_queue.get(), timeout=1.0
                )
                
                # Process diarization in thread pool
                diarization_result = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self._process_diarization_chunk,
                    chunk_info
                )
                
                if diarization_result:
                    # Add to buffer
                    self.diarization_buffer.append(diarization_result)
                    
                    # Send real-time update
                    await websocket_manager.send_message(session_id, {
                        "type": "streaming_diarization",
                        "chunk_id": chunk_info["chunk_id"],
                        "timestamp": chunk_info["timestamp"],
                        "speakers": diarization_result.get("speakers", [])
                    })
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Diarization worker error: {e}")
    
    async def _result_merger_worker(self, websocket_manager, session_id: str):
        """Worker for merging and optimizing results."""
        last_merge_time = time.time()
        merge_interval = 5.0  # Merge every 5 seconds
        
        while self.is_processing:
            try:
                current_time = time.time()
                
                if current_time - last_merge_time >= merge_interval:
                    # Merge recent results
                    merged_results = self._merge_recent_results()
                    
                    if merged_results:
                        await websocket_manager.send_message(session_id, {
                            "type": "merged_results",
                            "timestamp": current_time,
                            "transcription": merged_results.get("transcription", []),
                            "speakers": merged_results.get("speakers", []),
                            "confidence": merged_results.get("confidence", 0.0)
                        })
                    
                    last_merge_time = current_time
                
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Result merger error: {e}")
    
    def _process_transcription_chunk(self, chunk_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process transcription for a single chunk (runs in thread pool)."""
        try:
            if not self.transcription_service:
                return None
            
            audio_data = chunk_info["data"]
            
            # Ensure minimum chunk size
            if len(audio_data) < self.sample_rate * 0.5:  # Less than 0.5 seconds
                return None
            
            # Run transcription
            result = self.transcription_service.transcribe_chunk_sync(
                audio_data, self.sample_rate
            )
            
            if result and result.get("text"):
                return {
                    "chunk_id": chunk_info["chunk_id"],
                    "timestamp": chunk_info["timestamp"],
                    "duration": chunk_info["duration"],
                    "text": result["text"],
                    "confidence": result.get("confidence", 0.0),
                    "words": result.get("words", [])
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Transcription chunk processing error: {e}")
            return None
    
    def _process_diarization_chunk(self, chunk_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process speaker diarization for a single chunk (runs in thread pool)."""
        try:
            if not self.diarization_service:
                return None
            
            audio_data = chunk_info["data"]
            
            # Ensure minimum chunk size for diarization
            if len(audio_data) < self.sample_rate * 1.0:  # Less than 1 second
                return None
            
            # Create temporary file for diarization
            import tempfile
            import soundfile as sf
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                sf.write(temp_file.name, audio_data, self.sample_rate)
                
                # Run diarization
                segments = self.diarization_service.diarize(temp_file.name)
                
                # Clean up
                import os
                os.unlink(temp_file.name)
                
                if segments:
                    return {
                        "chunk_id": chunk_info["chunk_id"],
                        "timestamp": chunk_info["timestamp"],
                        "duration": chunk_info["duration"],
                        "speakers": segments
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Diarization chunk processing error: {e}")
            return None
    
    def _merge_recent_results(self) -> Optional[Dict[str, Any]]:
        """Merge recent transcription and diarization results."""
        try:
            # Get recent transcription results (last 10 seconds)
            current_time = time.time()
            recent_transcriptions = [
                t for t in self.transcription_buffer
                if current_time - t["timestamp"] <= 10.0
            ]
            
            recent_diarizations = [
                d for d in self.diarization_buffer
                if current_time - d["timestamp"] <= 10.0
            ]
            
            if not recent_transcriptions:
                return None
            
            # Merge transcriptions
            merged_text = " ".join([t["text"] for t in recent_transcriptions])
            avg_confidence = sum([t["confidence"] for t in recent_transcriptions]) / len(recent_transcriptions)
            
            # Merge speaker information
            all_speakers = []
            for d in recent_diarizations:
                all_speakers.extend(d.get("speakers", []))
            
            return {
                "transcription": recent_transcriptions,
                "speakers": all_speakers,
                "merged_text": merged_text,
                "confidence": avg_confidence,
                "chunk_count": len(recent_transcriptions)
            }
            
        except Exception as e:
            logger.error(f"Result merging error: {e}")
            return None
    
    async def get_final_results(self) -> Dict[str, Any]:
        """Get final processed results."""
        try:
            # Convert buffers to lists
            all_transcriptions = list(self.transcription_buffer)
            all_diarizations = list(self.diarization_buffer)
            
            # Merge all results
            final_text = " ".join([t["text"] for t in all_transcriptions])
            
            # Calculate overall confidence
            if all_transcriptions:
                avg_confidence = sum([t["confidence"] for t in all_transcriptions]) / len(all_transcriptions)
            else:
                avg_confidence = 0.0
            
            # Merge all speaker segments
            all_speakers = []
            for d in all_diarizations:
                all_speakers.extend(d.get("speakers", []))
            
            return {
                "final_transcription": final_text,
                "transcription_segments": all_transcriptions,
                "speaker_segments": all_speakers,
                "confidence": avg_confidence,
                "total_duration": self.total_duration,
                "chunk_count": len(all_transcriptions)
            }
            
        except Exception as e:
            logger.error(f"Final results error: {e}")
            return {}
    
    async def create_noise_profile(self, noise_audio: np.ndarray) -> bool:
        """Create noise profile for noise reduction."""
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.audio_processor.create_noise_profile,
                noise_audio
            )
            if result:
                logger.info("Noise profile created successfully")
            return result
        except Exception as e:
            logger.error(f"Failed to create noise profile: {e}")
            return False
    
    def update_processing_settings(self, settings: Dict[str, Any]) -> None:
        """Update advanced audio processing settings."""
        try:
            if "noise_reduction_factor" in settings:
                self.audio_processor.noise_reduction_factor = settings["noise_reduction_factor"]
            
            if "echo_decay" in settings:
                self.audio_processor.echo_decay = settings["echo_decay"]
            
            if "highpass_cutoff" in settings:
                self.audio_processor.highpass_cutoff = settings["highpass_cutoff"]
            
            logger.info(f"Updated processing settings: {settings}")
        except Exception as e:
            logger.error(f"Failed to update processing settings: {e}")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        try:
            stats = self.audio_processor.get_processing_stats()
            stats.update({
                "chunk_duration": self.chunk_duration,
                "overlap_duration": self.overlap_duration,
                "total_chunks_processed": self.chunk_counter,
                "total_duration": self.total_duration,
                "is_processing": self.is_processing,
                "transcription_buffer_size": len(self.transcription_buffer),
                "diarization_buffer_size": len(self.diarization_buffer)
            })
            return stats
        except Exception as e:
            logger.error(f"Failed to get processing stats: {e}")
            return {}
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            if self.executor:
                self.executor.shutdown(wait=False)
            logger.info("Streaming processor cleaned up")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")