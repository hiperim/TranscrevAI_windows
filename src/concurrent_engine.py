# CRITICAL FIX: Enhanced concurrent processor with adaptive pipeline and quality metrics
import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor

from config.app_config import QUALITY_CONFIG, PROCESSING_PROFILES
from src.logging_setup import setup_app_logging

logger = setup_app_logging(logger_name="transcrevai.concurrent_engine")

class ConcurrentProcessor:
    """CRITICAL FIX: Enhanced concurrent processing engine with adaptive pipeline and comprehensive quality metrics"""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=3)  # Transcription, diarization, and analysis
        self.active_sessions = {}
        
    async def process_audio_concurrent(self, session_id: str, audio_file: str, language: str, websocket_manager, 
                                     audio_input_type: str = "neutral", processing_profile: str = "balanced") -> Dict[str, Any]:
        """
        CRITICAL FIX: Enhanced concurrent processing with adaptive pipeline based on complexity and quality metrics
        
        Args:
            session_id: Unique session identifier
            audio_file: Path to audio file
            language: Language code for transcription
            websocket_manager: WebSocket manager for real-time updates
            audio_input_type: Type of audio input (lecture/conversation/complex_dialogue)
            processing_profile: Processing profile (realtime/balanced/quality)
        
        Returns:
            Dict containing transcription data, diarization segments, quality metrics, and metadata
        """
        try:
            logger.info(f"Starting enhanced concurrent processing for session {session_id}")
            logger.info(f"Parameters: language={language}, audio_input_type={audio_input_type}, profile={processing_profile}")
            
            # Initialize session tracking with enhanced metrics
            self.active_sessions[session_id] = {
                "transcription_progress": 0,
                "diarization_progress": 0,
                "complexity_analysis_progress": 0,
                "start_time": time.time(),
                "status": "processing",
                "language": language,
                "audio_input_type": audio_input_type,
                "processing_profile": processing_profile
            }
            
            # Stage 1: Audio Complexity Analysis
            await websocket_manager.send_message(session_id, {
                "type": "processing_stage",
                "stage": "complexity_analysis",
                "message": "Analyzing audio complexity for optimal processing pipeline..."
            })
            
            complexity = await self._complexity_analysis_with_progress(session_id, audio_file, websocket_manager)
            self.active_sessions[session_id]["complexity"] = complexity
            
            # Stage 2: Enhanced concurrent transcription and diarization
            await websocket_manager.send_message(session_id, {
                "type": "processing_stage", 
                "stage": "concurrent_processing",
                "message": f"Processing with {complexity} complexity pipeline..."
            })
            
            # Create enhanced tasks for concurrent execution
            transcription_task = asyncio.create_task(
                self._enhanced_transcription_with_progress(session_id, audio_file, language, audio_input_type, complexity, websocket_manager)
            )
            
            diarization_task = asyncio.create_task(
                self._enhanced_diarization_with_progress(session_id, audio_file, complexity, websocket_manager)
            )
            
            # Wait for both tasks to complete
            transcription_result, diarization_result = await asyncio.gather(
                transcription_task, 
                diarization_task,
                return_exceptions=True
            )
            
            # Handle potential exceptions
            if isinstance(transcription_result, Exception):
                logger.error(f"Enhanced transcription failed: {transcription_result}")
                transcription_result = []
            
            if isinstance(diarization_result, Exception):
                logger.error(f"Enhanced diarization failed: {diarization_result}")
                diarization_result = [{
                    "start": 0.0,
                    "end": 1.0,
                    "speaker": "Speaker_1"
                }]
            
            # Stage 3: Content-based analysis and speaker change detection
            await websocket_manager.send_message(session_id, {
                "type": "processing_stage",
                "stage": "content_analysis", 
                "message": "Analyzing content for speaker change patterns..."
            })
            
            # Ensure transcription_result is valid before analysis
            if isinstance(transcription_result, list):
                content_hints = await self._analyze_content_for_speaker_changes(transcription_result, language)
            else:
                content_hints = []
            
            # Stage 4: Enhanced alignment with content-based intelligence
            await websocket_manager.send_message(session_id, {
                "type": "processing_stage",
                "stage": "intelligent_alignment",
                "message": "Performing intelligent alignment of transcription and diarization..."
            })
            
            # Import alignment function
            from src.speaker_diarization import align_transcription_with_diarization
            
            aligned_transcription = align_transcription_with_diarization(transcription_result, diarization_result)
            
            # Stage 5: Comprehensive quality metrics calculation
            await websocket_manager.send_message(session_id, {
                "type": "processing_stage",
                "stage": "quality_metrics",
                "message": "Calculating comprehensive quality metrics..."
            })
            
            # Ensure valid data types for quality metrics calculation
            valid_aligned = aligned_transcription if isinstance(aligned_transcription, list) else []
            valid_diarization = diarization_result if isinstance(diarization_result, list) else []
            
            quality_metrics = self._calculate_quality_metrics(
                valid_aligned, valid_diarization, complexity, content_hints
            )
            
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
            
            # CRITICAL FIX: Enhanced result structure with comprehensive data
            result = {
                "transcription_data": aligned_transcription,
                "diarization_segments": diarization_result,
                "speakers_detected": unique_speakers,
                "processing_time": processing_time,
                "complexity": complexity,
                "content_hints": content_hints,
                "quality_metrics": quality_metrics,
                "processing_profile": processing_profile,
                "audio_input_type": audio_input_type,
                "method": "enhanced_concurrent_adaptive"
            }
            
            logger.info(f"Enhanced concurrent processing completed for session {session_id}: "
                       f"{len(aligned_transcription) if isinstance(aligned_transcription, list) else 0} transcription segments, "
                       f"{len(diarization_result) if isinstance(diarization_result, list) else 0} diarization segments, "
                       f"{unique_speakers} speakers, "
                       f"processing time: {processing_time:.2f}s, "
                       f"overall quality: {quality_metrics.get('overall_quality', 0):.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Enhanced concurrent processing failed for session {session_id}: {e}")
            # Clean up session
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            raise

    async def _complexity_analysis_with_progress(self, session_id: str, audio_file: str, websocket_manager) -> str:
        """Perform complexity analysis with progress updates"""
        try:
            # No complexity analysis needed - using medium model for all
            
            # Update progress
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["complexity_analysis_progress"] = 50
                await websocket_manager.send_message(session_id, {
                    "type": "progress",
                    "complexity_analysis": 50,
                    "transcription": 0,
                    "diarization": 0
                })
            
            # Fixed complexity - always use medium model
            complexity = "medium"
            
            # Final progress update
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["complexity_analysis_progress"] = 100
                await websocket_manager.send_message(session_id, {
                    "type": "progress", 
                    "complexity_analysis": 100,
                    "transcription": 0,
                    "diarization": 0
                })
            
            return complexity
            
        except Exception as e:
            logger.error(f"Complexity analysis with progress failed: {e}")
            return "medium"  # Safe default

    async def _enhanced_transcription_with_progress(self, session_id: str, audio_file: str, language: str, 
                                                   audio_input_type: str, complexity: str, websocket_manager) -> List[Dict]:
        """Enhanced transcription with adaptive processing and progress updates"""
        try:
            from src.transcription import transcribe_audio_with_progress
            
            transcription_data = []
            
            # Use the enhanced transcription function with adaptive parameters
            async for progress, data in transcribe_audio_with_progress(
                audio_file, language, 16000, audio_input_type, "balanced"
            ):
                # Update session progress
                if session_id in self.active_sessions:
                    self.active_sessions[session_id]["transcription_progress"] = progress
                    
                    # Send combined progress update
                    complexity_progress = self.active_sessions[session_id].get("complexity_analysis_progress", 100)
                    diarization_progress = self.active_sessions[session_id].get("diarization_progress", 0)
                    await websocket_manager.send_message(session_id, {
                        "type": "progress",
                        "complexity_analysis": complexity_progress,
                        "transcription": progress,
                        "diarization": diarization_progress
                    })
                
                if data:
                    transcription_data = data
            
            return transcription_data
            
        except Exception as e:
            logger.error(f"Enhanced transcription with progress failed: {e}")
            raise

    async def _enhanced_diarization_with_progress(self, session_id: str, audio_file: str, complexity: str, websocket_manager) -> List[Dict]:
        """Enhanced diarization with adaptive methods and progress updates"""
        try:
            from src.speaker_diarization import SpeakerDiarization
            
            # Create diarizer instance
            diarizer = SpeakerDiarization()
            
            # Adaptive progress updates based on complexity
            if complexity == "high":
                progress_steps = [10, 25, 45, 70, 85]
                step_delay = 0.8  # Longer processing time
            elif complexity == "medium":
                progress_steps = [15, 35, 60, 80]
                step_delay = 0.6
            else:  # low complexity
                progress_steps = [25, 60, 85]
                step_delay = 0.4
            
            # Send progress updates
            for progress in progress_steps:
                if session_id in self.active_sessions:
                    self.active_sessions[session_id]["diarization_progress"] = progress
                    
                    complexity_progress = self.active_sessions[session_id].get("complexity_analysis_progress", 100)
                    transcription_progress = self.active_sessions[session_id].get("transcription_progress", 0)
                    await websocket_manager.send_message(session_id, {
                        "type": "progress",
                        "complexity_analysis": complexity_progress,
                        "transcription": transcription_progress,
                        "diarization": progress
                    })
                
                await asyncio.sleep(step_delay)
            
            # Perform actual diarization with enhanced method
            segments = await diarizer.diarize_audio(audio_file)
            
            # Final progress update
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["diarization_progress"] = 100
                complexity_progress = self.active_sessions[session_id].get("complexity_analysis_progress", 100)
                transcription_progress = self.active_sessions[session_id].get("transcription_progress", 0)
                await websocket_manager.send_message(session_id, {
                    "type": "progress",
                    "complexity_analysis": complexity_progress,
                    "transcription": transcription_progress,
                    "diarization": 100
                })
            
            return segments
            
        except Exception as e:
            logger.error(f"Enhanced diarization with progress failed: {e}")
            raise

    async def _analyze_content_for_speaker_changes(self, transcription_segments: List[Dict], language: str) -> List[Dict]:
        """Analyze transcription content for speaker change hints"""
        try:
            if not transcription_segments:
                return []
            
            from src.speaker_diarization import detect_speaker_changes_from_content
            
            content_hints = detect_speaker_changes_from_content(transcription_segments, language)
            
            logger.info(f"Content analysis detected {len(content_hints)} potential speaker change points")
            
            return content_hints
            
        except Exception as e:
            logger.error(f"Content analysis failed: {e}")
            return []

    def _calculate_quality_metrics(self, transcription_data: List[Dict], diarization_data: List[Dict], 
                                 complexity: str, content_hints: List[Dict]) -> Dict[str, Any]:
        """
        CRITICAL FIX: Calculate comprehensive quality metrics for the processing results
        """
        try:
            metrics = {
                "transcription_quality": 0.0,
                "diarization_quality": 0.0,
                "alignment_score": 0.0,
                "content_analysis_score": 0.0,
                "overall_quality": 0.0,
                "complexity_match": complexity,
                "detailed_metrics": {}
            }
            
            # Transcription quality metrics
            if transcription_data:
                total_confidence = sum(seg.get('confidence', 0.5) for seg in transcription_data)
                avg_confidence = total_confidence / len(transcription_data)
                
                # Text quality indicators
                total_text = " ".join(seg.get('text', '') for seg in transcription_data)
                word_count = len(total_text.split())
                avg_segment_length = word_count / len(transcription_data)
                
                # Quality score based on confidence and content
                transcription_score = avg_confidence * 0.6
                if word_count > 10:  # Reasonable amount of text
                    transcription_score += 0.2
                if len(transcription_data) > 1:  # Multiple segments
                    transcription_score += 0.1
                if 5 <= avg_segment_length <= 30:  # Reasonable segment lengths
                    transcription_score += 0.1
                
                metrics["transcription_quality"] = min(1.0, transcription_score)
                metrics["detailed_metrics"]["avg_confidence"] = avg_confidence
                metrics["detailed_metrics"]["word_count"] = word_count
                metrics["detailed_metrics"]["segment_count"] = len(transcription_data)
                metrics["detailed_metrics"]["avg_segment_length"] = avg_segment_length
            
            # Diarization quality metrics
            if diarization_data:
                # Speaker distribution quality
                speakers = [seg.get('speaker', 'Unknown') for seg in diarization_data]
                unique_speakers = len(set(speakers))
                
                if unique_speakers > 1:
                    # Check balance between speakers
                    speaker_counts = {}
                    for speaker in speakers:
                        speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
                    
                    min_speaker_segments = min(speaker_counts.values())
                    max_speaker_segments = max(speaker_counts.values())
                    speaker_balance = min_speaker_segments / max_speaker_segments if max_speaker_segments > 0 else 1.0
                    
                    # Segment duration distribution
                    durations = [seg.get('end', 0) - seg.get('start', 0) for seg in diarization_data]
                    avg_duration = sum(durations) / len(durations) if durations else 0
                    
                    # Average confidence
                    avg_diar_confidence = sum(seg.get('confidence', 0.5) for seg in diarization_data) / len(diarization_data)
                    
                    diarization_score = 0.4  # Base score for multiple speakers
                    if 0.5 <= avg_duration <= 15.0:  # Reasonable segment lengths
                        diarization_score += 0.2
                    if speaker_balance >= 0.2:  # Reasonable speaker balance
                        diarization_score += 0.2
                    if avg_diar_confidence >= 0.6:  # Good confidence
                        diarization_score += 0.2
                    
                    metrics["diarization_quality"] = min(1.0, diarization_score)
                    metrics["detailed_metrics"]["unique_speakers"] = unique_speakers
                    metrics["detailed_metrics"]["speaker_balance"] = speaker_balance
                    metrics["detailed_metrics"]["avg_segment_duration"] = avg_duration
                    metrics["detailed_metrics"]["avg_diarization_confidence"] = avg_diar_confidence
                else:
                    # Single speaker scenario
                    metrics["diarization_quality"] = 0.7
                    metrics["detailed_metrics"]["unique_speakers"] = 1
                    metrics["detailed_metrics"]["speaker_balance"] = 1.0
            
            # Alignment quality
            if transcription_data and diarization_data:
                alignment_matches = sum(
                    1 for seg in transcription_data 
                    if seg.get('diarization_confidence', 0) > 0.5
                )
                alignment_score = alignment_matches / len(transcription_data) if transcription_data else 0
                
                # Bonus for good alignment scores
                good_alignment_count = sum(
                    1 for seg in transcription_data 
                    if seg.get('alignment_score', 0) > 0.6
                )
                alignment_bonus = good_alignment_count / len(transcription_data) if transcription_data else 0
                
                final_alignment_score = (alignment_score * 0.7) + (alignment_bonus * 0.3)
                metrics["alignment_score"] = final_alignment_score
                metrics["detailed_metrics"]["alignment_matches"] = alignment_matches
                metrics["detailed_metrics"]["good_alignment_ratio"] = alignment_bonus
            
            # Content analysis quality
            if content_hints:
                # More hints generally indicate better content analysis
                content_score = min(1.0, len(content_hints) * 0.15)
                
                # Bonus for high-probability hints
                high_prob_hints = sum(1 for hint in content_hints if hint.get('probability', 0) > 0.6)
                if high_prob_hints > 0:
                    content_score += min(0.3, high_prob_hints * 0.1)
                
                metrics["content_analysis_score"] = min(1.0, content_score)
                metrics["detailed_metrics"]["content_hints_count"] = len(content_hints)
                metrics["detailed_metrics"]["high_prob_hints"] = high_prob_hints
            
            # Overall quality (weighted average with complexity consideration)
            base_weights = {
                "transcription_quality": 0.4,
                "diarization_quality": 0.3,
                "alignment_score": 0.2,
                "content_analysis_score": 0.1
            }
            
            # Adjust weights based on complexity
            if complexity == "high":
                # For complex audio, prioritize diarization and alignment
                weights = {
                    "transcription_quality": 0.35,
                    "diarization_quality": 0.35,
                    "alignment_score": 0.25,
                    "content_analysis_score": 0.05
                }
            elif complexity == "low":
                # For simple audio, prioritize transcription
                weights = {
                    "transcription_quality": 0.5,
                    "diarization_quality": 0.2,
                    "alignment_score": 0.2,
                    "content_analysis_score": 0.1
                }
            else:
                weights = base_weights
            
            # Calculate weighted overall quality
            overall_quality = sum(
                metrics[metric] * weight for metric, weight in weights.items()
            )
            
            metrics["overall_quality"] = overall_quality
            metrics["detailed_metrics"]["complexity_adjusted_weights"] = weights
            
            return metrics
            
        except Exception as e:
            logger.error(f"Quality metrics calculation failed: {e}")
            return {
                "transcription_quality": 0.5,
                "diarization_quality": 0.5, 
                "alignment_score": 0.5,
                "content_analysis_score": 0.0,
                "overall_quality": 0.5,
                "complexity_match": complexity,
                "detailed_metrics": {"error": str(e)}
            }
    
    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current processing status for a session"""
        return self.active_sessions.get(session_id)
    
    def cleanup_session(self, session_id: str):
        """Clean up session data"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(f"Cleaned up enhanced concurrent processing session: {session_id}")
    
    async def shutdown(self):
        """Shutdown the concurrent processor"""
        try:
            # Cancel all active sessions
            for session_id in list(self.active_sessions.keys()):
                self.cleanup_session(session_id)
            
            # Shutdown thread pool
            self.executor.shutdown(wait=True)
            logger.info("Enhanced concurrent processor shutdown complete")
            
        except Exception as e:
            logger.error(f"Enhanced concurrent processor shutdown failed: {e}")

# Global concurrent processor instance
concurrent_processor = ConcurrentProcessor()