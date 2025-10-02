"""
TranscrevAI Optimized - Subtitle Generator Module
Sistema avançado de geração de SRT com alinhamento inteligente transcription-diarization
"""

import asyncio
import gc
import os
import re
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json

# Import our optimized modules
from logging_setup import get_logger, log_performance
from resource_manager import get_resource_manager, ResourceStatus
from config import CONFIG

logger = get_logger("transcrevai.subtitle_generator")


class SRTGenerationError(Exception):
    """Custom exception for SRT generation errors"""
    def __init__(self, message: str, error_type: str = "unknown"):
        self.error_type = error_type
        super().__init__(f"[{error_type}] {message}")


class TimestampValidator:
    """
    Validate and normalize timestamps for SRT generation
    """
    
    @staticmethod
    def format_srt_time(seconds: float) -> str:
        """
        Format time in seconds to SRT time format (HH:MM:SS,mmm)
        
        Args:
            seconds: Time in seconds
            
        Returns:
            str: Formatted time string for SRT format
        """
        try:
            if not isinstance(seconds, (int, float)) or seconds < 0:
                seconds = 0.0
                
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            sec = seconds % 60
            
            return f"{hours:02d}:{minutes:02d}:{sec:06.3f}".replace(".", ",")[:12]
            
        except (TypeError, ValueError):
            logger.warning(f"Invalid time value: {seconds}, using 00:00:00,000")
            return "00:00:00,000"
    
    @staticmethod
    def validate_timestamp_sequence(segments: List[Dict]) -> List[Dict]:
        """
        Validate and fix timestamp sequences in segments
        
        Args:
            segments: List of segments with start/end times
            
        Returns:
            List of validated segments
        """
        if not segments:
            return segments
        
        validated_segments = []
        last_end_time = 0.0
        
        for i, segment in enumerate(segments):
            validated_segment = segment.copy()
            
            # Validate start time
            start_time = segment.get("start", 0.0)
            if start_time < last_end_time:
                start_time = last_end_time
                logger.debug(f"Adjusted start time for segment {i}: {start_time}")
            
            # Validate end time
            end_time = segment.get("end", start_time + 1.0)
            if end_time <= start_time:
                end_time = start_time + 1.0
                logger.debug(f"Adjusted end time for segment {i}: {end_time}")
            
            # Ensure minimum duration
            min_duration = CONFIG["subtitles"]["min_segment_duration"]
            if end_time - start_time < min_duration:
                end_time = start_time + min_duration
            
            validated_segment["start"] = start_time
            validated_segment["end"] = end_time
            validated_segments.append(validated_segment)
            
            last_end_time = end_time
        
        return validated_segments


class IntelligentAlignment:
    """
    Intelligent alignment of transcription data with speaker diarization
    Provides enhanced synchronization and speaker-text matching
    """
    
    def __init__(self):
        self.overlap_threshold = CONFIG["subtitles"]["overlap_threshold"]
        self.proximity_threshold = CONFIG["subtitles"]["proximity_threshold"]
        self.resource_manager = get_resource_manager()
    
    async def align_transcription_with_diarization(self,
                                                 transcription_data: List[Dict],
                                                 diarization_segments: List[Dict],
                                                 progress_callback: Optional[callable] = None) -> List[Dict]:
        """
        Intelligent alignment of transcription data with diarization segments
        
        Args:
            transcription_data: List of transcription segments with text and timing
            diarization_segments: List of speaker segments with timing and speaker IDs
            progress_callback: Optional progress callback
            
        Returns:
            List of aligned segments with speaker information
        """
        try:
            if progress_callback:
                await progress_callback(5, "Iniciando alinhamento inteligente...")
            
            # Validate input data
            if not transcription_data:
                logger.warning("No transcription data provided for alignment")
                return []
            
            if not diarization_segments:
                logger.warning("No diarization segments provided, using single speaker")
                return self._create_single_speaker_segments(transcription_data)
            
            if progress_callback:
                await progress_callback(15, "Preparando dados para alinhamento...")
            
            # Sort data by start time
            sorted_transcription = sorted(
                [t for t in transcription_data if isinstance(t, dict) and "start" in t and "text" in t],
                key=lambda x: x.get("start", 0)
            )
            
            sorted_diarization = sorted(
                [d for d in diarization_segments if isinstance(d, dict) and "start" in d and "end" in d],
                key=lambda x: x.get("start", 0)
            )
            
            logger.info(f"Aligning {len(sorted_transcription)} transcription segments with {len(sorted_diarization)} diarization segments")
            
            if progress_callback:
                await progress_callback(25, "Executando alinhamento avançado...")
            
            # Perform advanced alignment
            aligned_segments = await self._perform_advanced_alignment(
                sorted_transcription, 
                sorted_diarization, 
                progress_callback
            )
            
            if progress_callback:
                await progress_callback(80, "Mesclando segmentos consecutivos...")
            
            # Post-process: merge consecutive segments from same speaker
            if CONFIG["subtitles"]["merge_consecutive_speakers"]:
                aligned_segments = self._merge_consecutive_speaker_segments(aligned_segments)
            
            if progress_callback:
                await progress_callback(100, f"Alinhamento concluído! {len(aligned_segments)} segmentos criados")
            
            logger.info(f"Alignment complete: {len(aligned_segments)} segments created")
            return aligned_segments
            
        except Exception as e:
            logger.error(f"Transcription-diarization alignment failed: {e}")
            if progress_callback:
                await progress_callback(0, f"Erro no alinhamento: {str(e)}")
            
            # Fallback to simple alignment
            return self._simple_alignment_fallback(transcription_data, diarization_segments)
    
    async def _perform_advanced_alignment(self,
                                        sorted_transcription: List[Dict],
                                        sorted_diarization: List[Dict],
                                        progress_callback: Optional[callable] = None) -> List[Dict]:
        """Perform advanced alignment with multiple scoring factors"""
        combined_segments = []
        progress_step = 50 / max(len(sorted_diarization), 1)
        current_progress = 25
        
        for i, d_segment in enumerate(sorted_diarization):
            try:
                # Update progress
                if progress_callback and i % 5 == 0:
                    current_progress = min(75, 25 + (i * progress_step))
                    await progress_callback(int(current_progress), f"Processando segmento {i+1}/{len(sorted_diarization)}...")
                
                # Get diarization segment info
                d_start = d_segment["start"]
                d_end = d_segment["end"]
                speaker = d_segment.get("speaker", "Speaker_1")
                
                # Find overlapping transcription segments
                overlapping_segments = self._find_overlapping_segments(
                    sorted_transcription, d_start, d_end
                )
                
                if overlapping_segments:
                    # Create combined segment from overlapping transcripts
                    combined_segment = self._combine_overlapping_segments(
                        overlapping_segments, d_start, d_end, speaker
                    )
                    combined_segments.append(combined_segment)
                else:
                    # No overlapping transcription - create empty segment
                    combined_segments.append({
                        "start": d_start,
                        "end": d_end,
                        "speaker": speaker,
                        "text": "",
                        "confidence": 0.0,
                        "word_count": 0,
                        "alignment_method": "no_overlap"
                    })
                
                # Browser-safe: yield control periodically
                if i % 10 == 0:
                    await asyncio.sleep(0.001)  # Brief yield
                    
            except Exception as e:
                logger.warning(f"Failed to process diarization segment {i}: {e}")
                continue
        
        return combined_segments
    
    def _find_overlapping_segments(self,
                                 transcription_segments: List[Dict],
                                 d_start: float,
                                 d_end: float) -> List[Dict]:
        """Find transcription segments that overlap with diarization segment"""
        overlapping_segments = []
        
        # Extended window for boundary words
        extended_start = max(0, d_start - self.proximity_threshold)
        extended_end = d_end + self.proximity_threshold
        
        for t_segment in transcription_segments:
            t_start = t_segment.get("start", 0)
            t_end = t_segment.get("end", t_start + 1)
            
            # Calculate overlap
            overlap_start = max(extended_start, t_start)
            overlap_end = min(extended_end, t_end)
            overlap_duration = max(0, overlap_end - overlap_start)
            
            # Check various alignment conditions
            if self._should_include_segment(t_segment, d_start, d_end, overlap_duration):
                # Calculate alignment scores
                alignment_info = self._calculate_alignment_scores(
                    t_segment, d_start, d_end, overlap_duration
                )
                
                overlapping_segments.append({
                    "segment": t_segment,
                    "overlap_duration": overlap_duration,
                    "alignment_info": alignment_info
                })
        
        # Sort by alignment score (best matches first)
        overlapping_segments.sort(
            key=lambda x: x["alignment_info"]["weighted_score"], 
            reverse=True
        )
        
        return overlapping_segments
    
    def _should_include_segment(self,
                              t_segment: Dict,
                              d_start: float,
                              d_end: float,
                              overlap_duration: float) -> bool:
        """Determine if transcription segment should be included in diarization segment"""
        t_start = t_segment.get("start", 0)
        t_end = t_segment.get("end", t_start + 1)
        
        # 1. Direct overlap
        if overlap_duration > 0:
            return True
        
        # 2. Proximity to speaker boundaries
        is_near_start = abs(t_start - d_start) <= self.proximity_threshold
        is_near_end = abs(t_end - d_end) <= self.proximity_threshold
        if is_near_start or is_near_end:
            return True
        
        # 3. Contained within diarization segment
        if d_start <= t_start <= d_end or d_start <= t_end <= d_end:
            return True
        
        # 4. Text quality check - include meaningful text
        text = t_segment.get("text", "").strip()
        if len(text) >= CONFIG["subtitles"]["min_text_length"]:
            return True
        
        return False
    
    def _calculate_alignment_scores(self,
                                  t_segment: Dict,
                                  d_start: float,
                                  d_end: float,
                                  overlap_duration: float) -> Dict[str, float]:
        """Calculate multiple alignment scores for segment matching"""
        t_start = t_segment.get("start", 0)
        t_end = t_segment.get("end", t_start + 1)
        t_duration = max(t_end - t_start, 0.1)
        
        # 1. Overlap ratio score
        overlap_ratio = overlap_duration / t_duration
        
        # 2. Temporal proximity score
        center_d = (d_start + d_end) / 2
        center_t = (t_start + t_end) / 2
        distance = abs(center_d - center_t)
        max_distance = max(d_end - d_start, 1.0)
        proximity_score = max(0, 1.0 - distance / max_distance)
        
        # 3. Boundary alignment score
        is_near_start = abs(t_start - d_start) <= self.proximity_threshold
        is_near_end = abs(t_end - d_end) <= self.proximity_threshold
        boundary_score = 0.3 if (is_near_start or is_near_end) else 0
        
        # 4. Coverage score
        d_duration = max(d_end - d_start, 0.1)
        coverage_score = overlap_duration / d_duration
        
        # 5. Text quality score
        text = t_segment.get("text", "").strip()
        text_score = min(1.0, len(text) / 20)  # Normalize to 0-1
        
        # 6. Confidence score
        confidence = t_segment.get("confidence", 1.0)
        if isinstance(confidence, (int, float)):
            # Whisper uses no_speech_prob, so invert it
            if confidence <= 1.0:
                confidence_score = 1.0 - confidence
            else:
                confidence_score = 0.5  # Default
        else:
            confidence_score = 0.5
        
        # Combined weighted score
        weighted_score = (
            overlap_ratio * 0.30 +         # Direct overlap (30%)
            proximity_score * 0.25 +       # Temporal proximity (25%)
            boundary_score * 0.15 +        # Boundary alignment (15%)
            coverage_score * 0.15 +        # Coverage (15%)
            text_score * 0.10 +            # Text quality (10%)
            confidence_score * 0.05        # Confidence (5%)
        )
        
        return {
            "overlap_ratio": overlap_ratio,
            "proximity_score": proximity_score,
            "boundary_score": boundary_score,
            "coverage_score": coverage_score,
            "text_score": text_score,
            "confidence_score": confidence_score,
            "weighted_score": weighted_score,
            "is_boundary": is_near_start or is_near_end
        }
    
    def _combine_overlapping_segments(self,
                                    overlapping_segments: List[Dict],
                                    d_start: float,
                                    d_end: float,
                                    speaker: str) -> Dict[str, Any]:
        """Combine overlapping transcription segments into single diarization segment"""
        if not overlapping_segments:
            return {
                "start": d_start,
                "end": d_end,
                "speaker": speaker,
                "text": "",
                "confidence": 0.0,
                "word_count": 0,
                "alignment_method": "no_segments"
            }
        
        # Extract text and metadata
        segment_texts = []
        total_confidence = 0
        used_segments = set()
        
        for overlap_info in overlapping_segments:
            segment = overlap_info["segment"]
            segment_id = id(segment)  # Use object id as unique identifier
            
            if segment_id in used_segments:
                continue  # Skip if segment already used
            
            # Include high-quality segments
            alignment_info = overlap_info["alignment_info"]
            if alignment_info["weighted_score"] >= 0.3:  # Threshold for inclusion
                text = segment.get("text", "").strip()
                if text and len(text) >= 2:  # Avoid single characters
                    segment_texts.append({
                        "text": text,
                        "start": segment.get("start", 0),
                        "end": segment.get("end", 0),
                        "confidence": segment.get("confidence", 1.0),
                        "weighted_score": alignment_info["weighted_score"]
                    })
                    total_confidence += segment.get("confidence", 1.0)
                    used_segments.add(segment_id)
        
        # Sort segment texts by start time for natural order
        segment_texts.sort(key=lambda x: x["start"])
        
        # Combine text
        combined_text = " ".join(seg["text"] for seg in segment_texts).strip()
        
        # Calculate boundaries based on transcription if available
        if segment_texts:
            actual_start = min(seg["start"] for seg in segment_texts)
            actual_end = max(seg["end"] for seg in segment_texts)
            
            # Use transcription boundaries if they're close to diarization
            final_start = actual_start if abs(d_start - actual_start) <= 0.5 else d_start
            final_end = actual_end if abs(d_end - actual_end) <= 0.5 else d_end
        else:
            final_start = d_start
            final_end = d_end
        
        # Calculate average confidence
        avg_confidence = total_confidence / len(segment_texts) if segment_texts else 0.0
        
        return {
            "start": final_start,
            "end": final_end,
            "speaker": speaker,
            "text": combined_text if combined_text else "",
            "confidence": avg_confidence,
            "word_count": len(segment_texts),
            "alignment_method": "enhanced_temporal",
            "transcription_segments": len(segment_texts)
        }
    
    def _merge_consecutive_speaker_segments(self, segments: List[Dict]) -> List[Dict]:
        """Merge consecutive segments from the same speaker if they're close together"""
        if not segments:
            return segments
        
        merged = []
        current_segment = None
        max_gap = CONFIG["subtitles"]["max_merge_gap"]
        min_duration = CONFIG["subtitles"]["min_segment_duration"]
        
        for segment in segments:
            if current_segment is None:
                current_segment = segment.copy()
            elif (segment["speaker"] == current_segment["speaker"] and 
                  segment["start"] - current_segment["end"] <= max_gap):
                
                # Merge segments
                current_segment["end"] = segment["end"]
                current_segment["text"] = f"{current_segment['text']} {segment['text']}".strip()
                current_segment["confidence"] = min(current_segment["confidence"], segment["confidence"])
                current_segment["word_count"] = current_segment.get("word_count", 0) + segment.get("word_count", 0)
            else:
                # Different speaker or gap too large - finalize current segment
                if current_segment["end"] - current_segment["start"] >= min_duration:
                    merged.append(current_segment)
                current_segment = segment.copy()
        
        # Don't forget the last segment
        if current_segment and current_segment["end"] - current_segment["start"] >= min_duration:
            merged.append(current_segment)
        
        return merged
    
    def _create_single_speaker_segments(self, transcription_data: List[Dict]) -> List[Dict]:
        """Create segments with single speaker when no diarization is available"""
        segments = []
        
        for i, trans_seg in enumerate(transcription_data):
            if isinstance(trans_seg, dict) and trans_seg.get("text", "").strip():
                segments.append({
                    "start": trans_seg.get("start", 0),
                    "end": trans_seg.get("end", 0),
                    "speaker": "Speaker_1",
                    "text": trans_seg.get("text", ""),
                    "confidence": 1.0 - trans_seg.get("confidence", 0.0),  # Invert no_speech_prob
                    "word_count": 1,
                    "alignment_method": "single_speaker"
                })
        
        return segments
    
    def _simple_alignment_fallback(self,
                                 transcription_data: List[Dict],
                                 diarization_segments: List[Dict]) -> List[Dict]:
        """Simple fallback alignment when advanced method fails"""
        try:
            combined_segments = []
            
            for d_segment in diarization_segments:
                if isinstance(d_segment, dict) and "start" in d_segment and "end" in d_segment:
                    d_start = d_segment["start"]
                    d_end = d_segment["end"]
                    speaker = d_segment.get("speaker", "Speaker_1")
                    
                    # Find text that overlaps with this diarization segment
                    matched_texts = []
                    for t_segment in transcription_data:
                        if isinstance(t_segment, dict) and "text" in t_segment and "start" in t_segment:
                            t_start = t_segment["start"]
                            t_end = t_segment.get("end", t_start + 2.0)
                            
                            # Simple overlap check
                            if not (t_end <= d_start or t_start >= d_end):
                                text = t_segment["text"].strip()
                                if text:
                                    matched_texts.append(text)
                    
                    combined_segments.append({
                        "start": d_start,
                        "end": d_end,
                        "speaker": speaker,
                        "text": " ".join(matched_texts) if matched_texts else "",
                        "confidence": 0.5,  # Default confidence
                        "word_count": len(matched_texts),
                        "alignment_method": "simple_fallback"
                    })
            
            return combined_segments
            
        except Exception as e:
            logger.error(f"Simple alignment fallback failed: {e}")
            return []


class SubtitleGenerator:
    """
    Advanced subtitle generator with intelligent alignment and browser-safe processing
    """
    
    def __init__(self):
        self.timestamp_validator = TimestampValidator()
        self.intelligent_alignment = IntelligentAlignment()
        self.resource_manager = get_resource_manager()
        
        # Configuration
        self.output_dir = Path(CONFIG["paths"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("SubtitleGenerator initialized")
    
    async def generate_srt(self,
                          transcription_data: List[Dict],
                          diarization_segments: Optional[List[Dict]] = None,
                          output_path: Optional[str] = None,
                          progress_callback: Optional[callable] = None) -> str:
        """
        Generate SRT subtitle file from transcription and diarization data
        
        Args:
            transcription_data: List of transcription segments
            diarization_segments: Optional list of speaker diarization segments  
            output_path: Optional output file path
            progress_callback: Optional progress callback
            
        Returns:
            str: Path to generated SRT file
        """
        generation_start = time.time()
        
        try:
            if progress_callback:
                await progress_callback(5, "Validando dados de entrada...")
            
            # Validate input data
            if not transcription_data or not isinstance(transcription_data, list):
                raise SRTGenerationError("Missing or invalid transcription data", "invalid_input")
            
            logger.info(f"SRT Generation - Transcription data: {len(transcription_data)} items")
            if diarization_segments:
                logger.info(f"SRT Generation - Diarization segments: {len(diarization_segments)} items")
            
            if progress_callback:
                await progress_callback(10, "Preparando arquivo de saída...")
            
            # Prepare output path
            if not output_path:
                output_path = self._generate_output_path()
            else:
                # Ensure directory exists
                output_dir = Path(output_path).parent
                output_dir.mkdir(parents=True, exist_ok=True)
            
            if progress_callback:
                await progress_callback(20, "Alinhando transcrição com diarização...")
            
            # Align transcription with diarization
            if diarization_segments and len(diarization_segments) > 0:
                combined_segments = await self.intelligent_alignment.align_transcription_with_diarization(
                    transcription_data, diarization_segments, progress_callback
                )
            else:
                combined_segments = self._create_simple_segments(transcription_data)
            
            if progress_callback:
                await progress_callback(75, "Validando timestamps...")
            
            # Validate timestamps
            validated_segments = self.timestamp_validator.validate_timestamp_sequence(combined_segments)
            
            # Filter empty segments
            final_segments = [
                seg for seg in validated_segments 
                if seg.get("text", "").strip()
            ]
            
            if not final_segments:
                logger.warning("No valid segments created, creating fallback content")
                final_segments = self._create_fallback_segments(transcription_data)
            
            if progress_callback:
                await progress_callback(85, "Gerando arquivo SRT...")
            
            # Generate SRT content
            srt_content = await self._generate_srt_content(final_segments)
            
            if progress_callback:
                await progress_callback(95, "Salvando arquivo...")
            
            # Write to file
            await self._write_srt_file(srt_content, output_path)
            
            # Calculate metrics
            generation_time = time.time() - generation_start
            
            # Log performance
            log_performance(
                "SRT generation completed",
                duration=generation_time,
                segments_count=len(final_segments),
                speakers_count=len(set(seg.get("speaker", "Speaker_1") for seg in final_segments)),
                output_file=output_path
            )
            
            if progress_callback:
                await progress_callback(100, f"SRT gerado com sucesso! {len(final_segments)} segmentos")
            
            logger.info(f"SRT file generated: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"SRT generation failed: {e}")
            if progress_callback:
                await progress_callback(0, f"Erro na geração do SRT: {str(e)}")
            raise SRTGenerationError(f"Subtitle creation error: {e}", "generation_error")
    
    def _generate_output_path(self) -> str:
        """Generate unique output file path"""
        timestamp = int(time.time())
        unique_id = str(uuid.uuid4())[:8]
        filename = f"subtitles_{timestamp}_{unique_id}.srt"
        return str(self.output_dir / filename)
    
    def _create_simple_segments(self, transcription_data: List[Dict]) -> List[Dict]:
        """Create simple segments when no diarization is available"""
        segments = []
        
        for i, t_data in enumerate(transcription_data):
            if isinstance(t_data, dict) and t_data.get("text", "").strip():
                start_time = t_data.get("start", i * 2.0)
                end_time = t_data.get("end", start_time + 2.0)
                
                segments.append({
                    "start": start_time,
                    "end": end_time,
                    "speaker": "Speaker_1",
                    "text": t_data["text"].strip(),
                    "confidence": 1.0 - t_data.get("confidence", 0.0),  # Invert no_speech_prob
                    "word_count": 1,
                    "alignment_method": "simple"
                })
        
        return segments
    
    def _create_fallback_segments(self, transcription_data: List[Dict]) -> List[Dict]:
        """Create fallback segments from all available transcription text"""
        try:
            # Combine all available text
            all_texts = []
            for t in transcription_data:
                if isinstance(t, dict) and t.get("text", "").strip():
                    all_texts.append(t["text"].strip())
            
            if all_texts:
                combined_text = " ".join(all_texts)
                return [{
                    "start": 0.0,
                    "end": max(5.0, len(combined_text) * 0.1),  # Estimate duration
                    "speaker": "Speaker_1",
                    "text": combined_text,
                    "confidence": 0.5,
                    "word_count": len(all_texts),
                    "alignment_method": "fallback"
                }]
            else:
                return []
                
        except Exception as e:
            logger.error(f"Failed to create fallback segments: {e}")
            return []
    
    async def _generate_srt_content(self, segments: List[Dict]) -> str:
        """Generate SRT content from segments"""
        srt_lines = []
        
        for i, segment in enumerate(segments, 1):
            try:
                start_time = segment["start"]
                end_time = segment["end"]
                text = segment.get("text", "").strip()
                speaker = segment.get("speaker", "Speaker_1")
                
                if not text:
                    continue
                
                # Format SRT entry
                srt_lines.append(str(i))
                
                start_srt = self.timestamp_validator.format_srt_time(start_time)
                end_srt = self.timestamp_validator.format_srt_time(end_time)
                srt_lines.append(f"{start_srt} --> {end_srt}")
                
                # Format speaker and text
                if CONFIG["subtitles"]["include_speaker_labels"]:
                    srt_lines.append(f"[{speaker}] {text}")
                else:
                    srt_lines.append(text)
                
                srt_lines.append("")  # Empty line between segments
                
                # Browser-safe: yield control periodically
                if i % 50 == 0:
                    await asyncio.sleep(0.001)
                
            except Exception as e:
                logger.warning(f"Failed to format segment {i}: {e}")
                continue
        
        return "\n".join(srt_lines)
    
    async def _write_srt_file(self, srt_content: str, output_path: str):
        """Write SRT content to file with proper encoding"""
        try:
            # Check memory pressure before writing
            if self.resource_manager.is_memory_pressure_high():
                logger.warning("High memory pressure during SRT file writing")
                await self.resource_manager.perform_cleanup(aggressive=False)
            
            # Write file asynchronously
            import aiofiles
            
            async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
                await f.write(srt_content)
                await f.flush()
            
            # Validate file was created successfully
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise SRTGenerationError("Output file is invalid", "file_write_error")
            
        except ImportError:
            # Fallback to synchronous write
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)
                f.flush()
        except Exception as e:
            raise SRTGenerationError(f"Failed to write SRT file: {e}", "file_write_error")
    
    def validate_srt_format(self, file_path: str) -> bool:
        """Validate SRT file format"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            if len(lines) % 4 != 0:
                return False
            
            for i in range(0, len(lines), 4):
                # Check index number
                if not lines[i].strip().isdigit():
                    return False
                
                # Check time range format
                if "-->" not in lines[i + 1]:
                    return False
                
                # Check text line is not empty (for valid segments)
                if i + 2 < len(lines) and not lines[i + 2].strip():
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"SRT validation failed: {e}")
            return False
    
    async def get_srt_statistics(self, file_path: str) -> Dict[str, Any]:
        """Get statistics about generated SRT file"""
        try:
            if not os.path.exists(file_path):
                return {"error": "File not found"}
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            segments_count = len([line for line in lines if line.strip().isdigit()])
            
            # Extract speakers if labeled
            speakers = set()
            for line in lines:
                if line.strip().startswith('[') and ']' in line:
                    speaker = line.split(']')[0].replace('[', '').strip()
                    speakers.add(speaker)
            
            # Calculate duration from last timestamp
            time_lines = [line for line in lines if '-->' in line]
            total_duration = 0.0
            
            if time_lines:
                last_time_line = time_lines[-1]
                end_time_str = last_time_line.split('-->')[1].strip()
                # Parse time format HH:MM:SS,mmm
                time_parts = end_time_str.replace(',', '.').split(':')
                if len(time_parts) == 3:
                    hours = float(time_parts[0])
                    minutes = float(time_parts[1])
                    seconds = float(time_parts[2])
                    total_duration = hours * 3600 + minutes * 60 + seconds
            
            return {
                "file_size_bytes": os.path.getsize(file_path),
                "segments_count": segments_count,
                "speakers_detected": len(speakers),
                "speakers_list": list(speakers),
                "total_duration": total_duration,
                "is_valid_format": self.validate_srt_format(file_path),
                "creation_time": os.path.getctime(file_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to get SRT statistics: {e}")
            return {"error": str(e)}


# Utility functions for external use
async def quick_generate_srt(transcription_data: List[Dict],
                           diarization_segments: Optional[List[Dict]] = None,
                           output_path: Optional[str] = None) -> str:
    """Quick SRT generation function for simple use cases"""
    generator = SubtitleGenerator()
    return await generator.generate_srt(transcription_data, diarization_segments, output_path)


def validate_srt_file(file_path: str) -> bool:
    """Validate SRT file format"""
    generator = SubtitleGenerator()
    return generator.validate_srt_format(file_path)


async def get_srt_info(file_path: str) -> Dict[str, Any]:
    """Get information about an SRT file"""
    generator = SubtitleGenerator()
    return await generator.get_srt_statistics(file_path)