"""
Enhanced Subtitle Generator - Fixed Type Hints and Improved Implementation

All type hints properly specified and segment handling optimized for production use.
"""

import aiofiles
import aiofiles.tempfile
import logging
import os
import time
import uuid
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from .file_manager import FileManager
from .logging_setup import setup_app_logging
import numpy as np

# Use proper logging setup
logger = setup_app_logging(logger_name="transcrevai.subtitle_generator")

def format_time_srt(seconds: float) -> str:
    """Convert seconds to SRT time format (HH:MM:SS,mmm) with type safety"""
    try:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    except (TypeError, ValueError, OverflowError):
        logger.warning(f"Invalid time value for SRT formatting: {seconds}")
        return "00:00:00,000"

def generate_srt_simple(transcription_segments: List[Dict[str, Any]]) -> str:
    """
    Generate SRT content from transcription segments - ENHANCED VERSION
    
    Args:
        transcription_segments: List of segments with start, end, text, and speaker
        
    Returns:
        str: SRT formatted content
        
    Type hints fixed: Explicit List[Dict[str, Any]] for segments
    """
    logger.info(f"generate_srt_simple called with: {type(transcription_segments)}")
    
    if not transcription_segments:
        logger.warning("Empty transcription_segments received")
        return ""
    
    if not isinstance(transcription_segments, list):
        logger.error(f"Expected list, got {type(transcription_segments)}")
        return ""
    
    logger.info(f"Processing {len(transcription_segments)} segments")
    
    srt_content: List[str] = []
    segment_counter = 0
    
    for i, segment in enumerate(transcription_segments, 1):
        if not isinstance(segment, dict):
            logger.warning(f"Segment {i} is not a dict, skipping: {type(segment)}")
            continue
            
        start_time = segment.get('start', 0.0)
        end_time = segment.get('end', 0.0)
        text = segment.get('text', '').strip()
        speaker = segment.get('speaker', 'Speaker_1')
        
        # Validate segment data
        if not isinstance(start_time, (int, float)) or not isinstance(end_time, (int, float)):
            logger.warning(f"Invalid time values in segment {i}, skipping")
            continue
            
        if end_time <= start_time:
            logger.warning(f"Invalid time range in segment {i}: {start_time}-{end_time}")
            continue
        
        logger.debug(f"Segment {i}: {start_time:.2f}s-{end_time:.2f}s, '{text[:50]}...', {speaker}")
        
        if not text:
            logger.debug(f"Skipping segment {i} - empty text")
            continue
        
        segment_counter += 1
        
        # Format: segment number
        srt_content.append(str(segment_counter))
        
        # Format: start --> end (with type-safe time formatting)
        start_srt = format_time_srt(float(start_time))
        end_srt = format_time_srt(float(end_time))
        srt_content.append(f"{start_srt} --> {end_srt}")
        
        # Format: speaker: text (with proper escaping)
        clean_text = text.replace('\n', ' ').replace('\r', '').strip()
        srt_content.append(f"{speaker}: {clean_text}")
        
        # Empty line between segments
        srt_content.append("")
    
    result = "\n".join(srt_content)
    logger.info(f"Generated SRT: {len(result)} characters, {segment_counter} segments")
    return result

async def generate_srt(
    transcription_data: List[Dict[str, Any]], 
    diarization_segments: Optional[List[Dict[str, Any]]], 
    filename: str = "output.srt"
) -> Optional[str]:
    """
    Generate SRT subtitle file from transcription and diarization data
    
    Args:
        transcription_data: List of transcription segments with text and timing
        diarization_segments: List of speaker diarization segments (can be None)
        filename: Output filename for the SRT file
        
    Returns:
        str: Path to the generated SRT file, or None if failed
        
    Enhanced error handling and type safety
    """
    temp_path: Optional[str] = None
    
    try:
        # Validate input parameters with explicit type checking
        if not transcription_data or not isinstance(transcription_data, list):
            raise ValueError("Invalid transcription data: must be non-empty list")
        
        # Handle None diarization_segments gracefully
        if diarization_segments is None:
            diarization_segments = []
        elif not isinstance(diarization_segments, list):
            raise ValueError("Diarization segments must be a list or None")
        
        # Enhanced logging
        logger.info(f"SRT Generation - Transcription: {len(transcription_data)} items")
        logger.info(f"SRT Generation - Diarization: {len(diarization_segments)} items")
        
        # Determine output path with proper validation
        output_path = _determine_output_path(filename)
        
        combined_segments: List[Dict[str, Any]] = []
        
        # Use aiofiles.tempfile for async temp file handling
        async with aiofiles.tempfile.NamedTemporaryFile('w', delete=False, encoding='utf-8') as tmp:
            temp_path = tmp.name
            
            try:
                # Enhanced segment processing
                if diarization_segments:
                    combined_segments = await _align_transcription_with_diarization(
                        transcription_data, diarization_segments
                    )
                else:
                    # Process transcription data without diarization
                    combined_segments = _process_transcription_only(transcription_data)
                
                # Filter and validate segments
                valid_segments = _validate_segments(combined_segments)
                
                if not valid_segments:
                    logger.warning("No valid segments after processing, creating fallback")
                    valid_segments = _create_fallback_segments(transcription_data)
                
                # Write SRT content with proper formatting
                await _write_srt_content(tmp, valid_segments)
                
            except Exception as e:
                logger.error(f"Error processing segments: {e}")
                raise RuntimeError(f"Failed to process SRT content: {str(e)}") from e
        
        # Move temp file to final location atomically
        try:
            os.replace(temp_path, output_path)
            temp_path = None  # Successfully moved, don't cleanup
        except OSError as e:
            logger.error(f"Failed to move temp file to final location: {e}")
            raise RuntimeError(f"Failed to write SRT file to {output_path}: {e}") from e
        
        logger.info(f"SRT file generated successfully: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"SRT generation failed: {str(e)}")
        
        # Enhanced cleanup with explicit temp_path handling
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as cleanup_error:
                logger.warning(f"Temp file cleanup failed: {cleanup_error}")
        
        raise RuntimeError(f"Subtitle creation error: {str(e)}") from e

def _determine_output_path(filename: str) -> str:
    """Determine and validate output path for SRT file"""
    try:
        dirname = str(Path(filename).parent)
        
        if dirname and dirname != "." and dirname != "":
            # Use provided path
            Path(dirname).mkdir(parents=True, exist_ok=True)
            return filename
        else:
            # Use default directory with unique naming
            output_dir = FileManager.get_data_path("transcripts")
            base_name = Path(filename).stem
            unique_id = str(uuid.uuid4())[:8]
            timestamp = int(time.time())
            unique_name = f"{base_name}_{timestamp}_{unique_id}.srt"
            return str(Path(output_dir) / unique_name)
            
    except Exception as e:
        logger.error(f"Failed to determine output path: {e}")
        raise RuntimeError(f"Cannot determine output path: {e}") from e

def _process_transcription_only(transcription_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process transcription data without diarization"""
    segments: List[Dict[str, Any]] = []
    
    for i, t_data in enumerate(transcription_data):
        if not isinstance(t_data, dict):
            continue
            
        text = t_data.get("text", "").strip()
        if not text:
            continue
            
        start_time = t_data.get("start", i * 2.0)
        end_time = t_data.get("end", start_time + 2.0)
        
        segments.append({
            "start": float(start_time),
            "end": float(end_time),
            "speaker": "Speaker_1",
            "text": text,
            "confidence": t_data.get("confidence", 0.8)
        })
    
    return segments

def _validate_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate and filter segments"""
    valid_segments: List[Dict[str, Any]] = []
    
    for segment in segments:
        if not isinstance(segment, dict):
            continue
            
        text = segment.get("text", "").strip()
        start = segment.get("start")
        end = segment.get("end")
        
        # Validate required fields
        if not text:
            continue
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        if end <= start:
            continue
            
        valid_segments.append(segment)
    
    return valid_segments

def _create_fallback_segments(transcription_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create fallback segments when processing fails"""
    all_text = " ".join([
        t.get("text", "") for t in transcription_data
        if isinstance(t, dict) and t.get("text", "").strip()
    ])
    
    if all_text.strip():
        return [{
            "start": 0.0,
            "end": 5.0,
            "speaker": "Speaker_1",
            "text": all_text.strip(),
            "confidence": 0.5
        }]
    
    return []

async def _write_srt_content(tmp: Any, segments: List[Dict[str, Any]]) -> None:
    """Write SRT content to temporary file"""
    for idx, segment in enumerate(segments, 1):
        start_time = format_time_srt(segment['start'])
        end_time = format_time_srt(segment['end'])
        speaker = segment.get('speaker', 'Speaker_1')
        text = segment['text']
        
        await tmp.write(f"{idx}\n")
        await tmp.write(f"{start_time} --> {end_time}\n")
        await tmp.write(f"{speaker}: {text}\n\n")
    
    await tmp.flush()

def format_time(seconds: float) -> str:
    """
    Format time in seconds to SRT time format (HH:MM:SS,mmm)
    
    Args:
        seconds: Time in seconds (float)
        
    Returns:
        str: Formatted time string for SRT format
    """
    try:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        sec = seconds % 60
        return f"{hours:02}:{minutes:02}:{sec:06.3f}".replace('.', ',')[:12]
    except (TypeError, ValueError):
        raise ValueError(f"Invalid time value: {seconds}")

def validate_srt_format(content: str) -> bool:
    """Validate SRT file format with enhanced checks"""
    if not content.strip():
        return False
        
    lines = content.split('\n')
    
    # Basic structure validation
    segment_count = 0
    i = 0
    
    while i < len(lines):
        # Skip empty lines
        if not lines[i].strip():
            i += 1
            continue
            
        # Check segment number
        if not lines[i].strip().isdigit():
            return False
        
        # Check time range format
        i += 1
        if i >= len(lines) or '-->' not in lines[i]:
            return False
            
        # Validate time format
        time_line = lines[i].strip()
        time_parts = time_line.split(' --> ')
        if len(time_parts) != 2:
            return False
            
        for time_part in time_parts:
            if not _validate_time_format(time_part):
                return False
        
        # Check text line exists and is not empty
        i += 1
        if i >= len(lines) or not lines[i].strip():
            return False
            
        segment_count += 1
        i += 1
    
    return segment_count > 0

def _validate_time_format(time_str: str) -> bool:
    """Validate SRT time format HH:MM:SS,mmm"""
    pattern = r'^\d{2}:\d{2}:\d{2},\d{3}$'
    return bool(re.match(pattern, time_str))

async def _align_transcription_with_diarization(
    transcription_data: List[Dict[str, Any]], 
    diarization_segments: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Enhanced alignment of transcription data with diarization segments
    
    This function provides sophisticated synchronization by:
    1. Using overlap-based matching with weighted scoring
    2. Handling partial overlaps intelligently  
    3. Grouping words/phrases by speaker
    4. Maintaining temporal coherence
    """
    try:
        combined_segments: List[Dict[str, Any]] = []
        
        # Sort both datasets by start time for efficient processing
        sorted_transcription = sorted([
            t for t in transcription_data 
            if isinstance(t, dict) and "start" in t and "text" in t
        ], key=lambda x: x.get("start", 0))
        
        sorted_diarization = sorted([
            d for d in diarization_segments 
            if isinstance(d, dict) and "start" in d and "end" in d  
        ], key=lambda x: x.get("start", 0))
        
        if not sorted_diarization:
            logger.warning("No valid diarization segments for alignment")
            return _process_transcription_only(transcription_data)
        
        logger.info(f"Aligning {len(sorted_transcription)} transcription segments with {len(sorted_diarization)} diarization segments")
        
        used_transcription_ids: set = set()
        
        # Process each diarization segment
        for d_segment in sorted_diarization:
            d_start = float(d_segment["start"])
            d_end = float(d_segment["end"])
            speaker = d_segment.get("speaker", "Speaker_1")
            
            # Find overlapping transcription segments with enhanced matching
            overlapping_segments: List[Dict[str, Any]] = []
            
            for t_idx, t_segment in enumerate(sorted_transcription):
                if t_idx in used_transcription_ids:
                    continue
                    
                t_start = float(t_segment.get("start", 0))
                t_end = float(t_segment.get("end", t_start + 1))
                
                # Calculate overlap with extended tolerance window
                overlap_start = max(d_start - 0.2, t_start)  # 200ms tolerance
                overlap_end = min(d_end + 0.2, t_end)
                overlap_duration = max(0, overlap_end - overlap_start)
                
                # Multiple matching criteria
                t_duration = max(t_end - t_start, 0.1)
                overlap_ratio = overlap_duration / t_duration
                
                # Temporal proximity scoring
                center_d = (d_start + d_end) / 2
                center_t = (t_start + t_end) / 2
                distance = abs(center_d - center_t)
                proximity_score = max(0, 1.0 - (distance / max(d_end - d_start, 1.0)))
                
                # Combined scoring
                confidence = t_segment.get("confidence", 1.0)
                weighted_score = (overlap_ratio * 0.6 + proximity_score * 0.4) * confidence
                
                # Include if meets threshold
                if weighted_score >= 0.3 or overlap_duration > 0:
                    overlapping_segments.append({
                        "segment": t_segment,
                        "weighted_score": weighted_score,
                        "transcription_idx": t_idx
                    })
            
            # Sort by score and combine text
            overlapping_segments.sort(key=lambda x: x["weighted_score"], reverse=True)
            
            segment_texts: List[str] = []
            total_confidence = 0.0
            
            for overlap_info in overlapping_segments:
                if overlap_info["weighted_score"] >= 0.25:  # Quality threshold
                    segment = overlap_info["segment"]
                    text = segment["text"].strip()
                    
                    if text and len(text) > 1:
                        segment_texts.append(text)
                        total_confidence += segment.get("confidence", 1.0)
                        used_transcription_ids.add(overlap_info["transcription_idx"])
            
            # Create combined segment if we have text
            if segment_texts:
                combined_text = " ".join(segment_texts)
                avg_confidence = total_confidence / len(segment_texts) if segment_texts else 0
                
                combined_segments.append({
                    "start": d_start,
                    "end": d_end,
                    "speaker": speaker,
                    "text": combined_text,
                    "confidence": avg_confidence,
                    "word_count": len(segment_texts),
                    "alignment_method": "enhanced_temporal"
                })
                
                logger.debug(f"Aligned segment: {speaker} ({d_start:.2f}-{d_end:.2f}s): '{combined_text[:50]}...'")
            else:
                # Keep timing but mark as no speech detected
                combined_segments.append({
                    "start": d_start,
                    "end": d_end,
                    "speaker": speaker,
                    "text": "[No speech detected]",
                    "confidence": 0.0,
                    "word_count": 0
                })
        
        # Post-process: merge consecutive segments from same speaker
        merged_segments = _merge_consecutive_speaker_segments(combined_segments)
        
        logger.info(f"Alignment complete: {len(combined_segments)} segments -> {len(merged_segments)} after merging")
        return merged_segments
        
    except Exception as e:
        logger.error(f"Enhanced alignment failed: {e}")
        return _simple_alignment_fallback(transcription_data, diarization_segments)

def _merge_consecutive_speaker_segments(
    segments: List[Dict[str, Any]], 
    max_gap: float = 2.0, 
    min_duration: float = 0.5
) -> List[Dict[str, Any]]:
    """Merge consecutive segments from the same speaker if they're close together"""
    if not segments:
        return segments
    
    merged: List[Dict[str, Any]] = []
    current_segment: Optional[Dict[str, Any]] = None
    
    for segment in segments:
        if current_segment is None:
            current_segment = segment.copy()
        elif (segment["speaker"] == current_segment["speaker"] and
              segment["start"] - current_segment["end"] <= max_gap):
            # Merge with current segment
            current_segment["end"] = segment["end"]
            current_segment["text"] = f"{current_segment['text']} {segment['text']}".strip()
            
            # Average confidence
            curr_conf = current_segment.get("confidence", 0)
            seg_conf = segment.get("confidence", 0)
            current_segment["confidence"] = (curr_conf + seg_conf) / 2
            
            # Sum word count
            current_segment["word_count"] = (
                current_segment.get("word_count", 0) + segment.get("word_count", 0)
            )
        else:
            # Different speaker or gap too large - finalize current segment
            if current_segment["end"] - current_segment["start"] >= min_duration:
                merged.append(current_segment)
            current_segment = segment.copy()
    
    # Don't forget the last segment
    if current_segment and current_segment["end"] - current_segment["start"] >= min_duration:
        merged.append(current_segment)
    
    return merged

def _simple_alignment_fallback(
    transcription_data: List[Dict[str, Any]], 
    diarization_segments: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Simple fallback alignment when enhanced method fails"""
    try:
        combined_segments: List[Dict[str, Any]] = []
        
        for d_segment in diarization_segments:
            if not isinstance(d_segment, dict) or "start" not in d_segment or "end" not in d_segment:
                continue
            
            # Simple overlap check
            matched_text: List[str] = []
            for t in transcription_data:
                if (isinstance(t, dict) and "text" in t and "start" in t and
                    d_segment["start"] <= t["start"] < d_segment["end"]):
                    text = t["text"].strip()
                    if text:
                        matched_text.append(text)
            
            combined_segments.append({
                "start": d_segment["start"],
                "end": d_segment["end"],
                "speaker": d_segment.get("speaker", "Speaker_1"),
                "text": " ".join(matched_text) if matched_text else "[No speech detected]"
            })
        
        return combined_segments
        
    except Exception as e:
        logger.error(f"Simple alignment fallback failed: {e}")
        return []

def create_test_data() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Create test data for SRT generation with proper type hints"""
    transcription_data: List[Dict[str, Any]] = [
        {"start": 0.0, "end": 1.5, "text": "First test segment", "confidence": 0.9},
        {"start": 2.0, "end": 3.5, "text": "Second test segment", "confidence": 0.8}
    ]
    
    diarization_segments: List[Dict[str, Any]] = [
        {"start": 0.0, "end": 1.5, "speaker": "Speaker_1"},
        {"start": 2.0, "end": 3.5, "speaker": "Speaker_2"}
    ]
    
    return transcription_data, diarization_segments