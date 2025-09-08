import aiofiles
import aiofiles.tempfile
import logging
import os
import time
import uuid
from pathlib import Path
from src.file_manager import FileManager
from src.logging_setup import setup_app_logging
import numpy as np

# Use proper logging setup
logger = setup_app_logging(logger_name="transcrevai.subtitle_generator")

async def generate_srt(transcription_data, diarization_segments, filename="output.srt"):
    """
    Generate SRT subtitle file from transcription and diarization data

    Args:
        transcription_data: List of transcription segments with text and timing
        diarization_segments: List of speaker diarization segments
        filename: Output filename for the SRT file

    Returns:
        str: Path to the generated SRT file
    """
    temp_path = None  # Initialize to avoid "possibly unbound" error
    
    try:
        if not transcription_data or not isinstance(transcription_data, list):
            raise ValueError("Missing required transcription data")
        
        if diarization_segments is None:
            diarization_segments = []
        elif not isinstance(diarization_segments, list):
            raise ValueError("Diarization segments must be a list")

        # Debug logging
        logger.info(f"SRT Generation - Transcription data: {len(transcription_data)} items")
        logger.info(f"SRT Generation - Diarization segments: {len(diarization_segments)} items")
        
        if transcription_data:
            logger.info(f"First transcription item: {transcription_data[0] if transcription_data else 'None'}")

        # If filename contains directory path, use provided path
        dirname = str(Path(filename).parent)
        if dirname:
            output_path = filename
            try:
                Path(dirname).mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.error(f"Failed to create output directory {dirname}: {e}")
                raise RuntimeError(f"Cannot create output directory: {e}") from e
        
        # If plain filename, use default directory
        else:
            try:
                output_dir = FileManager.get_data_path("transcripts")
                base_name = filename.split('.')[0]
                unique_id = str(uuid.uuid4())[:8]
                timestamp = int(time.time())
                unique_name = f"{base_name}_{timestamp}_{unique_id}.srt"
                output_path = str(Path(output_dir) / unique_name)
            except Exception as e:
                logger.error(f"Failed to get output directory: {e}")
                raise RuntimeError(f"Cannot determine output path: {e}") from e

        combined_segments = []

        # Use aiofiles.tempfile for async temp file handling
        async with aiofiles.tempfile.NamedTemporaryFile('w', delete=False, encoding='utf-8') as tmp:
            temp_path = tmp.name

            try:
                # ENHANCED: Process transcription data with intelligent synchronization
                if diarization_segments:
                    # Use enhanced diarization-transcription alignment
                    combined_segments = await _align_transcription_with_diarization(
                        transcription_data, diarization_segments
                    )
                else:
                    # No diarization, use transcription data directly
                    for i, t_data in enumerate(transcription_data):
                        if isinstance(t_data, dict) and "text" in t_data and t_data["text"].strip():
                            start_time = t_data.get("start", i * 2.0)
                            end_time = t_data.get("end", start_time + 2.0)

                            combined_segments.append({
                                "start": start_time,
                                "end": end_time,
                                "speaker": "Speaker_1",
                                "text": t_data["text"]
                            })

                # Filter out empty segments
                combined_segments = [s for s in combined_segments if s.get("text", "").strip()]

                if not combined_segments:
                    logger.warning("No valid segments created, attempting fallback with all transcription data")
                    
                    # Fallback: create segments from all available transcription text
                    all_text = " ".join([t.get("text", "") for t in transcription_data 
                                       if isinstance(t, dict) and t.get("text", "").strip()])
                    
                    if all_text.strip():
                        combined_segments = [{
                            "start": 0.0,
                            "end": 5.0,  # Default 5 second segment
                            "speaker": "Speaker_1",
                            "text": all_text.strip()
                        }]
                    else:
                        logger.error("No valid text found in transcription data")
                        return None

                # Write SRT content to temp file with error handling
                try:
                    for idx, segment in enumerate(combined_segments, 1):
                        # FIXED: Use proper arrow format (not HTML entity)
                        await tmp.write(f"{idx}\n"
                                      f"{format_time(segment['start'])} --> {format_time(segment['end'])}\n"
                                      f"{segment['speaker']}: {segment['text']}\n\n")

                    # Ensure data is written to disk
                    await tmp.flush()
                    
                except OSError as e:
                    logger.error(f"Failed to write to temp file: {e}")
                    raise RuntimeError(f"Failed to write SRT content: {e}") from e

            finally:
                pass  # Context manager will handle closing

        # Move temp file to final location with error handling
        try:
            # temp_path is already a string path
            os.replace(str(temp_path), str(output_path))
        except OSError as e:
            logger.error(f"Failed to move temp file to final location: {e}")
            # Clean up temp file
            if os.path.exists(str(temp_path)):
                try:
                    os.remove(str(temp_path))
                except Exception:
                    pass
            raise RuntimeError(f"Failed to write SRT file to {output_path}: {e}") from e

        logger.info(f"SRT file generated: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"SRT generation failed: {str(e)}")
        # Clean up temp file if it exists and is in scope
        if 'temp_path' in locals() and temp_path and os.path.exists(str(temp_path)):
            try:
                os.remove(str(temp_path))
            except Exception as cleanup_error:
                logger.warning(f"Temp file cleanup failed: {cleanup_error}")
        
        raise RuntimeError(f"Subtitle creation error: {str(e)}") from e

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
    """Validate SRT file format"""
    lines = content.split('\n')
    if len(lines) % 4 != 0:
        return False
        
    for i in range(0, len(lines), 4):
        # Check index number
        if not lines[i].strip().isdigit():
            return False
        # Check time range format
        if '-->' not in lines[i+1]:
            return False
        # Check text line is not empty
        if not lines[i+2].strip():
            return False
    
    return True

async def _align_transcription_with_diarization(transcription_data, diarization_segments):
    """
    Intelligent alignment of transcription data with diarization segments
    
    This function provides better synchronization by:
    1. Using overlap-based matching with weighted scoring
    2. Handling partial overlaps intelligently
    3. Grouping words/phrases by speaker
    4. Maintaining temporal coherence
    """
    try:
        combined_segments = []
        
        # Sort both datasets by start time
        sorted_transcription = sorted(
            [t for t in transcription_data if isinstance(t, dict) and "start" in t and "text" in t],
            key=lambda x: x.get("start", 0)
        )
        sorted_diarization = sorted(
            [d for d in diarization_segments if isinstance(d, dict) and "start" in d and "end" in d],
            key=lambda x: x.get("start", 0)
        )
        
        if not sorted_diarization:
            logger.warning("No valid diarization segments for alignment")
            return []
        
        logger.info(f"Aligning {len(sorted_transcription)} transcription segments with {len(sorted_diarization)} diarization segments")
        
        # ENHANCED: Advanced alignment algorithm with overlapping windows
        for d_segment in sorted_diarization:
            d_start = d_segment["start"]
            d_end = d_segment["end"]
            speaker = d_segment.get("speaker", "Speaker_1")
            
            # Find all transcription segments that overlap with this diarization segment
            overlapping_segments = []
            
            # ENHANCED: Use expanded time window for better matching
            # Extend diarization window by ±200ms to catch boundary words
            extended_start = max(0, d_start - 0.2)
            extended_end = d_end + 0.2
            
            for t_segment in sorted_transcription:
                t_start = t_segment.get("start", 0)
                t_end = t_segment.get("end", t_start + 1)
                
                # Calculate overlap with extended window
                overlap_start = max(extended_start, t_start)
                overlap_end = min(extended_end, t_end)
                overlap_duration = max(0, overlap_end - overlap_start)
                
                # Also check for proximity (words near speaker boundary)
                proximity_threshold = 0.3  # 300ms proximity
                is_near_start = abs(t_start - d_start) <= proximity_threshold
                is_near_end = abs(t_end - d_end) <= proximity_threshold
                is_within = d_start <= t_start < d_end or d_start < t_end <= d_end
                
                if overlap_duration > 0 or is_near_start or is_near_end or is_within:
                    # Calculate multiple alignment scores
                    t_duration = max(t_end - t_start, 0.1)
                    
                    # 1. Overlap ratio score
                    overlap_ratio = overlap_duration / t_duration
                    
                    # 2. Temporal proximity score
                    center_d = (d_start + d_end) / 2
                    center_t = (t_start + t_end) / 2
                    distance = abs(center_d - center_t)
                    proximity_score = max(0, 1.0 - (distance / max(d_end - d_start, 1.0)))
                    
                    # 3. Boundary alignment score (higher for words at speaker boundaries)
                    boundary_score = 0
                    if is_near_start or is_near_end:
                        boundary_score = 0.3
                    
                    # 4. Coverage score (how much of diarization segment is covered)
                    coverage_score = overlap_duration / max(d_end - d_start, 0.1)
                    
                    # Weight by confidence if available
                    confidence = t_segment.get("confidence", 1.0)
                    
                    # Combined weighted score with multiple factors
                    weighted_score = (
                        overlap_ratio * 0.4 +           # Direct overlap
                        proximity_score * 0.3 +         # Temporal proximity  
                        boundary_score * 0.2 +          # Boundary alignment
                        coverage_score * 0.1            # Coverage
                    ) * confidence
                    
                    overlapping_segments.append({
                        "segment": t_segment,
                        "overlap_duration": overlap_duration,
                        "overlap_ratio": overlap_ratio,
                        "proximity_score": proximity_score,
                        "boundary_score": boundary_score,
                        "coverage_score": coverage_score,
                        "weighted_score": weighted_score,
                        "is_boundary": is_near_start or is_near_end
                    })
            
            # Sort by weighted score (best matches first)
            overlapping_segments.sort(key=lambda x: x["weighted_score"], reverse=True)
            
            # ENHANCED: Combine text from overlapping segments with smarter filtering
            segment_texts = []
            total_confidence = 0
            used_segments = set()  # Track used segments to avoid duplicates across speakers
            
            for overlap_info in overlapping_segments:
                segment = overlap_info["segment"]
                segment_id = id(segment)  # Use object id as unique identifier
                
                # Skip if segment already used by another speaker
                if segment_id in used_segments:
                    continue
                
                # ENHANCED: More sophisticated inclusion criteria
                include_segment = False
                
                # 1. High overlap ratio
                if overlap_info["overlap_ratio"] >= 0.3:
                    include_segment = True
                
                # 2. High weighted score (combines multiple factors)
                elif overlap_info["weighted_score"] >= 0.4:
                    include_segment = True
                
                # 3. Boundary words with decent proximity
                elif overlap_info["is_boundary"] and overlap_info["proximity_score"] >= 0.5:
                    include_segment = True
                
                # 4. High coverage of diarization segment
                elif overlap_info["coverage_score"] >= 0.6:
                    include_segment = True
                
                if include_segment:
                    text = segment["text"].strip()
                    if text and len(text) > 1:  # Avoid single characters
                        segment_texts.append({
                            "text": text,
                            "start": segment.get("start", 0),
                            "end": segment.get("end", 0),
                            "confidence": segment.get("confidence", 1.0),
                            "weighted_score": overlap_info["weighted_score"]
                        })
                        total_confidence += segment.get("confidence", 1.0)
                        used_segments.add(segment_id)
            
            # Sort segment texts by start time for natural order
            segment_texts.sort(key=lambda x: x["start"])
            
            # Create combined segment
            if segment_texts:
                # Extract just the text strings for joining
                text_strings = [seg["text"] for seg in segment_texts]
                combined_text = " ".join(text_strings)
                avg_confidence = total_confidence / len(segment_texts) if segment_texts else 0
                
                # Calculate actual temporal boundaries from transcription
                actual_start = min(seg["start"] for seg in segment_texts) if segment_texts else d_start
                actual_end = max(seg["end"] for seg in segment_texts) if segment_texts else d_end
                
                # Prefer diarization timing but adjust if transcription boundaries are better
                final_start = min(d_start, actual_start) if abs(d_start - actual_start) < 0.5 else d_start
                final_end = max(d_end, actual_end) if abs(d_end - actual_end) < 0.5 else d_end
                
                combined_segment = {
                    "start": final_start,
                    "end": final_end,
                    "speaker": speaker,
                    "text": combined_text,
                    "confidence": avg_confidence,
                    "word_count": len(segment_texts),
                    "alignment_method": "enhanced_temporal",
                    "transcription_segments": len(segment_texts)
                }
                
                combined_segments.append(combined_segment)
                logger.debug(f"Aligned segment: {speaker} ({d_start:.2f}-{d_end:.2f}s): '{combined_text[:50]}...'")
            else:
                # No text found for this speaker segment, but keep the timing
                logger.debug(f"No text found for {speaker} segment ({d_start:.2f}-{d_end:.2f}s)")
                combined_segments.append({
                    "start": d_start,
                    "end": d_end,
                    "speaker": speaker,
                    "text": "[No speech detected]",
                    "confidence": 0.0,
                    "word_count": 0
                })
        
        # Post-processing: merge very short segments from same speaker
        merged_segments = _merge_consecutive_speaker_segments(combined_segments)
        
        logger.info(f"Alignment complete: {len(combined_segments)} segments -> {len(merged_segments)} after merging")
        return merged_segments
        
    except Exception as e:
        logger.error(f"Transcription-diarization alignment failed: {e}")
        # Fallback to simple alignment
        return _simple_alignment_fallback(transcription_data, diarization_segments)

def _merge_consecutive_speaker_segments(segments, max_gap=2.0, min_duration=0.5):
    """
    Merge consecutive segments from the same speaker if they're close together
    """
    if not segments:
        return segments
    
    merged = []
    current_segment = None
    
    for segment in segments:
        if current_segment is None:
            current_segment = segment.copy()
        elif (segment["speaker"] == current_segment["speaker"] and 
              segment["start"] - current_segment["end"] <= max_gap):
            # Merge with current segment
            current_segment["end"] = segment["end"]
            current_segment["text"] = f"{current_segment['text']} {segment['text']}".strip()
            current_segment["confidence"] = (current_segment.get("confidence", 0) + segment.get("confidence", 0)) / 2
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

def _simple_alignment_fallback(transcription_data, diarization_segments):
    """
    Simple fallback alignment when advanced method fails
    """
    try:
        combined_segments = []
        
        for d_segment in diarization_segments:
            if isinstance(d_segment, dict) and "start" in d_segment and "end" in d_segment:
                # Simple overlap check
                matched_text = []
                for t in transcription_data:
                    if (isinstance(t, dict) and "text" in t and "start" in t and
                        d_segment["start"] <= t["start"] < d_segment["end"]):
                        matched_text.append(t["text"])

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

def create_test_data():
    """Create test data for SRT generation"""
    transcription_data = [
        {"start": 0.0, "end": 1.5, "text": "First test segment"},
        {"start": 2.0, "end": 3.5, "text": "Second test segment"}
    ]
    
    diarization_segments = [
        {"start": 0.0, "end": 1.5, "speaker": "Speaker_1"},
        {"start": 2.0, "end": 3.5, "speaker": "Speaker_2"}
    ]
    
    return transcription_data, diarization_segments