import aiofiles
import aiofiles.tempfile
import logging
import os
import time
import uuid
from src.file_manager import FileManager
from src.logging_setup import setup_app_logging

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
        dirname = os.path.dirname(filename)
        if dirname:
            output_path = filename
            try:
                os.makedirs(dirname, exist_ok=True)
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
                output_path = os.path.join(output_dir, unique_name)
            except Exception as e:
                logger.error(f"Failed to get output directory: {e}")
                raise RuntimeError(f"Cannot determine output path: {e}") from e

        combined_segments = []
        temp_path = None

        # Use aiofiles.tempfile for async temp file handling
        async with aiofiles.tempfile.NamedTemporaryFile('w', delete=False, encoding='utf-8') as tmp:
            temp_path = tmp.name

            try:
                # Process transcription data and create combined segments
                if diarization_segments:
                    # Use diarization segments
                    for d_segment in diarization_segments:
                        if isinstance(d_segment, dict) and "start" in d_segment and "end" in d_segment:
                            matched_text = [t["text"] for t in transcription_data
                                          if isinstance(t, dict)
                                          and "text" in t
                                          and "start" in t
                                          and d_segment["start"] <= t["start"] < d_segment["end"]]

                            combined_segments.append({
                                "start": d_segment["start"],
                                "end": d_segment["end"],
                                "speaker": d_segment.get("speaker", "Speaker"),
                                "text": " ".join(matched_text)
                            })
                else:
                    # No diarization, use transcription data directly
                    for i, t_data in enumerate(transcription_data):
                        if isinstance(t_data, dict) and "text" in t_data and t_data["text"].strip():
                            start_time = t_data.get("start", i * 2.0)
                            end_time = t_data.get("end", start_time + 2.0)

                            combined_segments.append({
                                "start": start_time,
                                "end": end_time,
                                "speaker": "Speaker",
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
                            "speaker": "Speaker",
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
        if 'temp_path' in locals() and temp_path and os.path.exists(temp_path):
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
