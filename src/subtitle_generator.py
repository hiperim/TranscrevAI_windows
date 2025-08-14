import aiofiles
import aiofiles.tempfile
import logging
import os
import time
import uuid
from src.file_manager import FileManager
import tempfile
from unittest import IsolatedAsyncioTestCase as AsyncTestCase

logger = logging.getLogger(__name__)

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
                    all_text = " ".join([t.get("text", "") for t in transcription_data if isinstance(t, dict) and t.get("text", "").strip()])
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
                        # Use proper arrow syntax instead of HTML entities
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
            os.replace(temp_path, output_path)
        except OSError as e:
            logger.error(f"Failed to move temp file to final location: {e}")
            # Clean up temp file
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            raise RuntimeError(f"Failed to write SRT file to {output_path}: {e}") from e

        logger.info(f"SRT file generated: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"SRT generation failed: {str(e)}")

        # Clean up temp file if it exists
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
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
    except TypeError:
        raise ValueError(f"Invalid time value: {seconds}")

class TestSubtitles(AsyncTestCase):
    """Test class for subtitle generation functionality"""

    async def test_srt_integrity(self, temp_path):
        """Test SRT file generation and format integrity"""
        test_data = [
            {"start": 0.0, "end": 1.5, "text": "First segment", "speaker": "Speaker_1"},
            {"start": 2.0, "end": 3.5, "text": "Second segment", "speaker": "Speaker_2"}
        ]

        diarization_segments = [
            {"start": 0.0, "end": 1.5, "speaker": "Speaker_1"},
            {"start": 2.0, "end": 3.5, "speaker": "Speaker_2"}
        ]

        output_path = temp_path / "test_output.srt"
        generated_path = await generate_srt(test_data, diarization_segments, str(output_path))

        # Verify file was created and has content
        assert os.path.exists(generated_path), "SRT file was not created"
        assert os.path.getsize(generated_path) > 0, "SRT file is empty"

        # Read and verify SRT format
        async with aiofiles.open(generated_path, "r", encoding="utf-8") as f:
            lines = await f.readlines()

        # SRT format validation
        assert len(lines) % 4 == 0, "Unexpected SRT format: Number of lines is not a multiple of 4"

        for i in range(0, len(lines), 4):
            # Check index number
            assert lines[i].strip().isdigit(), "Expected index number"
            # Check time range format (should contain "-->")
            assert "-->" in lines[i+1], "Expected time range format"
            # Check text line is not empty
            assert len(lines[i+2].strip()) > 0, "Expected non-empty text line"

    async def test_format_time(self):
        """Test time formatting function"""
        # Test normal time
        result = format_time(3661.123)  # 1:01:01.123
        assert result == "01:01:01,123", f"Expected '01:01:01,123', got '{result}'"

        # Test zero time
        result = format_time(0.0)
        assert result == "00:00:00,000", f"Expected '00:00:00,000', got '{result}'"

        # Test fractional seconds
        result = format_time(1.5)
        assert result == "00:00:01,500", f"Expected '00:00:01,500', got '{result}'"

    async def test_empty_input(self):
        """Test handling of empty input data"""
        try:
            await generate_srt([], [])
            assert False, "Should raise ValueError for empty input"
        except ValueError as e:
            assert "Missing required input data" in str(e)

    async def test_invalid_input_types(self):
        """Test handling of invalid input types"""
        try:
            await generate_srt("invalid", "invalid")
            assert False, "Should raise ValueError for invalid input types"
        except ValueError as e:
            assert "Missing required input data" in str(e)