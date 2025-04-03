import aiofiles
import logging
import os
import time
from src.file_manager import FileManager
import tempfile
from unittest import IsolatedAsyncioTestCase as AsyncTestCase

logger = logging.getLogger(__name__)

async def generate_srt(transcription_data, diarization_segments, filename="output.srt"):
    try:
        if not transcription_data or not isinstance(diarization_segments, list):
            raise ValueError("Missing required input data")
        # If filename contains directory path, use provided path
        dirname = os.path.dirname(filename)
        if dirname:
            output_path = filename
            os.makedirs(dirname, exist_ok=True)
        # If plain filename, use default directory
        else:
            output_dir = FileManager.get_data_path("transcripts")
            unique_name = f"{filename.split('.')[0]}_{int(time.time())}.srt"
            output_path = os.path.join(output_dir, unique_name)
        combined_segments = []
        with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as tmp:
            temp_path = tmp.name
        try:
            async with aiofiles.open(temp_path, "w", encoding="utf-8") as f:
                for d_segment in diarization_segments:
                    matched_text = [t["text"] for t in transcription_data
                                    if isinstance(t, dict) 
                                    and isinstance(d_segment, dict) 
                                    and "start" in t 
                                    and "start" in d_segment 
                                    and "end" in d_segment 
                                    and d_segment["start"] <= t["start"] < d_segment["end"]]
                    combined_segments.append({"start": d_segment["start"],
                                              "end": d_segment["end"],
                                              "speaker": d_segment.get("speaker", "Speaker"),
                                              "text": " ".join(matched_text)})
                for idx, segment in enumerate(combined_segments, 1):
                    await f.write(f"{idx}\n"
                                  f"{format_time(segment['start'])} --> {format_time(segment['end'])}\n"
                                  f"{segment['speaker']}: {segment['text']}\n\n")
            os.replace(temp_path, output_path)
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as cleanup_error:
                    logger.warning(f"Temp file cleanup failed: {cleanup_error}")
        logger.info(f"SRT file generated: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"SRT generation failed: {str(e)}")
        raise RuntimeError(f"Subtitle creation error: {str(e)}") from e
    
def format_time(seconds: float) -> str:
    try:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        sec = seconds % 60
        return f"{hours:02}:{minutes:02}:{sec:06.3f}".replace('.', ',')[:12]
    except TypeError:
        raise ValueError(f"Invalid time value: {seconds}")

class TestSubtitles(AsyncTestCase):
    async def test_srt_integrity(self, temp_path): 
        test_data = [{"start": 0.0, "end": 1.5, "text": "First segment", "speaker": "Speaker_1"},
                     {"start": 2.0, "end": 3.5, "text": "Second segment", "speaker": "Speaker_2"}]
        diarization_segments = [{"start": 0.0, "end": 1.5, "speaker": "Speaker_1"},
                                {"start": 2.0, "end": 3.5, "speaker": "Speaker_2"}]
        output_path = temp_path / "test_output.srt"
        generated_path = await generate_srt(test_data, diarization_segments, output_path)
        async with aiofiles.open(generated_path, "r", encoding="utf-8") as f:
            lines = await f.readlines()
        assert len(lines) % 4 == 0, "Unexpected SRT format: Number of lines is not a multiple of 4"
        for i in range(0, len(lines), 4):
            assert lines[i].strip().isdigit(), "Expected index number"
            assert "-->" in lines[i+1], "Expected time range format"
            assert len(lines[i+2].strip()) > 0, "Expected non-empty text line"