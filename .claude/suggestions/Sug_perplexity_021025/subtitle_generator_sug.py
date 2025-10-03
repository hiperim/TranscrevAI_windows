# FIXED - Enhanced Subtitle Generator with UTF-8 Windows Support
"""
Enhanced Subtitle Generator Module for TranscrevAI
Production-ready SRT generation with comprehensive UTF-8 support for Windows

FIXES APPLIED:
- Complete UTF-8 encoding support for Windows SRT files
- UTF-8 BOM for maximum Windows compatibility
- Proper Portuguese character handling
- Enhanced error handling with encoding fallbacks
- Thread-safe file operations
"""

import logging
import asyncio
import os
import time
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import threading

logger = logging.getLogger(__name__)

@dataclass
class SubtitleSegment:
    """Enhanced subtitle segment with proper typing and encoding support"""
    start_time: float
    end_time: float
    text: str
    speaker: Optional[str] = None
    confidence: Optional[float] = None

class SRTGenerator:
    """Enhanced SRT generator with comprehensive UTF-8 Windows support"""
    
    def __init__(self):
        # Thread safety for concurrent subtitle generation
        self._generation_lock = threading.RLock()
        
        # SRT formatting parameters
        self.max_line_length = 42  # Characters per line for readability
        self.max_lines_per_subtitle = 2
        self.min_subtitle_duration = 0.5  # Minimum subtitle display time
        self.max_subtitle_duration = 6.0  # Maximum subtitle display time
        
        # Portuguese-specific character handling
        self._init_portuguese_formatting()

    def _init_portuguese_formatting(self):
        """Initialize Portuguese-specific formatting rules"""
        
        # Common Portuguese punctuation and formatting
        self.portuguese_punctuation = {
            '"': '"',  # Smart quotes
            '"': '"',
            ''': "'",
            ''': "'",
            '…': '...',  # Ellipsis normalization
            '–': '-',   # En dash to hyphen
            '—': '-',   # Em dash to hyphen
        }
        
        # Portuguese abbreviations that shouldn't be split
        self.portuguese_abbreviations = {
            'Dr.', 'Dra.', 'Sr.', 'Sra.', 'Jr.', 'Prof.', 'Profa.',
            'etc.', 'ex.', 'obs.', 'p.', 'pp.', 'vol.', 'cap.',
            'art.', 'inc.', 'ltd.', 'ltda.', 'S.A.', 'LTDA.'
        }

    def _format_time_srt(self, seconds: float) -> str:
        """Format time for SRT format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds - int(seconds)) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

    def _clean_text_for_subtitle(self, text: str) -> str:
        """Clean and format text for subtitle display with Portuguese support"""
        if not text:
            return ""
        
        # Normalize Unicode for consistent Portuguese character handling
        import unicodedata
        text = unicodedata.normalize('NFC', text)
        
        # Replace problematic punctuation
        for old, new in self.portuguese_punctuation.items():
            text = text.replace(old, new)
        
        # Clean up extra whitespace
        text = ' '.join(text.split())
        
        # Ensure proper capitalization
        text = text.strip()
        if text:
            text = text[0].upper() + text[1:]
        
        # Fix common transcription artifacts
        text = self._fix_transcription_artifacts(text)
        
        return text

    def _fix_transcription_artifacts(self, text: str) -> str:
        """Fix common transcription artifacts in Portuguese"""
        
        fixes = [
            # Fix repeated words
            (r'\b(\w+)\s+\1\b', r'\1'),
            
            # Fix spacing around punctuation
            (r'\s+([,.!?;:])', r'\1'),
            (r'([,.!?;:])\s*([a-zA-ZÀ-ÿ])', r'\1 \2'),
            
            # Fix multiple spaces
            (r'\s{2,}', ' '),
            
            # Fix common Portuguese transcription errors
            (r'\bé\s+([aeiou])', r'é \1'),  # Fix accent spacing
            (r'\bão\s+([bcdfgjklmnpqrstvwxyz])', r'ão \1'),  # Fix nasal spacing
            
            # Fix sentence boundaries
            (r'\.{2,}', '.'),  # Multiple periods to single
            (r'([.!?])\s*([a-z])', lambda m: m.group(1) + ' ' + m.group(2).upper()),  # Capitalize after sentence end
        ]
        
        for pattern, replacement in fixes:
            if callable(replacement):
                text = re.sub(pattern, replacement, text)
            else:
                text = re.sub(pattern, replacement, text)
        
        return text

    def _split_text_into_lines(self, text: str) -> List[str]:
        """Split text into appropriate subtitle lines respecting Portuguese word boundaries"""
        if len(text) <= self.max_line_length:
            return [text]
        
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            # Check if adding this word would exceed line length
            test_line = f"{current_line} {word}".strip()
            
            if len(test_line) <= self.max_line_length:
                current_line = test_line
            else:
                # Current line is full, start new line
                if current_line:
                    lines.append(current_line)
                    current_line = word
                else:
                    # Single word is too long, force it on a line
                    lines.append(word)
                    current_line = ""
        
        # Add the last line
        if current_line:
            lines.append(current_line)
        
        # Limit to maximum lines per subtitle
        if len(lines) > self.max_lines_per_subtitle:
            # Try to combine lines intelligently
            lines = self._optimize_line_breaks(lines)
        
        return lines[:self.max_lines_per_subtitle]

    def _optimize_line_breaks(self, lines: List[str]) -> List[str]:
        """Optimize line breaks for better readability in Portuguese"""
        if len(lines) <= self.max_lines_per_subtitle:
            return lines
        
        # Strategy: combine shorter lines while respecting max line length
        optimized = []
        i = 0
        
        while i < len(lines):
            current_line = lines[i]
            
            # Try to combine with next line if possible
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                combined = f"{current_line} {next_line}"
                
                if len(combined) <= self.max_line_length:
                    optimized.append(combined)
                    i += 2  # Skip next line as it's been combined
                    continue
            
            optimized.append(current_line)
            i += 1
        
        return optimized

    def _create_srt_content(self, segments: List[Dict[str, Any]]) -> str:
        """Create SRT content from segments with proper UTF-8 encoding"""
        srt_entries = []
        subtitle_index = 1
        
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            
            text = segment.get("text", "").strip()
            if not text:
                continue
            
            start_time = float(segment.get("start", 0))
            end_time = float(segment.get("end", start_time + 2.0))
            speaker = segment.get("speaker")
            
            # Ensure minimum and maximum subtitle duration
            duration = end_time - start_time
            if duration < self.min_subtitle_duration:
                end_time = start_time + self.min_subtitle_duration
            elif duration > self.max_subtitle_duration:
                end_time = start_time + self.max_subtitle_duration
            
            # Clean and format text
            clean_text = self._clean_text_for_subtitle(text)
            if not clean_text:
                continue
            
            # Add speaker label if available
            if speaker and speaker.strip():
                clean_text = f"[{speaker}] {clean_text}"
            
            # Split into appropriate lines
            lines = self._split_text_into_lines(clean_text)
            subtitle_text = "\n".join(lines)
            
            # Format time stamps
            start_formatted = self._format_time_srt(start_time)
            end_formatted = self._format_time_srt(end_time)
            
            # Create SRT entry
            srt_entry = f"{subtitle_index}\n{start_formatted} --> {end_formatted}\n{subtitle_text}\n"
            srt_entries.append(srt_entry)
            
            subtitle_index += 1
        
        return "\n".join(srt_entries)

    async def generate_srt_file(self, segments: List[Dict[str, Any]], 
                               output_path: Optional[Union[str, Path]] = None,
                               filename: Optional[str] = None) -> Optional[str]:
        """
        Generate SRT file with comprehensive UTF-8 Windows support
        
        Args:
            segments: List of transcription segments
            output_path: Output directory path
            filename: Output filename
            
        Returns:
            Path to generated SRT file or None if failed
        """
        with self._generation_lock:
            try:
                # Determine output path
                if output_path is None:
                    output_path = Path("data/subtitles")
                else:
                    output_path = Path(output_path)
                
                # Create output directory
                output_path.mkdir(parents=True, exist_ok=True)
                
                # Determine filename
                if filename is None:
                    timestamp = int(time.time())
                    filename = f"transcript_{timestamp}.srt"
                elif not filename.endswith('.srt'):
                    filename = f"{filename}.srt"
                
                # Full file path
                srt_file_path = output_path / filename
                
                # Generate SRT content
                srt_content = self._create_srt_content(segments)
                
                if not srt_content.strip():
                    logger.warning("No valid segments to generate SRT file")
                    return None
                
                # Write SRT file with UTF-8 encoding and BOM for Windows compatibility
                await self._write_srt_file_safe(srt_file_path, srt_content)
                
                logger.info(f"SRT file generated successfully: {srt_file_path}")
                return str(srt_file_path)
                
            except Exception as e:
                logger.error(f"Failed to generate SRT file: {e}")
                return None

    async def _write_srt_file_safe(self, file_path: Path, content: str):
        """Write SRT file safely with UTF-8 encoding and Windows compatibility"""
        
        # Multiple encoding strategies for maximum Windows compatibility
        encoding_strategies = [
            ('utf-8-sig', 'UTF-8 with BOM (best Windows compatibility)'),
            ('utf-8', 'UTF-8 without BOM'),
            ('cp1252', 'Windows-1252 fallback'),
            ('latin-1', 'Latin-1 fallback')
        ]
        
        last_error = None
        
        for encoding, description in encoding_strategies:
            try:
                # Run file write in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None, 
                    self._write_file_with_encoding,
                    file_path, content, encoding
                )
                
                logger.info(f"SRT file written successfully using {description}")
                return
                
            except UnicodeEncodeError as e:
                last_error = e
                logger.warning(f"Encoding {encoding} failed: {e}")
                
                # Try to clean problematic characters and retry
                if encoding == 'utf-8-sig':
                    try:
                        cleaned_content = self._clean_content_for_encoding(content)
                        await loop.run_in_executor(
                            None,
                            self._write_file_with_encoding,
                            file_path, cleaned_content, encoding
                        )
                        logger.info(f"SRT file written with cleaned content using {description}")
                        return
                    except Exception:
                        continue
                        
            except Exception as e:
                last_error = e
                logger.warning(f"Failed to write with {encoding}: {e}")
                continue
        
        # If all strategies failed, raise the last error
        if last_error:
            raise RuntimeError(f"Failed to write SRT file with any encoding strategy: {last_error}")

    def _write_file_with_encoding(self, file_path: Path, content: str, encoding: str):
        """Write file with specific encoding (blocking operation)"""
        with open(file_path, 'w', encoding=encoding, newline='\r\n') as f:
            f.write(content)

    def _clean_content_for_encoding(self, content: str) -> str:
        """Clean content to be compatible with various encodings"""
        import unicodedata
        
        # Normalize unicode
        content = unicodedata.normalize('NFKD', content)
        
        # Replace problematic characters
        problematic_chars = {
            'ắ': 'á', 'ằ': 'à', 'ẵ': 'ã', 'ẳ': 'ả', 'ặ': 'ạ',
            'ế': 'é', 'ề': 'è', 'ễ': 'ẽ', 'ể': 'ẻ', 'ệ': 'ẹ',
            'ố': 'ó', 'ồ': 'ò', 'ỗ': 'õ', 'ổ': 'ỏ', 'ộ': 'ọ',
            'ứ': 'ú', 'ừ': 'ù', 'ữ': 'ũ', 'ử': 'ủ', 'ự': 'ụ',
            'ý': 'ý', 'ỳ': 'ỳ', 'ỹ': 'ỹ', 'ỷ': 'ỷ', 'ỵ': 'ỵ',
        }
        
        for old, new in problematic_chars.items():
            content = content.replace(old, new)
        
        # Remove or replace any remaining non-ASCII characters as last resort
        # This is only for extreme compatibility cases
        content = content.encode('ascii', 'replace').decode('ascii')
        content = content.replace('?', '')  # Remove replacement characters
        
        return content

    def validate_srt_content(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate segments before SRT generation"""
        
        validation_result = {
            "valid": True,
            "total_segments": len(segments),
            "valid_segments": 0,
            "issues": [],
            "estimated_duration": 0.0
        }
        
        if not segments:
            validation_result["valid"] = False
            validation_result["issues"].append("No segments provided")
            return validation_result
        
        valid_count = 0
        max_end_time = 0.0
        
        for i, segment in enumerate(segments):
            segment_issues = []
            
            if not isinstance(segment, dict):
                segment_issues.append(f"Segment {i}: Not a dictionary")
                continue
            
            # Check required fields
            if "text" not in segment:
                segment_issues.append(f"Segment {i}: Missing 'text' field")
            elif not segment["text"].strip():
                segment_issues.append(f"Segment {i}: Empty text")
            
            if "start" not in segment:
                segment_issues.append(f"Segment {i}: Missing 'start' time")
            elif not isinstance(segment["start"], (int, float)):
                segment_issues.append(f"Segment {i}: Invalid start time type")
            
            if "end" not in segment:
                segment_issues.append(f"Segment {i}: Missing 'end' time")
            elif not isinstance(segment["end"], (int, float)):
                segment_issues.append(f"Segment {i}: Invalid end time type")
            
            # Check timing logic
            if "start" in segment and "end" in segment:
                start_time = float(segment["start"])
                end_time = float(segment["end"])
                
                if start_time >= end_time:
                    segment_issues.append(f"Segment {i}: Start time >= end time")
                
                if start_time < 0:
                    segment_issues.append(f"Segment {i}: Negative start time")
                
                max_end_time = max(max_end_time, end_time)
            
            if segment_issues:
                validation_result["issues"].extend(segment_issues)
            else:
                valid_count += 1
        
        validation_result["valid_segments"] = valid_count
        validation_result["estimated_duration"] = max_end_time
        
        if valid_count == 0:
            validation_result["valid"] = False
            validation_result["issues"].append("No valid segments found")
        
        return validation_result


# Global SRT generator instance
srt_generator = SRTGenerator()

async def generate_srt(segments: List[Dict[str, Any]], 
                      output_path: Optional[Union[str, Path]] = None,
                      filename: Optional[str] = None) -> Optional[str]:
    """
    Convenience function to generate SRT file
    
    Args:
        segments: List of transcription segments
        output_path: Output directory path
        filename: Output filename
        
    Returns:
        Path to generated SRT file or None if failed
    """
    try:
        # Validate segments first
        validation = srt_generator.validate_srt_content(segments)
        
        if not validation["valid"]:
            logger.error(f"SRT validation failed: {validation['issues']}")
            return None
        
        if validation["valid_segments"] < validation["total_segments"]:
            logger.warning(f"Only {validation['valid_segments']}/{validation['total_segments']} segments are valid")
        
        # Generate SRT file
        return await srt_generator.generate_srt_file(segments, output_path, filename)
        
    except Exception as e:
        logger.error(f"SRT generation failed: {e}")
        return None

def create_subtitle_segment(start_time: float, end_time: float, text: str, 
                          speaker: Optional[str] = None, 
                          confidence: Optional[float] = None) -> Dict[str, Any]:
    """
    Create a properly formatted subtitle segment
    
    Args:
        start_time: Start time in seconds
        end_time: End time in seconds
        text: Subtitle text
        speaker: Optional speaker identifier
        confidence: Optional confidence score
        
    Returns:
        Formatted segment dictionary
    """
    segment = {
        "start": float(start_time),
        "end": float(end_time),
        "text": str(text).strip()
    }
    
    if speaker is not None:
        segment["speaker"] = str(speaker).strip()
    
    if confidence is not None:
        segment["confidence"] = float(confidence)
    
    return segment

def merge_subtitle_segments(segments: List[Dict[str, Any]], 
                          max_gap: float = 0.5) -> List[Dict[str, Any]]:
    """
    Merge consecutive segments with small gaps for better subtitle flow
    
    Args:
        segments: List of subtitle segments
        max_gap: Maximum gap between segments to merge (seconds)
        
    Returns:
        List of merged segments
    """
    if not segments:
        return []
    
    merged = []
    current_segment = None
    
    for segment in segments:
        if not isinstance(segment, dict) or "start" not in segment or "end" not in segment:
            continue
        
        if current_segment is None:
            current_segment = segment.copy()
            continue
        
        # Check if segments can be merged
        gap = segment["start"] - current_segment["end"]
        same_speaker = (
            current_segment.get("speaker") == segment.get("speaker") or
            (not current_segment.get("speaker") and not segment.get("speaker"))
        )
        
        if gap <= max_gap and same_speaker:
            # Merge segments
            current_segment["end"] = segment["end"]
            current_segment["text"] = f"{current_segment['text']} {segment['text']}".strip()
            
            # Update confidence (average)
            if "confidence" in current_segment and "confidence" in segment:
                curr_conf = current_segment["confidence"]
                new_conf = segment["confidence"]
                current_segment["confidence"] = (curr_conf + new_conf) / 2
        else:
            # Can't merge, add current segment and start new one
            merged.append(current_segment)
            current_segment = segment.copy()
    
    # Add the last segment
    if current_segment is not None:
        merged.append(current_segment)
    
    return merged