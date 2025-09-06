# Optimized Audio Chunker for TranscrevAI
# Real-time audio chunking with VAD and smart boundaries

"""
OptimizedAudioChunker

High-performance audio chunking system optimized for real-time transcription:
- 30-second chunks with 2-second overlap for optimal balance
- Voice Activity Detection (VAD) for intelligent boundaries
- Prevention of word cuts through smart boundary detection
- Context preservation between chunks
- Memory-efficient streaming processing
"""

import asyncio
import logging
import numpy as np
import time
from typing import List, Dict, AsyncGenerator, Tuple, Optional
from pathlib import Path
import tempfile
import os

# Audio processing imports
try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    librosa = None
    sf = None
    LIBROSA_AVAILABLE = False

try:
    import webrtcvad
    VAD_AVAILABLE = True
except ImportError:
    webrtcvad = None
    VAD_AVAILABLE = False

from config.whisper_optimization import ADVANCED_CONFIG, PERFORMANCE_CONSTRAINTS
from src.file_manager import FileManager

logger = logging.getLogger(__name__)

class AudioChunkingError(Exception):
    """Raised when audio chunking fails"""
    pass

class OptimizedAudioChunker:
    """
    Optimized audio chunker for real-time processing
    
    Features:
    - Smart chunking with optimal 30s/2s overlap configuration
    - Voice Activity Detection to prevent word cuts
    - Context preservation between chunks
    - Memory-efficient streaming
    - Real-time performance monitoring
    """
    
    def __init__(self):
        """Initialize the audio chunker with optimized settings"""
        # Configuration from research-based optimizations
        self.chunk_duration = ADVANCED_CONFIG["chunk_size"]  # 30.0 seconds
        self.overlap_duration = ADVANCED_CONFIG["overlap_duration"]  # 2.0 seconds
        self.vad_threshold = ADVANCED_CONFIG["vad_threshold"]  # 0.5
        self.min_segment_duration = ADVANCED_CONFIG["min_segment_duration"]  # 0.1s
        
        # Audio processing settings
        self.sample_rate = 16000  # Whisper's native sample rate
        self.frame_duration = 30  # VAD frame duration in ms
        
        # Performance tracking
        self.processing_times = []
        
        # VAD initialization
        self.vad = None
        if VAD_AVAILABLE and webrtcvad is not None:
            try:
                self.vad = webrtcvad.Vad(2)  # Moderate aggressiveness
                logger.info("WebRTC VAD initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize WebRTC VAD: {e}")
                self.vad = None
        else:
            logger.warning("WebRTC VAD not available, using fallback energy-based VAD")
        
        logger.info(f"OptimizedAudioChunker initialized: chunk={self.chunk_duration}s, overlap={self.overlap_duration}s")
    
    async def create_optimized_chunks(self, audio_file: str) -> List[Dict]:
        """
        Create optimized audio chunks for real-time processing
        
        Args:
            audio_file (str): Path to audio file
            
        Returns:
            List[Dict]: List of chunk information with timing and file paths
            
        Raises:
            AudioChunkingError: If chunking fails
        """
        try:
            start_time = time.time()
            
            # Load and validate audio
            audio_data, sr = await self._load_audio(audio_file)
            logger.info(f"Loaded audio: {len(audio_data)} samples at {sr}Hz")
            
            # Get total duration
            total_duration = len(audio_data) / sr
            
            if total_duration <= self.chunk_duration:
                # Audio is short enough to process as single chunk
                chunks = await self._create_single_chunk(audio_file, audio_data, sr, total_duration)
            else:
                # Create multiple chunks with smart boundaries
                chunks = await self._create_multiple_chunks(audio_file, audio_data, sr, total_duration)
            
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            logger.info(f"Created {len(chunks)} optimized chunks in {processing_time:.2f}s")
            return chunks
            
        except Exception as e:
            logger.error(f"Audio chunking failed: {e}")
            raise AudioChunkingError(f"Chunking error: {str(e)}")
    
    async def _load_audio(self, audio_file: str) -> Tuple[np.ndarray, int]:
        """Load audio file with error handling"""
        if not LIBROSA_AVAILABLE or librosa is None:
            raise AudioChunkingError("librosa not available for audio loading")
        
        try:
            # Load audio with librosa (mono, 16kHz for Whisper)
            audio_data, sr = librosa.load(audio_file, sr=self.sample_rate, mono=True)
            
            if len(audio_data) == 0:
                raise AudioChunkingError("Loaded audio is empty")
            
            return audio_data, int(sr)
            
        except Exception as e:
            raise AudioChunkingError(f"Failed to load audio: {str(e)}")
    
    async def _create_single_chunk(self, audio_file: str, audio_data: np.ndarray, sr: int, duration: float) -> List[Dict]:
        """Create single chunk for short audio"""
        chunk_info = {
            "index": 0,
            "start_time": 0.0,
            "end_time": duration,
            "duration": duration,
            "file_path": audio_file,  # Use original file
            "has_overlap": False,
            "context_start": 0.0,
            "context_end": duration
        }
        
        logger.debug(f"Created single chunk: {duration:.2f}s")
        return [chunk_info]
    
    async def _create_multiple_chunks(self, audio_file: str, audio_data: np.ndarray, sr: int, total_duration: float) -> List[Dict]:
        """Create multiple chunks with smart boundaries"""
        chunks = []
        
        # Calculate chunk positions
        chunk_positions = await self._calculate_chunk_positions(audio_data, sr, total_duration)
        
        # Create temporary directory for chunk files
        temp_dir = Path(FileManager.get_data_path("temp"))
        FileManager.ensure_directory_exists(str(temp_dir))
        
        # Generate chunks
        for i, (start_time, end_time) in enumerate(chunk_positions):
            chunk_info = await self._create_chunk_file(
                audio_data, sr, start_time, end_time, i, temp_dir, total_duration
            )
            chunks.append(chunk_info)
        
        logger.info(f"Created {len(chunks)} chunks with smart boundaries")
        return chunks
    
    async def _calculate_chunk_positions(self, audio_data: np.ndarray, sr: int, total_duration: float) -> List[Tuple[float, float]]:
        """Calculate optimal chunk positions with smart boundaries"""
        positions = []
        current_time = 0.0
        
        while current_time < total_duration:
            # Calculate end time for this chunk
            chunk_end = min(current_time + self.chunk_duration, total_duration)
            
            # If this isn't the last chunk, try to find a good boundary
            if chunk_end < total_duration:
                optimal_end = await self._find_optimal_boundary(
                    audio_data, sr, current_time + self.chunk_duration, total_duration
                )
                if optimal_end:
                    chunk_end = optimal_end
            
            positions.append((current_time, chunk_end))
            
            # Move to next chunk with overlap
            if chunk_end >= total_duration:
                break
            
            # Next chunk starts with overlap
            next_start = chunk_end - self.overlap_duration
            current_time = max(next_start, current_time + self.min_segment_duration)
        
        return positions
    
    async def _find_optimal_boundary(self, audio_data: np.ndarray, sr: int, target_time: float, max_time: float) -> Optional[float]:
        """Find optimal chunk boundary to avoid cutting words"""
        try:
            # Define search window (±1 second around target)
            search_window = 1.0
            search_start = max(target_time - search_window, 0)
            search_end = min(target_time + search_window, max_time)
            
            # Convert to sample indices
            start_sample = int(search_start * sr)
            end_sample = int(search_end * sr)
            target_sample = int(target_time * sr)
            
            if end_sample <= start_sample:
                return None
            
            # Get audio segment for analysis
            segment = audio_data[start_sample:end_sample]
            
            # Find optimal boundary using multiple methods
            boundary_candidates = []
            
            # Method 1: Energy-based silence detection
            energy_boundary = await self._find_energy_boundary(segment, sr, search_start, target_time)
            if energy_boundary:
                boundary_candidates.append(('energy', energy_boundary))
            
            # Method 2: VAD-based boundary (if available)
            if self.vad:
                vad_boundary = await self._find_vad_boundary(segment, sr, search_start, target_time)
                if vad_boundary:
                    boundary_candidates.append(('vad', vad_boundary))
            
            # Method 3: Zero-crossing rate boundary
            zcr_boundary = await self._find_zcr_boundary(segment, sr, search_start, target_time)
            if zcr_boundary:
                boundary_candidates.append(('zcr', zcr_boundary))
            
            # Select best boundary (prefer VAD, then energy, then ZCR)
            if boundary_candidates:
                # Sort by method preference and distance from target
                boundary_candidates.sort(key=lambda x: (
                    {'vad': 0, 'energy': 1, 'zcr': 2}.get(x[0], 3),
                    abs(x[1] - target_time)
                ))
                
                selected_boundary = boundary_candidates[0][1]
                logger.debug(f"Selected {boundary_candidates[0][0]} boundary at {selected_boundary:.2f}s (target: {target_time:.2f}s)")
                return selected_boundary
            
            return None
            
        except Exception as e:
            logger.warning(f"Boundary detection failed: {e}")
            return None
    
    async def _find_energy_boundary(self, segment: np.ndarray, sr: int, search_start: float, target_time: float) -> Optional[float]:
        """Find boundary based on energy levels (silence detection)"""
        try:
            # Calculate RMS energy in small windows
            window_size = int(0.1 * sr)  # 100ms windows
            hop_size = int(0.05 * sr)    # 50ms hop
            
            energies = []
            positions = []
            
            for i in range(0, len(segment) - window_size, hop_size):
                window = segment[i:i + window_size]
                energy = np.sqrt(np.mean(window ** 2))
                energies.append(energy)
                positions.append(search_start + i / sr)
            
            if not energies:
                return None
            
            # Find energy threshold (lower 25th percentile)
            energy_threshold = np.percentile(energies, 25)
            
            # Find positions with low energy near target
            target_idx = len(positions) // 2  # Middle of search window
            
            # Search around target position
            search_radius = min(len(positions) // 4, 10)  # Within reasonable distance
            
            best_boundary = None
            min_energy = float('inf')
            
            for i in range(max(0, target_idx - search_radius), 
                          min(len(energies), target_idx + search_radius)):
                if energies[i] < energy_threshold and energies[i] < min_energy:
                    min_energy = energies[i]
                    best_boundary = positions[i]
            
            return best_boundary
            
        except Exception as e:
            logger.debug(f"Energy boundary detection failed: {e}")
            return None
    
    async def _find_vad_boundary(self, segment: np.ndarray, sr: int, search_start: float, target_time: float) -> Optional[float]:
        """Find boundary using Voice Activity Detection"""
        if not self.vad:
            return None
        
        try:
            # Resample to 16kHz if needed (VAD requirement)
            if sr != 16000:
                if librosa is not None:
                    segment_16k = librosa.resample(segment, orig_sr=sr, target_sr=16000)
                else:
                    return None  # Cannot resample without librosa
            else:
                segment_16k = segment
            
            # Convert to 16-bit PCM
            pcm_data = (segment_16k * 32767).astype(np.int16).tobytes()
            
            # Process in VAD frames (30ms)
            frame_size = int(16000 * 0.03)  # 30ms at 16kHz
            frames = []
            positions = []
            
            for i in range(0, len(pcm_data), frame_size * 2):  # 2 bytes per sample
                frame = pcm_data[i:i + frame_size * 2]
                if len(frame) < frame_size * 2:
                    break
                
                is_speech = self.vad.is_speech(frame, 16000)
                frames.append(is_speech)
                positions.append(search_start + i / (2 * 16000))
            
            if not frames:
                return None
            
            # Find speech/non-speech transitions near target
            target_idx = len(positions) // 2
            search_radius = min(len(positions) // 4, 10)
            
            # Look for speech to non-speech transition (good boundary)
            for i in range(max(1, target_idx - search_radius), 
                          min(len(frames) - 1, target_idx + search_radius)):
                if frames[i-1] and not frames[i]:  # Speech to non-speech
                    return positions[i]
            
            return None
            
        except Exception as e:
            logger.debug(f"VAD boundary detection failed: {e}")
            return None
    
    async def _find_zcr_boundary(self, segment: np.ndarray, sr: int, search_start: float, target_time: float) -> Optional[float]:
        """Find boundary based on zero-crossing rate"""
        try:
            # Calculate zero-crossing rate in windows
            window_size = int(0.1 * sr)  # 100ms windows
            hop_size = int(0.05 * sr)    # 50ms hop
            
            zcrs = []
            positions = []
            
            for i in range(0, len(segment) - window_size, hop_size):
                window = segment[i:i + window_size]
                zcr = np.sum(np.diff(np.sign(window)) != 0) / len(window)
                zcrs.append(zcr)
                positions.append(search_start + i / sr)
            
            if not zcrs:
                return None
            
            # Find local minima in ZCR (likely silence/low activity)
            target_idx = len(positions) // 2
            search_radius = min(len(positions) // 4, 10)
            
            min_zcr = float('inf')
            best_boundary = None
            
            for i in range(max(1, target_idx - search_radius), 
                          min(len(zcrs) - 1, target_idx + search_radius)):
                if (zcrs[i] < zcrs[i-1] and zcrs[i] < zcrs[i+1] and 
                    zcrs[i] < min_zcr):
                    min_zcr = zcrs[i]
                    best_boundary = positions[i]
            
            return best_boundary
            
        except Exception as e:
            logger.debug(f"ZCR boundary detection failed: {e}")
            return None
    
    async def _create_chunk_file(self, audio_data: np.ndarray, sr: int, start_time: float, end_time: float, 
                                index: int, temp_dir: Path, total_duration: float) -> Dict:
        """Create individual chunk file with context information"""
        try:
            # Calculate sample indices
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            
            # Extract chunk audio
            chunk_audio = audio_data[start_sample:end_sample]
            
            # Create chunk file
            chunk_filename = f"chunk_{index:03d}_{int(start_time*1000)}_{int(end_time*1000)}.wav"
            chunk_path = temp_dir / chunk_filename
            
            # Save chunk to file
            if sf is not None:
                sf.write(str(chunk_path), chunk_audio, sr)
            else:
                raise AudioChunkingError("soundfile not available for saving chunk")
            
            # Determine context boundaries (for overlap handling)
            context_start = max(0, start_time - self.overlap_duration / 2)
            context_end = min(total_duration, end_time + self.overlap_duration / 2)
            
            chunk_info = {
                "index": index,
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
                "file_path": str(chunk_path),
                "has_overlap": index > 0,  # All chunks except first have overlap
                "context_start": context_start,
                "context_end": context_end,
                "sample_rate": sr,
                "samples": len(chunk_audio)
            }
            
            logger.debug(f"Created chunk {index}: {start_time:.2f}s - {end_time:.2f}s ({chunk_info['duration']:.2f}s)")
            return chunk_info
            
        except Exception as e:
            raise AudioChunkingError(f"Failed to create chunk file: {str(e)}")
    
    async def process_streaming_audio(self, audio_data: np.ndarray, sample_rate: int) -> AsyncGenerator[Dict, None]:
        """
        Process audio in real-time streaming chunks
        
        Args:
            audio_data: Audio data stream
            sample_rate: Sample rate
            
        Yields:
            Dict: Chunk information for processing
        """
        try:
            chunk_samples = int(self.chunk_duration * sample_rate)
            overlap_samples = int(self.overlap_duration * sample_rate)
            
            position = 0
            chunk_index = 0
            
            while position < len(audio_data):
                # Calculate chunk boundaries
                chunk_end = min(position + chunk_samples, len(audio_data))
                chunk_audio = audio_data[position:chunk_end]
                
                # Create temporary chunk file
                temp_dir = Path(FileManager.get_data_path("temp"))
                chunk_filename = f"stream_chunk_{chunk_index:03d}_{int(time.time() * 1000)}.wav"
                chunk_path = temp_dir / chunk_filename
                
                if sf is not None:
                    sf.write(str(chunk_path), chunk_audio, sample_rate)
                else:
                    raise AudioChunkingError("soundfile not available for saving stream chunk")
                
                # Create chunk info
                start_time = position / sample_rate
                end_time = chunk_end / sample_rate
                
                chunk_info = {
                    "index": chunk_index,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": end_time - start_time,
                    "file_path": str(chunk_path),
                    "has_overlap": chunk_index > 0,
                    "is_streaming": True,
                    "sample_rate": sample_rate
                }
                
                yield chunk_info
                
                # Move to next chunk with overlap
                position = max(position + chunk_samples - overlap_samples, position + 1)
                chunk_index += 1
                
        except Exception as e:
            logger.error(f"Streaming audio processing failed: {e}")
            raise AudioChunkingError(f"Streaming error: {str(e)}")
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.processing_times:
            return {"average_processing_time": 0, "total_chunks_processed": 0}
        
        return {
            "average_processing_time": np.mean(self.processing_times),
            "max_processing_time": np.max(self.processing_times),
            "min_processing_time": np.min(self.processing_times),
            "total_chunks_processed": len(self.processing_times),
            "real_time_ratio": np.mean(self.processing_times) / self.chunk_duration
        }
    
    async def cleanup_chunk_files(self, chunks: List[Dict]):
        """Clean up temporary chunk files"""
        cleaned_count = 0
        
        for chunk in chunks:
            try:
                chunk_path = Path(chunk["file_path"])
                if chunk_path.exists() and "temp" in str(chunk_path):
                    chunk_path.unlink()
                    cleaned_count += 1
            except Exception as e:
                logger.warning(f"Failed to cleanup chunk file {chunk.get('file_path', 'unknown')}: {e}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} temporary chunk files")

# Global instance for easy access
_global_audio_chunker: Optional[OptimizedAudioChunker] = None

def get_audio_chunker() -> OptimizedAudioChunker:
    """Get or create the global audio chunker instance"""
    global _global_audio_chunker
    if _global_audio_chunker is None:
        _global_audio_chunker = OptimizedAudioChunker()
    return _global_audio_chunker