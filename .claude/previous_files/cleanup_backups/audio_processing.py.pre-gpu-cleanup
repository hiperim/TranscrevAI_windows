"""
Enhanced Audio Processing Module - Optimized Chunking for ≤0.5:1 ratio
Includes adaptive chunking system merged from FASE 2 optimizations
FASE 3: GPU parallel processing support added
"""
import logging
import os
import asyncio
import time
# import concurrent.futures  # Removed for pure asyncio
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union

# Lazy imports for performance
_torch = None
_librosa = None
_soundfile = None

def _get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch

def _get_librosa():
    global _librosa
    if _librosa is None:
        import librosa
        _librosa = librosa
    return _librosa

def _get_soundfile():
    global _soundfile
    if _soundfile is None:
        import soundfile as sf
        _soundfile = sf
    return _soundfile

logger = logging.getLogger(__name__)

class DynamicMemoryManager:
    """ESTRATÉGIA 7: Dynamic memory allocation manager"""

    def __init__(self):
        self.allocated_buffers = []
        self.current_usage = 0
        self.max_buffer_size = 50 * 1024 * 1024  # FASE 1 FIX: 50MB chunks (was 100MB)

    def allocate_buffer(self, size_bytes: int) -> np.ndarray:
        """FASE 1 FIX: Memory-optimized buffer allocation"""
        try:
            # FASE 1: More conservative buffer sizing
            optimal_size = min(size_bytes, self.max_buffer_size)

            # FASE 1: Check total memory usage before allocation
            if self.current_usage + optimal_size > 200 * 1024 * 1024:  # 200MB limit
                logger.warning(f"FASE 1: Memory limit approaching, forcing cleanup")
                self.cleanup_all()

            logger.info(f"FASE 1: Allocating {optimal_size / 1024 / 1024:.1f}MB buffer (usage: {self.current_usage / 1024 / 1024:.1f}MB)")

            # Allocate buffer
            buffer = np.empty(optimal_size // 4, dtype=np.float32)  # 4 bytes per float32
            self.allocated_buffers.append(buffer)
            self.current_usage += optimal_size

            return buffer

        except Exception as e:
            logger.error(f"FASE 1: Buffer allocation failed: {e}")
            # FASE 1: Even more conservative fallback
            return np.empty(size_bytes // 16, dtype=np.float32)  # Smaller fallback

    def deallocate_buffer(self, buffer: np.ndarray) -> None:
        """Deallocate buffer when no longer needed"""
        try:
            if buffer in self.allocated_buffers:
                self.allocated_buffers.remove(buffer)
                buffer_size = buffer.nbytes
                self.current_usage -= buffer_size
                logger.info(f"ESTRATÉGIA 7: Deallocated {buffer_size / 1024 / 1024:.1f}MB buffer")

            # Force garbage collection
            del buffer
            import gc
            gc.collect()

        except Exception as e:
            logger.warning(f"ESTRATÉGIA 7: Buffer deallocation warning: {e}")

    def get_current_usage(self) -> float:
        """Get current memory usage in GB"""
        return self.current_usage / (1024 * 1024 * 1024)

    def cleanup_all(self) -> None:
        """Emergency cleanup of all buffers"""
        logger.info("ESTRATÉGIA 7: Emergency cleanup of all buffers")
        for buffer in self.allocated_buffers:
            del buffer
        self.allocated_buffers.clear()
        self.current_usage = 0
        import gc
        gc.collect()

# Global memory manager instance
dynamic_memory_manager = DynamicMemoryManager()

class OptimizedAudioProcessor:
    """Enhanced audio utilities with VAD and optimized parallel processing for ≤0.5:1 ratio"""

    @staticmethod
    def torchaudio_get_duration(audio_path: str) -> float:
        """FASE 1 OPT 1: Ultra-fast duration extraction using torchaudio"""
        try:
            import torchaudio
            info = torchaudio.info(audio_path)
            return info.num_frames / info.sample_rate
        except Exception as e:
            logger.warning(f"FASE 1: torchaudio duration failed: {e}, using soundfile fallback")
            try:
                with _get_soundfile().SoundFile(audio_path) as f:
                    return len(f) / f.samplerate
            except Exception as e2:
                logger.error(f"FASE 1: All duration methods failed: {e2}")
                return 30.0  # Safe fallback

    @staticmethod
    def torchaudio_optimized_resample(audio_data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """FASE 1 OPT 1: Optimized resampling using torchaudio tensors"""
        try:
            import torchaudio
            import torch

            # Convert to tensor efficiently
            audio_tensor = _get_torch().from_numpy(audio_data).float()
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

            # Use torchaudio's optimized resampler
            resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
            resampled = resampler(audio_tensor)

            return resampled.squeeze().numpy()

        except Exception as e:
            logger.warning(f"FASE 1: torchaudio resample failed: {e}, using basic fallback")
            # Simple fallback using basic interpolation
            from scipy import signal
            import numpy as np
            return np.array(signal.resample(audio_data, int(len(audio_data) * target_sr / orig_sr)))

    @staticmethod
    def optimized_audio_load(audio_path: str, target_sr: int = 16000) -> Tuple[Any, int]:
        """Optimized audio loading with smart resampling - FASE 2.2 optimization"""
        try:
            import torchaudio
            import torch

            # First, check the original sample rate without loading the full audio
            info = _get_soundfile().info(audio_path)
            original_sr = info.samplerate

            logger.info(f"FASE 2.2: Audio SR {original_sr}Hz → target {target_sr}Hz")

            # Load audio with torchaudio (more stable)
            audio_tensor, sr = torchaudio.load(audio_path)

            # Convert to mono if needed
            if audio_tensor.shape[0] > 1:
                audio_tensor = _get_torch().mean(audio_tensor, dim=0, keepdim=True)

            # Convert to numpy
            audio_data = audio_tensor.squeeze().numpy()

            # If already at target sample rate, no resampling needed
            if sr == target_sr:
                logger.info("FASE 2.2: No resampling needed - optimal loading")
                return audio_data, sr

            # If close to target (±10%), use original to avoid quality loss
            elif abs(sr - target_sr) / target_sr <= 0.1:
                logger.info(f"FASE 2.2: SR close enough ({sr}Hz), using original")
                return audio_data, sr

            # Otherwise, resample efficiently using torchaudio
            else:
                logger.info(f"FASE 2.2: Resampling {sr}Hz → {target_sr}Hz")
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
                resampled = resampler(audio_tensor)
                audio_data = resampled.squeeze().numpy()
                return audio_data, target_sr

        except Exception as e:
            logger.warning(f"FASE 1: Optimized torchaudio load failed: {e}, using soundfile fallback")
            # FASE 1 OPT 1: Remove librosa dependency, use soundfile
            try:
                audio_data, sr = _get_soundfile().read(audio_path)
                if audio_data.ndim > 1:
                    audio_data = audio_data.mean(axis=1)  # Convert to mono

                # Resample if needed using optimized torchaudio
                if sr != target_sr:
                    audio_data = OptimizedAudioProcessor.torchaudio_optimized_resample(audio_data, sr, target_sr)
                    sr = target_sr

                return audio_data, sr
            except Exception as e2:
                logger.error(f"FASE 1: All audio loading methods failed: {e2}")
                raise e2

    @staticmethod
    def normalize_audio_optimized(audio_data: Any, method: str = "peak") -> Any:
        """Optimized audio normalization - FASE 2.2 optimization"""
        try:
            if method == "peak":
                # Peak normalization - fastest
                max_val = abs(audio_data).max()
                if max_val > 0:
                    normalized = audio_data / max_val * 0.95  # Leave 5% headroom
                    logger.info("FASE 2.2: Peak normalization applied")
                    return normalized
                return audio_data

            elif method == "rms":
                # RMS normalization - better quality but slower
                import numpy as np
                rms = np.sqrt(np.mean(audio_data**2))
                if rms > 0:
                    target_rms = 0.2  # Target RMS level
                    normalized = audio_data * (target_rms / rms)
                    # Apply soft limiting
                    normalized = np.tanh(normalized)
                    logger.info("FASE 2.2: RMS normalization applied")
                    return normalized
                return audio_data

            else:
                logger.warning(f"FASE 2.2: Unknown normalization method: {method}")
                return audio_data

        except Exception as e:
            logger.warning(f"FASE 2.2: Normalization failed: {e}")
            return audio_data

    @staticmethod
    def memory_mapped_audio_load(audio_path: str, target_sr: int = 16000) -> Tuple[Any, int]:
        """ESTRATÉGIA 4: Memory-mapped audio loading to reduce RAM usage"""
        try:
            import tempfile

            logger.info(f"ESTRATÉGIA 4: Memory-mapped loading for {audio_path}")

            # Get audio info without loading full file
            info = _get_soundfile().info(audio_path)
            original_sr = info.samplerate
            duration = info.frames / info.samplerate

            logger.info(f"Audio info: {duration:.1f}s, {original_sr}Hz, {info.frames} frames")

            # For short audio (<30s), use standard loading (memory map overhead not worth it)
            if duration < 30:
                logger.info("Short audio - using standard loading")
                return OptimizedAudioProcessor.optimized_audio_load(audio_path, target_sr)

            # Create temporary memory-mapped file for processing
            with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as temp_file:
                temp_path = temp_file.name

            try:
                # Load and convert to target format in chunks to save memory
                chunk_size = target_sr * 10  # 10 second chunks

                # Open memory mapped array for writing
                total_samples = int(duration * target_sr)
                mapped_audio = np.memmap(temp_path, dtype=np.float32, mode='w+', shape=(total_samples,))

                # Process audio in chunks
                write_offset = 0

                with _get_soundfile().SoundFile(audio_path) as f:
                    while True:
                        # Read chunk
                        audio_chunk = f.read(chunk_size, dtype='float32')
                        if len(audio_chunk) == 0:
                            break

                        # Convert to mono if needed
                        if len(audio_chunk.shape) > 1:
                            audio_chunk = np.mean(audio_chunk, axis=1)

                        # FASE 1 OPT 1: Resample if needed using optimized torchaudio
                        if original_sr != target_sr:
                            audio_chunk = OptimizedAudioProcessor.torchaudio_optimized_resample(audio_chunk, original_sr, target_sr)

                        # Write to memory map
                        chunk_len = len(audio_chunk)
                        if write_offset + chunk_len <= total_samples:
                            mapped_audio[write_offset:write_offset + chunk_len] = audio_chunk
                            write_offset += chunk_len
                        else:
                            # Last chunk - trim to fit
                            remaining = total_samples - write_offset
                            mapped_audio[write_offset:] = audio_chunk[:remaining]
                            break

                logger.info(f"ESTRATÉGIA 4: Memory-mapped file created: {total_samples} samples")

                # Return memory-mapped array (read-only access)
                final_mapped = np.memmap(temp_path, dtype=np.float32, mode='r', shape=(total_samples,))
                return final_mapped, target_sr

            except Exception as e:
                # Cleanup on error
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise e

        except Exception as e:
            logger.warning(f"ESTRATÉGIA 4: Memory-mapped loading failed: {e}, falling back")
            return OptimizedAudioProcessor.optimized_audio_load(audio_path, target_sr)

    @staticmethod
    async def get_audio_duration(audio_file: str) -> float:
        """Get audio duration using fastest available method"""
        try:
            # FASE 1 OPT 1: Use optimized torchaudio duration
            duration = OptimizedAudioProcessor.torchaudio_get_duration(audio_file)
            return duration
        except ImportError:
            try:
                import soundfile as sf
                with _get_soundfile().SoundFile(audio_file) as f:
                    return len(f) / f.samplerate
            except ImportError:
                try:
                    import subprocess
                    result = subprocess.run([
                        'ffprobe', '-v', 'quiet', '-show_entries', 
                        'format=duration', '-of', 'csv=p=0', audio_file
                    ], capture_output=True, text=True, timeout=5)
                    return float(result.stdout.strip())
                except:
                    logger.warning("Could not get audio duration - using fallback")
                    return 30.0  # Fallback duration
        except Exception as e:
            logger.error(f"Error getting audio duration: {e}")
            return 30.0

    @staticmethod 
    def get_optimal_sample_rate(language: str = "pt") -> int:
        """Get optimal sample rate for language (8kHz for speed, 16kHz for accuracy)"""
        # Start with 8kHz for maximum speed - fallback to 16kHz if accuracy drops
        return 8000  # ULTRA-FAST processing optimized for ≤0.5:1 ratio
    
    @staticmethod
    def apply_vad_preprocessing(audio_path: str, output_path: Union[str, None] = None) -> str:
        """Apply Voice Activity Detection to remove silence and optimize processing speed"""
        try:
            import librosa
            import numpy as np
            import soundfile as sf
            
            logger.info("Applying VAD preprocessing to optimize processing speed")
            
            # Load audio with optimal sample rate
            audio, sr = OptimizedAudioProcessor.optimized_audio_load(audio_path, 16000)  # 16kHz for VAD accuracy
            
            # Simple but effective VAD using energy-based approach
            frame_length = int(0.025 * sr)  # 25ms frames
            hop_length = int(0.01 * sr)     # 10ms hop
            
            # FASE 1 OPT 1: Calculate frame energy using numpy (remove librosa dependency)
            # Create frames manually with numpy for better performance
            num_frames = 1 + (len(audio) - frame_length) // hop_length
            frames = np.zeros((frame_length, num_frames))
            for i in range(num_frames):
                start = i * hop_length
                end = start + frame_length
                if end <= len(audio):
                    frames[:, i] = audio[start:end]
            energy = np.sum(frames**2, axis=0)
            
            # Adaptive threshold (top 20% of energy levels)
            energy_threshold = np.percentile(energy, 80)
            
            # Create voice activity mask
            voice_frames = energy > energy_threshold
            
            # Apply morphological operations to clean up mask
            from scipy import ndimage
            voice_frames = ndimage.binary_opening(voice_frames, iterations=1)
            voice_frames = ndimage.binary_closing(voice_frames, iterations=2)
            
            # Extract voice segments with small padding
            padding_frames = int(0.1 * sr / hop_length)  # 100ms padding
            
            voice_segments = []
            in_voice = False
            start_frame = 0
            
            for i, is_voice in enumerate(voice_frames):
                if is_voice and not in_voice:
                    start_frame = max(0, i - padding_frames)
                    in_voice = True
                elif not is_voice and in_voice:
                    end_frame = min(len(voice_frames), i + padding_frames)
                    voice_segments.append((start_frame, end_frame))
                    in_voice = False
            
            # Handle case where audio ends with voice
            if in_voice is not None and in_voice:
                voice_segments.append((start_frame, len(voice_frames)))
            
            # Extract and concatenate voice segments
            voice_audio = []
            for start_frame, end_frame in voice_segments:
                start_sample = start_frame * hop_length
                end_sample = min(len(audio), end_frame * hop_length)
                voice_audio.extend(audio[start_sample:end_sample])
            
            if not voice_audio:
                logger.warning("VAD removed all audio - using original")
                return audio_path
            
            voice_audio = np.array(voice_audio)
            
            # Save processed audio
            if output_path is None:
                output_path = audio_path.replace('.wav', '_vad.wav')
            
            _get_soundfile().write(output_path, voice_audio, sr)
            
            original_duration = len(audio) / sr
            processed_duration = len(voice_audio) / sr
            reduction_percent = (1 - processed_duration / original_duration) * 100
            
            logger.info(f"VAD preprocessing completed: {original_duration:.1f}s → {processed_duration:.1f}s "
                       f"({reduction_percent:.1f}% reduction)")
            
            return output_path
            
        except Exception as e:
            logger.error(f"VAD preprocessing failed: {e}")
            return audio_path  # Return original if VAD fails

class GPUParallelProcessor:
    """FASE 3: GPU Parallel chunk processing for ≤0.5:1 ratio target"""
    
    @staticmethod
    def can_use_gpu_parallel() -> bool:
        """Check if GPU parallel processing is available"""
        try:
            return _get_torch().cuda.is_available() and _get_torch().cuda.device_count() > 0
        except:
            return False
    
    @staticmethod
    async def process_chunks_parallel_gpu(model, chunks_with_timestamps: List[Tuple], language: str) -> Dict[str, Any]:
        """Process multiple chunks in parallel using GPU streams for ≤0.5:1 target"""
        try:
            if not GPUParallelProcessor.can_use_gpu_parallel():
                logger.info("FASE 3: GPU not available, falling back to sequential processing")
                return await GPUParallelProcessor._process_sequential_fallback(model, chunks_with_timestamps, language)
            
            from .hardware_optimization import phase3_optimizer
            
            # Get CUDA streams for parallel processing
            if not hasattr(phase3_optimizer, 'cuda_streams') or not phase3_optimizer.cuda_streams:
                logger.warning("FASE 3: CUDA streams not initialized, using sequential fallback")
                return await GPUParallelProcessor._process_sequential_fallback(model, chunks_with_timestamps, language)
            
            logger.info(f"FASE 3: Processing {len(chunks_with_timestamps)} chunks in parallel on GPU")
            
            # Batch chunks for parallel processing
            max_concurrent = len(phase3_optimizer.cuda_streams)
            chunk_batches = [chunks_with_timestamps[i:i + max_concurrent] 
                           for i in range(0, len(chunks_with_timestamps), max_concurrent)]
            
            all_segments = []
            total_text = ""
            
            for batch_idx, batch in enumerate(chunk_batches):
                logger.info(f"FASE 3: Processing batch {batch_idx + 1}/{len(chunk_batches)} ({len(batch)} chunks)")
                
                # Process batch in parallel
                batch_results = await GPUParallelProcessor._process_batch_parallel(
                    model, batch, language, phase3_optimizer.cuda_streams
                )
                
                # Collect results
                for result_data in batch_results:
                    if result_data and "segments" in result_data:
                        all_segments.extend(result_data["segments"])
                    if result_data and "text" in result_data:
                        chunk_text = result_data["text"].strip()
                        if chunk_text is not None and chunk_text:
                            total_text += " " + chunk_text
                
                # Memory cleanup between batches
                _get_torch().cuda.empty_cache()
            
            # Deduplicate and clean results
            final_text = AdaptiveChunker._deduplicate_segments_text(total_text)
            
            return {
                "segments": all_segments,
                "text": final_text.strip(),
                "processing_info": {
                    "parallel_batches": len(chunk_batches),
                    "total_chunks": len(chunks_with_timestamps),
                    "gpu_accelerated": True
                }
            }
            
        except Exception as e:
            logger.error(f"FASE 3: GPU parallel processing failed: {e}")
            return await GPUParallelProcessor._process_sequential_fallback(model, chunks_with_timestamps, language)
    
    @staticmethod
    async def _process_batch_parallel(model, batch: List[Tuple], language: str, cuda_streams: List) -> List[Dict]:
        """Process a batch of chunks in parallel using CUDA streams"""
        try:
            import asyncio
            
            # Create tasks for parallel processing
            tasks = []
            for i, (chunk, timestamps) in enumerate(batch):
                stream_idx = i % len(cuda_streams)
                stream = cuda_streams[stream_idx]
                
                task = GPUParallelProcessor._process_single_chunk_async(
                    model, chunk, timestamps, language, stream
                )
                tasks.append(task)
            
            # Wait for all chunks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter successful results
            successful_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"FASE 3: Chunk {i} failed: {result}")
                else:
                    successful_results.append(result)
            
            return successful_results
            
        except Exception as e:
            logger.error(f"FASE 3: Batch processing failed: {e}")
            return []
    
    @staticmethod
    async def _process_single_chunk_async(model, chunk, timestamps, language: str, cuda_stream) -> Dict:
        """Process a single chunk asynchronously with CUDA stream"""
        try:
            start_time, end_time = timestamps
            
            # Run in executor to avoid blocking - converted to pure asyncio
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,  # Use default executor
                GPUParallelProcessor._process_chunk_with_stream,
                model, chunk, language, cuda_stream, start_time
            )
            
            return result
            
        except Exception as e:
            logger.warning(f"FASE 3: Async chunk processing failed: {e}")
            return {"segments": [], "text": ""}
    
    @staticmethod
    def _process_chunk_with_stream(model, chunk, language: str, cuda_stream, start_time: float) -> Dict:
        """Process chunk with specific CUDA stream for parallel execution"""
        try:
            # Get optimized parameters
            options = AdaptiveChunker._get_chunk_parameters(language)
            
            # Use CUDA stream for this operation
            with _get_torch().cuda.stream(cuda_stream):
                # Move chunk to GPU with stream
                if _get_torch().cuda.is_available():
                    chunk_tensor = _get_torch().from_numpy(chunk).cuda(non_blocking=True)
                    chunk = chunk_tensor.cpu().numpy()  # Whisper expects numpy
                
                # Process with model
                result = model.transcribe(chunk, **options)
                
                # Adjust timestamps
                if result and "segments" in result:
                    for segment in result["segments"]:
                        segment["start"] += start_time
                        segment["end"] += start_time
            
            # Synchronize stream
            cuda_stream.synchronize()
            
            return result
            
        except Exception as e:
            logger.warning(f"FASE 3: Stream chunk processing failed: {e}")
            return {"segments": [], "text": ""}
    
    @staticmethod
    async def _process_sequential_fallback(model, chunks_with_timestamps: List[Tuple], language: str) -> Dict[str, Any]:
        """Fallback to sequential processing if GPU parallel fails"""
        try:
            logger.info("FASE 3: Using sequential fallback processing")
            
            all_segments = []
            total_text = ""
            
            for i, (chunk, (start_time, end_time)) in enumerate(chunks_with_timestamps):
                logger.info(f"FASE 3: Processing chunk {i+1}/{len(chunks_with_timestamps)} (fallback)")

                # ESTRATÉGIA 7: Dynamic memory allocation for chunk processing
                chunk_size = len(chunk) * 4 if hasattr(chunk, '__len__') else 1024 * 1024  # Estimate size
                processing_buffer = dynamic_memory_manager.allocate_buffer(chunk_size)

                try:
                    options = AdaptiveChunker._get_chunk_parameters(language)
                    result = model.transcribe(chunk, **options)
                finally:
                    # Always deallocate buffer after processing
                    dynamic_memory_manager.deallocate_buffer(processing_buffer)
                
                if result and "segments" in result:
                    for segment in result["segments"]:
                        segment["start"] += start_time
                        segment["end"] += start_time
                    all_segments.extend(result["segments"])
                
                if result and "text" in result:
                    chunk_text = result["text"].strip()
                    if chunk_text is not None and chunk_text:
                        total_text += " " + chunk_text
            
            final_text = AdaptiveChunker._deduplicate_segments_text(total_text)
            
            return {
                "segments": all_segments,
                "text": final_text.strip(),
                "processing_info": {
                    "parallel_batches": 0,
                    "total_chunks": len(chunks_with_timestamps),
                    "gpu_accelerated": False
                }
            }
            
        except Exception as e:
            logger.error(f"FASE 3: Sequential fallback failed: {e}")
            return {"segments": [], "text": ""}

class AdaptiveChunker:
    """Enhanced chunking system for FASE 2 optimizations - merged from chunking_phase2.py"""
    
    @staticmethod
    def get_optimal_chunk_threshold(duration: float) -> float:
        """FASE 2.3: Intelligent chunk threshold based on audio duration and performance targets"""
        # Updated logic for better performance/accuracy balance
        if duration <= 15:
            return 30  # No chunking for short audio - process as single
        elif duration <= 45:
            return 20  # 20s threshold for medium audio
        elif duration <= 90:
            return 12  # 12s threshold for longer audio
        elif duration <= 180:
            return 8   # 8s threshold for very long audio
        else:
            return 6   # 6s aggressive threshold for extremely long audio
    
    @staticmethod
    def get_optimal_chunk_size(duration: float) -> float:
        """FASE 2.3: Intelligent chunk size for optimal processing speed"""
        # Optimize chunk sizes for medium model in memory-constrained environments
        # Research-backed: 10-12s chunks optimal for medium model speed and accuracy
        if duration <= 30:
            return 10.0  # 10s chunks for short-medium audio
        elif duration <= 90:
            return 10.0  # 10s chunks for medium audio (consistent performance)
        elif duration <= 180:
            return 8.0   # 8s chunks for longer audio (memory efficiency)
        else:
            return 6.0   # 6s chunks for very long audio (balance speed/accuracy)
    
    @staticmethod
    def should_use_chunking(audio_path: str) -> Tuple[bool, float]:
        """Determine if chunking should be used and what threshold"""
        try:
            # FASE 1 OPT 1: Use optimized torchaudio duration
            duration = OptimizedAudioProcessor.torchaudio_get_duration(audio_path)
            threshold = AdaptiveChunker.get_optimal_chunk_threshold(duration)
            should_chunk = duration > threshold
            
            logger.info(f"FASE 2: Audio {duration:.1f}s, threshold {threshold}s, chunking: {should_chunk}")
            return should_chunk, duration
            
        except Exception as e:
            logger.warning(f"FASE 2: Duration check failed: {e}")
            return False, 0.0
    
    @staticmethod
    def process_with_enhanced_chunking(model, audio_path: str, language: str, duration: float) -> Dict[str, Any]:
        """Process audio with FASE 2 enhanced chunking + FASE 3 GPU parallel processing"""
        try:
            logger.info(f"FASE 2+3: Processing {duration:.1f}s audio with optimized chunking")
            
            chunk_duration = AdaptiveChunker.get_optimal_chunk_size(duration)
            overlap_duration = 1.0  # 1 second overlap
            
            logger.info(f"FASE 2: Using {chunk_duration}s chunks with {overlap_duration}s overlap")
            
            # ESTRATÉGIA 4 + 7: Memory-mapped loading with dynamic allocation
            optimal_sr = 16000

            # Use memory-mapped loading for better memory efficiency
            audio_data, sr = OptimizedAudioProcessor.memory_mapped_audio_load(audio_path, optimal_sr)

            # Apply normalization optimization
            audio_data = OptimizedAudioProcessor.normalize_audio_optimized(audio_data, method="peak")

            logger.info(f"ESTRATÉGIA 4+7: Audio loaded with memory optimization")
            logger.info(f"Dynamic memory manager usage: {dynamic_memory_manager.get_current_usage():.2f}GB")
            
            # Create overlapping chunks
            chunks, timestamps = AdaptiveChunker._create_overlapping_chunks(
                audio_data, int(sr), chunk_duration, overlap_duration
            )
            
            logger.info(f"FASE 2: Created {len(chunks)} optimized chunks")
            
            # FASE 3: Try GPU parallel processing for ≤0.5:1 ratio target
            if GPUParallelProcessor.can_use_gpu_parallel() and len(chunks) > 1:
                logger.info("FASE 3: Attempting GPU parallel chunk processing for ≤0.5:1 ratio")
                try:
                    # Prepare chunks with timestamps for parallel processing
                    chunks_with_timestamps = list(zip(chunks, timestamps))
                    
                    # FASE 2 FIX: Improved async processing with event loop management
                    import asyncio
                    result = asyncio.run(
                        GPUParallelProcessor.process_chunks_parallel_gpu(
                            model, chunks_with_timestamps, language
                        )
                    )
                    # FASE 1: Add garbage collection
                    import gc
                    gc.collect()
                    logger.info("FASE 1: Garbage collection after sync GPU processing")
                        
                    if result and result.get("processing_info", {}).get("gpu_accelerated", False):
                        logger.info("FASE 3: GPU parallel processing completed successfully")
                        return result
                    else:
                        logger.info("FASE 3: GPU parallel processing failed, using sequential fallback")
                            
                except Exception as e:
                    logger.warning(f"FASE 3: GPU parallel processing failed: {e}, using sequential")
            
            # FASE 2: Sequential processing fallback
            logger.info("FASE 2: Using sequential chunk processing")
            all_segments = []
            total_text = ""
            
            for i, (chunk, (start_time, end_time)) in enumerate(zip(chunks, timestamps)):
                logger.info(f"FASE 2: Processing chunk {i+1}/{len(chunks)} ({start_time:.1f}s-{end_time:.1f}s)")
                
                # Memory cleanup
                import gc
                gc.collect()
                if _get_torch().cuda.is_available():
                    _get_torch().cuda.empty_cache()
                
                # Get optimized parameters for chunk
                options = AdaptiveChunker._get_chunk_parameters(language)
                
                # Process chunk
                try:
                    result = model.transcribe(chunk, **options)
                    
                    if result and "segments" in result:
                        # Adjust timestamps
                        for segment in result["segments"]:
                            segment["start"] += start_time
                            segment["end"] += start_time
                        all_segments.extend(result["segments"])
                    
                    if result and "text" in result:
                        chunk_text = result["text"].strip()
                        if chunk_text is not None and chunk_text:
                            total_text += " " + chunk_text
                            
                except Exception as e:
                    logger.warning(f"FASE 2: Chunk {i+1} processing failed: {e}")
                    continue
            
            # Deduplicate overlapping content
            all_segments = AdaptiveChunker._deduplicate_segments(all_segments, overlap_duration)
            
            result_dict = {
                "segments": all_segments,
                "text": total_text.strip(),
                "language": language
            }
            
            logger.info(f"FASE 2: Enhanced chunking completed - {len(all_segments)} segments")
            return result_dict
            
        except Exception as e:
            logger.error(f"FASE 2: Enhanced chunking failed: {e}")
            raise
    
    @staticmethod
    def _create_overlapping_chunks(audio_data, sr: int, chunk_duration: float, overlap_duration: float):
        """Create overlapping audio chunks"""
        chunk_size = int(chunk_duration * sr)
        overlap_size = int(overlap_duration * sr)
        step_size = chunk_size - overlap_size
        
        chunks = []
        timestamps = []
        
        start = 0
        while start < len(audio_data):
            end = min(start + chunk_size, len(audio_data))
            chunk = audio_data[start:end]
            
            # Only process chunks with meaningful audio (>0.5s)
            if len(chunk) > sr * 0.5:
                chunks.append(chunk)
                start_time = start / sr
                end_time = end / sr
                timestamps.append((start_time, end_time))
            
            if end >= len(audio_data):
                break
                
            start += step_size
        
        return chunks, timestamps
    
    @staticmethod
    def _get_chunk_parameters(language: str) -> Dict[str, Any]:
        """Get optimized parameters for chunk processing"""
        try:
            # Try to get FASE 1 optimized parameters
            from .model_parameters import get_optimized_params
            options = get_optimized_params(use_phase1=True)
            
            # FASE 2 specific overrides for chunks
            options.update({
                "fp16": _get_torch().cuda.is_available(),
                "beam_size": 1,
                "best_of": 1,
                "temperature": 0.0,
                "condition_on_previous_text": False,
                "word_timestamps": False,
                "without_timestamps": True,
            })
            
            return options
            
        except Exception as e:
            logger.warning(f"FASE 2: Parameter optimization failed: {e} - using safe defaults")
            # Safe fallback parameters
            return {
                "language": language,
                "task": "transcribe",
                "fp16": _get_torch().cuda.is_available(),
                "verbose": False,
                "beam_size": 1,
                "best_of": 1,
                "temperature": 0.0,
                "condition_on_previous_text": False,
                "compression_ratio_threshold": 2.0,
                "logprob_threshold": -0.5,
                "no_speech_threshold": 0.8,
                "word_timestamps": False,
                "without_timestamps": True,
            }
    
    @staticmethod
    def _deduplicate_segments(segments: List[Dict], overlap_duration: float) -> List[Dict]:
        """Remove duplicate content from overlapping chunks"""
        try:
            if not segments:
                return segments
            
            # Sort by start time
            segments.sort(key=lambda x: x.get("start", 0))
            
            deduplicated = []
            last_end_time = -1
            
            for segment in segments:
                start_time = segment.get("start", 0)
                end_time = segment.get("end", 0)
                
                # Skip segments entirely within overlap region
                if start_time < last_end_time - overlap_duration * 0.5:
                    continue
                
                deduplicated.append(segment)
                last_end_time = max(last_end_time, end_time)
            
            logger.info(f"FASE 2: Deduplication: {len(segments)} -> {len(deduplicated)} segments")
            return deduplicated
            
        except Exception as e:
            logger.warning(f"FASE 2: Deduplication failed: {e} - returning original")
            return segments
    
    @staticmethod
    def _deduplicate_segments_text(text: str) -> str:
        """Remove duplicate sentences and phrases from text"""
        try:
            if not text or not isinstance(text, str):
                return ""
            
            # Split by sentence-like patterns
            sentences = text.replace('. ', '.\n').replace('? ', '?\n').replace('! ', '!\n').split('\n')
            deduplicated = []
            seen = set()
            
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and sentence.lower() not in seen:
                    deduplicated.append(sentence)
                    seen.add(sentence.lower())
            
            return ' '.join(deduplicated)
            
        except Exception as e:
            logger.warning(f"Text deduplication failed: {e}")
            return text if isinstance(text, str) else ""

# Global instances for compatibility
audio_utils = OptimizedAudioProcessor()
adaptive_chunker = AdaptiveChunker()

# Backwards compatibility aliases
Phase2Chunker = AdaptiveChunker  # For existing imports

class StreamingAudioProcessor:
    """ESTRATÉGIA 2: Streaming Audio Processing for memory optimization

    Processes audio in sequential chunks with minimal memory footprint.
    Target: 60-80% memory reduction while preserving accuracy.
    """

    def __init__(self, chunk_duration: float = 8.0, overlap_duration: float = 1.5):
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.max_chunks_in_memory = 2  # Keep only 2 chunks in RAM
        self.memory_manager = DynamicMemoryManager()

        logger.info(f"ESTRATÉGIA 2: StreamingAudioProcessor initialized - {chunk_duration}s chunks, {overlap_duration}s overlap")

    async def process_audio_streaming(self, model, audio_path: str, language: str) -> Dict[str, Any]:
        """Main streaming processing method with memory optimization"""
        try:
            logger.info(f"ESTRATÉGIA 2: Starting streaming processing for {audio_path}")

            # Get audio info without loading full file
            audio_info = self._get_audio_info(audio_path)
            duration = audio_info['duration']
            sample_rate = audio_info['sample_rate']

            logger.info(f"ESTRATÉGIA 2: Audio duration {duration:.1f}s, SR {sample_rate}Hz")

            # Calculate chunk parameters
            chunk_samples = int(self.chunk_duration * sample_rate)
            overlap_samples = int(self.overlap_duration * sample_rate)
            step_samples = chunk_samples - overlap_samples

            # Initialize result collectors
            all_segments = []
            full_text = ""
            chunk_count = 0

            # Process audio in streaming chunks
            with _get_soundfile().SoundFile(audio_path) as audio_file:
                logger.info(f"ESTRATÉGIA 2: Processing {duration:.1f}s audio in {self.chunk_duration}s chunks")

                current_position = 0
                previous_chunk = None

                while current_position < audio_file.frames:
                    # Calculate chunk boundaries
                    chunk_start = current_position
                    chunk_end = min(chunk_start + chunk_samples, audio_file.frames)

                    # Read chunk from disk (streaming approach)
                    audio_file.seek(chunk_start)
                    chunk_data = audio_file.read(chunk_end - chunk_start)

                    if len(chunk_data) < sample_rate * 0.5:  # Skip chunks < 0.5s
                        break

                    # Convert to float32 and normalize
                    if chunk_data.ndim > 1:
                        chunk_data = chunk_data.mean(axis=1)
                    chunk_data = chunk_data.astype(np.float32)

                    # Apply memory-optimized normalization
                    chunk_data = self._normalize_chunk_streaming(chunk_data)

                    # Calculate timestamps
                    start_time = chunk_start / sample_rate
                    end_time = chunk_end / sample_rate

                    logger.info(f"ESTRATÉGIA 2: Processing chunk {chunk_count + 1} ({start_time:.1f}s-{end_time:.1f}s)")

                    # Process chunk with context preservation
                    chunk_result = await self._process_chunk_with_context(
                        model, chunk_data, previous_chunk, language, start_time, chunk_count
                    )

                    # Collect results
                    if chunk_result is not None and chunk_result:
                        segments = chunk_result.get('segments', [])
                        text = chunk_result.get('text', '')

                        # Adjust timestamps and collect segments
                        for segment in segments:
                            segment['start'] += start_time
                            segment['end'] += start_time
                        all_segments.extend(segments)

                        if text.strip():
                            full_text += " " + text.strip()

                    # Memory management: keep only current chunk
                    previous_chunk = chunk_data[-overlap_samples:] if len(chunk_data) > overlap_samples else None
                    del chunk_data  # Explicit cleanup

                    # Force garbage collection every few chunks
                    if chunk_count % 3 == 0:
                        import gc
                        gc.collect()
                        if _get_torch().cuda.is_available():
                            _get_torch().cuda.empty_cache()

                    # Move to next chunk
                    current_position += step_samples
                    chunk_count += 1

            # Post-process results
            final_segments = self._deduplicate_streaming_segments(all_segments)
            final_text = self._clean_streaming_text(full_text)

            result = {
                "segments": final_segments,
                "text": final_text.strip(),
                "language": language,
                "processing_info": {
                    "streaming_chunks": chunk_count,
                    "memory_optimized": True,
                    "strategy": "ESTRATÉGIA 2"
                }
            }

            logger.info(f"ESTRATÉGIA 2: Streaming processing completed - {chunk_count} chunks, {len(final_segments)} segments")
            return result

        except Exception as e:
            logger.error(f"ESTRATÉGIA 2: Streaming processing failed: {e}")
            raise

    def _get_audio_info(self, audio_path: str) -> Dict[str, Any]:
        """Get audio information without loading full file"""
        try:
            with _get_soundfile().SoundFile(audio_path) as f:
                return {
                    'duration': len(f) / f.samplerate,
                    'sample_rate': f.samplerate,
                    'channels': f.channels,
                    'frames': len(f)
                }
        except Exception as e:
            logger.error(f"ESTRATÉGIA 2: Failed to get audio info: {e}")
            raise

    def _normalize_chunk_streaming(self, chunk_data: np.ndarray) -> np.ndarray:
        """Memory-efficient chunk normalization"""
        try:
            # Peak normalization (memory efficient)
            max_val = np.abs(chunk_data).max()
            if max_val > 0:
                chunk_data = chunk_data / max_val * 0.95

            return chunk_data

        except Exception as e:
            logger.warning(f"ESTRATÉGIA 2: Chunk normalization failed: {e}")
            return chunk_data

    async def _process_chunk_with_context(self, model, chunk_data: np.ndarray,
                                        previous_chunk: Union[np.ndarray, None],
                                        language: str, start_time: float,
                                        chunk_index: int) -> Dict[str, Any]:
        """Process chunk with context preservation from previous chunk"""
        try:
            # Prepare audio data with context
            if previous_chunk is not None and chunk_index > 0:
                # Add context from previous chunk for better accuracy
                context_data = np.concatenate([previous_chunk, chunk_data])
                logger.debug(f"ESTRATÉGIA 2: Added {len(previous_chunk)} samples context to chunk {chunk_index}")
            else:
                context_data = chunk_data

            # Get optimized transcription parameters
            options = self._get_streaming_parameters(language)

            # Run transcription in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._transcribe_chunk_sync, model, context_data, options
            )

            # Filter segments to remove context overlap if added
            if previous_chunk is not None and chunk_index > 0:
                context_duration = len(previous_chunk) / 16000  # Assuming 16kHz
                result = self._filter_context_segments(result, context_duration)

            return result

        except Exception as e:
            logger.warning(f"ESTRATÉGIA 2: Chunk {chunk_index} processing failed: {e}")
            return {"segments": [], "text": ""}

    def _transcribe_chunk_sync(self, model, audio_data: np.ndarray, options: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous transcription for executor"""
        try:
            return model.transcribe(audio_data, **options)
        except Exception as e:
            logger.error(f"ESTRATÉGIA 2: Transcription failed: {e}")
            return {"segments": [], "text": ""}

    def _get_streaming_parameters(self, language: str) -> Dict[str, Any]:
        """Get optimized parameters for streaming chunks"""
        return {
            "language": language,
            "task": "transcribe",
            "fp16": _get_torch().cuda.is_available(),
            "verbose": False,
            "beam_size": 2,  # Better accuracy
            "best_of": 2,    # Better accuracy
            "temperature": 0.1,  # Slight randomness for better results
            "condition_on_previous_text": True,  # Better context preservation
            "compression_ratio_threshold": 2.4,
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,  # Lower threshold for better detection
            "word_timestamps": True,
            "without_timestamps": False,
        }

    def _filter_context_segments(self, result: Dict[str, Any], context_duration: float) -> Dict[str, Any]:
        """Filter out segments from context portion"""
        try:
            if not result or "segments" not in result:
                return result

            filtered_segments = []
            for segment in result["segments"]:
                segment_start = segment.get("start", 0)
                if (segment_start or 0) >= (context_duration or 0):
                    # Adjust timestamp to remove context offset
                    segment["start"] -= context_duration
                    segment["end"] -= context_duration
                    filtered_segments.append(segment)

            # Update text by removing context portion
            filtered_text = " ".join([seg.get("text", "") for seg in filtered_segments])

            return {
                "segments": filtered_segments,
                "text": filtered_text.strip(),
                "language": result.get("language", "")
            }

        except Exception as e:
            logger.warning(f"ESTRATÉGIA 2: Context filtering failed: {e}")
            return result

    def _deduplicate_streaming_segments(self, segments: List[Dict]) -> List[Dict]:
        """Remove overlapping segments from streaming processing"""
        try:
            if not segments:
                return segments

            # Sort by start time
            segments.sort(key=lambda x: x.get("start", 0))

            deduplicated = []
            last_end = -1

            for segment in segments:
                start = segment.get("start", 0)
                end = segment.get("end", 0)

                # Skip segments that significantly overlap with previous
                if start < last_end - (self.overlap_duration * 0.7):
                    continue

                deduplicated.append(segment)
                last_end = max(last_end, end)

            logger.info(f"ESTRATÉGIA 2: Segment deduplication: {len(segments)} -> {len(deduplicated)}")
            return deduplicated

        except Exception as e:
            logger.warning(f"ESTRATÉGIA 2: Segment deduplication failed: {e}")
            return segments

    def _clean_streaming_text(self, text: str) -> str:
        """Clean and deduplicate text from streaming processing"""
        try:
            if not text:
                return ""

            # Remove duplicate phrases and clean up
            sentences = []
            seen_phrases = set()

            for sentence in text.split('.'):
                sentence = sentence.strip()
                if sentence and sentence.lower() not in seen_phrases:
                    sentences.append(sentence)
                    seen_phrases.add(sentence.lower())

            return '. '.join(sentences).strip()

        except Exception as e:
            logger.warning(f"ESTRATÉGIA 2: Text cleaning failed: {e}")
            return text

# ==========================================
# LIBROSA-FREE MEL-SPECTROGRAM IMPLEMENTATION
# ==========================================
# Merged from librosa_free_melspec.py for file consolidation

def create_mel_filterbank(sr: int = 16000, n_fft: int = 400, n_mels: int = 80,
                         fmin: float = 0.0, fmax: float = 8000.0) -> np.ndarray:
    """
    Create mel-scale filter bank matrix (librosa-free implementation)
    """
    # Helper functions for mel scale conversion
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0**(mel / 2595.0) - 1.0)

    # Calculate number of frequency bins
    n_freqs = n_fft // 2 + 1

    # Create frequency bins
    freqs = np.linspace(0, sr / 2, n_freqs)

    # Convert to mel scale
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)

    # Create mel points
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    # Convert hz points to fft bin numbers
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    # Create filter bank
    filterbank = np.zeros((n_mels, n_freqs))

    for i in range(n_mels):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]

        # Left slope
        for j in range(left, center):
            if center > left:
                filterbank[i, j] = (j - left) / (center - left)

        # Right slope
        for j in range(center, right):
            if right > center:
                filterbank[i, j] = (right - j) / (right - center)

    return filterbank

def stft_magnitude(y: np.ndarray, n_fft: int = 400, hop_length: int = 160, window: str = 'hann') -> np.ndarray:
    """
    Compute Short-Time Fourier Transform magnitude (librosa-free)
    """
    # Create window function
    if window == 'hann':
        win = np.hanning(n_fft)
    else:
        win = np.ones(n_fft)

    # Pad signal
    pad_amount = n_fft // 2
    y_padded = np.pad(y, pad_amount, mode='constant')

    # Calculate number of frames
    n_frames = 1 + (len(y_padded) - n_fft) // hop_length

    # Initialize STFT matrix
    stft_matrix = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex64)

    # Compute STFT
    for i in range(n_frames):
        start = i * hop_length
        end = start + n_fft

        if end <= len(y_padded):
            frame = y_padded[start:end] * win
            fft_frame = np.fft.rfft(frame)
            stft_matrix[:, i] = fft_frame

    # Return magnitude
    return np.abs(stft_matrix)

def mel_spectrogram_librosa_free(y: np.ndarray, sr: int = 16000, n_fft: int = 400,
                                hop_length: int = 160, n_mels: int = 80,
                                fmin: float = 0.0, fmax: float = 8000.0) -> np.ndarray:
    """
    Compute mel-spectrogram without librosa dependency
    """
    try:
        # Compute magnitude spectrogram
        S = stft_magnitude(y, n_fft=n_fft, hop_length=hop_length)

        # Create mel filter bank
        mel_basis = create_mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)

        # Apply mel filters
        mel_spec = np.dot(mel_basis, S)

        # Ensure non-zero values for log
        mel_spec = np.maximum(mel_spec, 1e-10)

        logger.debug(f"✅ Librosa-free mel-spectrogram computed: {mel_spec.shape}")
        return mel_spec

    except Exception as e:
        logger.error(f"❌ Librosa-free mel-spectrogram failed: {e}")
        raise

def preprocess_audio_for_whisper(audio_array: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Complete audio preprocessing for OpenAI Whisper without librosa
    """
    try:
        # Ensure correct sample rate (should already be handled by RobustAudioLoader)
        if sr != 16000:
            # Simple resampling if needed
            target_length = int(len(audio_array) * 16000 / sr)
            audio_array = np.interp(
                np.linspace(0, len(audio_array), target_length),
                np.arange(len(audio_array)),
                audio_array
            ).astype(np.float32)

        # Normalize audio
        if np.max(np.abs(audio_array)) > 0:
            audio_array = audio_array / np.max(np.abs(audio_array))

        # Pad or trim audio to 30 seconds (480000 samples at 16kHz)
        max_length = 480000
        if len(audio_array) > max_length:
            audio_array = audio_array[:max_length]
        else:
            padding = max_length - len(audio_array)
            audio_array = np.pad(audio_array, (0, padding), mode='constant')

        # For OpenAI Whisper, return the raw audio array
        # Whisper will handle mel-spectrogram conversion internally
        logger.debug(f"Audio preprocessing successful: {audio_array.shape}")
        return audio_array.astype(np.float32)

    except Exception as e:
        logger.error(f"❌ Librosa-free preprocessing failed: {e}")
        raise RuntimeError(f"Audio preprocessing failed: {e}")

# ==========================================
# ROBUST AUDIO LOADER IMPLEMENTATION
# ==========================================
# Merged from robust_audio_loader.py for file consolidation

class RobustAudioLoader:
    """
    Robust audio loading with fallback strategies for Windows compatibility
    Handles various audio formats and loading libraries gracefully
    """

    def __init__(self):
        self.fallback_strategies = [
            self._load_with_soundfile,
            self._load_with_librosa_safe
        ]

    def load_audio(self, audio_path: str, target_sr: int = 16000, duration: Union[float, None] = None) -> Tuple[np.ndarray, int]:
        """
        Load audio with multiple fallback strategies and comprehensive validation
        """
        # Pre-processing validation
        logger.info(f"Loading audio: {audio_path}")

        # Validate input parameters
        if not audio_path:
            raise ValueError("Audio path cannot be empty")

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Check file size
        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            raise ValueError(f"Audio file is empty: {audio_path}")

        if file_size < 100:  # Less than 100 bytes
            raise ValueError(f"Audio file too small ({file_size} bytes): {audio_path}")

        logger.info(f"✓ Pre-validation passed: {file_size} bytes")

        # Try each loading strategy
        last_exception = None
        for i, strategy in enumerate(self.fallback_strategies):
            try:
                logger.info(f"Trying audio loading strategy {i+1}/{len(self.fallback_strategies)}")
                data, sr = strategy(audio_path, target_sr, duration)

                # Comprehensive validation of loaded data
                if data is None:
                    raise ValueError("Strategy returned None data")

                if not isinstance(data, np.ndarray):
                    raise ValueError(f"Strategy returned invalid data type: {type(data)}")

                if len(data) == 0:
                    raise ValueError("Strategy returned empty audio data")

                # Check for silent audio
                max_amplitude = np.max(np.abs(data))
                if max_amplitude < 1e-6:
                    logger.warning(f"Audio appears to be very quiet (max amplitude: {max_amplitude:.8f})")

                # Check sample rate
                if sr <= 0:
                    raise ValueError(f"Invalid sample rate: {sr}")

                # Calculate duration
                duration_seconds = len(data) / sr
                logger.info(f"✅ Audio loading successful with strategy {i+1}:")
                logger.info(f"  Shape: {data.shape}")
                logger.info(f"  Sample rate: {sr} Hz")
                logger.info(f"  Duration: {duration_seconds:.2f} seconds")
                logger.info(f"  Max amplitude: {max_amplitude:.6f}")

                return data, sr

            except Exception as e:
                last_exception = e
                logger.error(f"Audio loading strategy {i+1} failed: {e}")
                continue

        # If all strategies fail, raise comprehensive error
        logger.error(f"CRITICAL: All audio loading strategies failed for: {audio_path}")
        logger.error(f"File size: {file_size} bytes")
        logger.error(f"Last exception: {last_exception}")
        raise RuntimeError(f"All audio loading strategies failed for: {audio_path}. Last error: {last_exception}")

    def _load_with_soundfile(self, audio_path: str, target_sr: int, duration: Union[float, None]) -> Tuple[np.ndarray, int]:
        """
        Primary loading strategy using soundfile (most reliable)
        """
        try:
            # Load with soundfile (handles most formats well)
            data, sr = _get_soundfile().read(audio_path, dtype='float32')

            # Convert to mono if stereo
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)

            # Handle duration limiting
            if duration is not None:
                max_samples = int(duration * sr)
                if len(data) > max_samples:
                    data = data[:max_samples]

            # Resample if needed
            if sr != target_sr:
                data = self._simple_resample(data, sr, target_sr)
                sr = target_sr

            logger.debug(f"✅ Soundfile loading success: shape={data.shape}, sr={sr}")
            return data, int(sr)

        except Exception as e:
            logger.debug(f"Soundfile loading failed: {e}")
            raise

    def _load_with_librosa_safe(self, audio_path: str, target_sr: int, duration: Union[float, None]) -> Tuple[np.ndarray, int]:
        """
        Fallback loading strategy using librosa with conservative settings
        """
        try:
            # First attempt with librosa
            try:
                data, sr = _get_librosa().load(
                    audio_path,
                    sr=target_sr,
                    duration=duration,
                    mono=True,
                    res_type='kaiser_fast'  # Resampling mais rápido e estável
                )

                logger.debug(f"✅ Librosa loading success: shape={data.shape}, sr={sr}")
                return data, int(sr)

            except Exception as e:
                # Se falhar, tentar com configurações ainda mais conservadoras
                logger.warning(f"Librosa first attempt failed: {e}, trying conservative mode")

                data, sr = _get_librosa().load(
                    audio_path,
                    sr=None,  # Manter sample rate original
                    duration=duration,
                    mono=True
                )

                # Resample manually se necessário
                if sr != target_sr:
                    data = self._simple_resample(data, int(sr), target_sr)
                    sr = target_sr

                logger.debug(f"✅ Librosa conservative loading success: shape={data.shape}, sr={sr}")
                return data, int(sr)

        except Exception as e:
            logger.error(f"All audio loading attempts failed: {e}")
            raise

    def _simple_resample(self, data: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
        """Resampling simples usando interpolação linear"""
        if original_sr == target_sr:
            return data

        # Calculate new length
        new_length = int(len(data) * target_sr / original_sr)

        # Create indices for interpolation
        old_indices = np.linspace(0, len(data) - 1, len(data))
        new_indices = np.linspace(0, len(data) - 1, new_length)

        # Interpolate
        resampled = np.interp(new_indices, old_indices, data)

        return resampled.astype(np.float32)

# Module-level helper function
def load_audio_robust(audio_path: str, target_sr: int = 16000, duration: Union[float, None] = None) -> Tuple[np.ndarray, int]:
    """
    Module-level function for loading audio robustly
    """
    loader = RobustAudioLoader()
    return loader.load_audio(audio_path, target_sr, duration)

# ==========================================
# CONSOLIDATED AUDIO CAPTURE AND RECORDING
# ==========================================
# Consolidated from audio_capture_process.py, audio_recorder.py, and robust_audio_loader.py

import wave
import threading
import queue
import sounddevice as sd
from pathlib import Path

# Try to import pyaudio for fallback recording
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    logger.warning("PyAudio not available - some recording features disabled")

class AudioCaptureProcess:
    """Consolidated real-time audio capture process with level monitoring"""

    def __init__(self, process_id: int = 0, queue_manager=None, shared_memory=None):
        self.process_id = process_id
        self.queue_manager = queue_manager
        self.shared_memory = shared_memory

        # Audio configuration
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 1024
        self.dtype = np.float32

        # Recording state
        self.is_recording = False
        self.is_paused = False
        self.stream = None
        self.frames = []
        self.output_file = None
        self.recording_start_time = None

    def start(self):
        """Start the audio capture process"""
        try:
            logger.info(f"Starting audio capture process {self.process_id}")
            self.start_recording()
        except Exception as e:
            logger.error(f"Error starting audio capture process: {e}")

        # Buffers and control
        self.audio_buffer = queue.Queue(maxsize=100)
        self.control_thread = None
        self.running = False

        # Audio level monitoring (from audio_recorder.py)
        self.audio_level_callback = None
        self.max_recording_seconds = 300  # 5 minutes max for safety

        # Format configuration
        self.format_type = "wav"  # wav or mp4
        self.conversion_needed = False

        logger.info("AudioCaptureProcess initialized")

    def set_audio_level_callback(self, callback):
        """Set callback for real-time audio level monitoring"""
        self.audio_level_callback = callback

    def _calculate_audio_level(self, audio_data: np.ndarray) -> float:
        """Calculate audio level (0-100) from audio data"""
        try:
            if len(audio_data) == 0:
                return 0.0

            # Calculate RMS (root mean square) for audio level
            rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
            # Normalize to 0-100 scale
            level = min(100.0, (rms * 1000.0) * 100)
            return level
        except Exception as e:
            logger.error(f"Audio level calculation error: {e}")
            return 0.0

    def start_recording(self, config: Optional[dict] = None):
        """Start audio recording with configuration"""
        try:
            if config is None:
                config = {}
            if self.is_recording:
                logger.warning("Recording already active")
                return False

            config = config or {}
            self.format_type = config.get("format", "wav")
            self._setup_output_file()

            # Clear previous frames
            self.frames = []

            # Audio callback with level monitoring
            def audio_callback(indata, frames, time, status):
                if status is not None and status:
                    logger.warning(f"Audio stream status: {status}")

                if self.is_recording and not self.is_paused:
                    if indata is not None and len(indata) > 0:
                        # Check for valid audio (not just silence)
                        if np.any(np.abs(indata) > 0.001):
                            self.frames.append(indata.copy())

                            # Calculate and report audio level
                            if self.audio_level_callback:
                                try:
                                    level = self._calculate_audio_level(indata.flatten())
                                    self.audio_level_callback(level)
                                except Exception as e:
                                    logger.error(f"Audio level callback error: {e}")

                            # Add to shared buffer if available
                            if self.queue_manager and self.shared_memory:
                                try:
                                    audio_data = {
                                        "timestamp": time.inputBufferAdcTime if hasattr(time, 'inputBufferAdcTime') else time,
                                        "data": indata.copy(),
                                        "sample_rate": self.sample_rate,
                                        "channels": self.channels
                                    }
                                    self.audio_buffer.put_nowait(audio_data)
                                except queue.Full:
                                    # Buffer full, discard oldest frame
                                    try:
                                        self.audio_buffer.get_nowait()
                                        self.audio_buffer.put_nowait(audio_data)
                                    except queue.Empty:
                                        pass
                        else:
                            # Still add silent frames to maintain timing
                            self.frames.append(indata.copy())

            # Start audio stream
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=audio_callback,
                dtype=self.dtype,
                blocksize=self.chunk_size,
                latency='low'
            )

            self.stream.start()
            self.is_recording = True
            self.recording_start_time = time.time()

            logger.info(f"Recording started: {self.output_file}")
            return True

        except Exception as e:
            logger.error(f"Error starting recording: {e}")
            return False

    def stop_recording(self):
        """Stop audio recording"""
        try:
            if not self.is_recording:
                logger.warning("No recording active")
                return None

            self.is_recording = False

            # Stop stream
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None

            # Save audio file
            duration = time.time() - (self.recording_start_time or time.time())
            output_path = self._save_audio_file()

            logger.info(f"Recording stopped after {duration:.2f} seconds")
            return output_path

        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
            return None

    def pause_recording(self):
        """Pause recording"""
        if self.is_recording and not self.is_paused:
            self.is_paused = True
            logger.info("Recording paused")
            return True
        return False

    def resume_recording(self):
        """Resume recording"""
        if self.is_recording and self.is_paused:
            self.is_paused = False
            logger.info("Recording resumed")
            return True
        return False

    def _setup_output_file(self):
        """Configure output file"""
        try:
            # Use FileManager if available
            try:
                from src.file_manager import FileManager
                recordings_dir = FileManager.get_data_path("recordings")
                FileManager.ensure_directory_exists(recordings_dir)
            except ImportError:
                # Fallback to simple directory creation
                recordings_dir = Path("data/recordings")
                recordings_dir.mkdir(parents=True, exist_ok=True)

            timestamp = int(time.time())
            extension = "wav" if self.format_type == "wav" else "mp4"
            filename = f"recording_{timestamp}.{extension}"

            self.output_file = Path(recordings_dir) / filename
            self.conversion_needed = (self.format_type == "mp4")

        except Exception as e:
            logger.error(f"Error setting up output file: {e}")
            # Fallback to current directory
            self.output_file = Path(f"recording_{int(time.time())}.wav")

    def _save_audio_file(self):
        """Save audio file"""
        try:
            if not self.frames:
                logger.warning("No audio frames to save")
                return None

            # Concatenate all frames
            audio_data = np.concatenate(self.frames)

            # Normalize audio
            if len(audio_data) > 0:
                max_val = np.max(np.abs(audio_data))
                if max_val > 0:
                    audio_data = audio_data / max_val * 0.8  # Normalize to 80%

            # Save as WAV
            if self.output_file is not None:
                temp_wav = self.output_file.with_suffix('.wav')

                # Use soundfile for saving
                sf = _get_soundfile()
                sf.write(str(temp_wav), audio_data, self.sample_rate)

                # Convert to MP4 if needed
                if self.conversion_needed and self.format_type == "mp4":
                    try:
                        self._convert_to_mp4(temp_wav, self.output_file)
                        # Remove temporary WAV
                        if temp_wav != self.output_file:
                            temp_wav.unlink(missing_ok=True)
                    except Exception as e:
                        logger.warning(f"MP4 conversion failed: {e}, keeping WAV")
                        self.output_file = temp_wav
                else:
                    self.output_file = temp_wav

                logger.info(f"Audio file saved: {self.output_file}")
                return str(self.output_file)

        except Exception as e:
            logger.error(f"Error saving audio file: {e}")
            return None

    def _convert_to_mp4(self, wav_file: Path, mp4_file: Path):
        """Convert WAV to MP4 using FFmpeg"""
        try:
            import subprocess

            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-i", str(wav_file),
                "-f", "mp4",
                "-c:a", "aac",
                "-b:a", "192k",
                "-movflags", "frag_keyframe+empty_moov",
                str(mp4_file)
            ]

            subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                timeout=60,
                check=True
            )

            logger.info(f"WAV->MP4 conversion completed: {mp4_file}")

        except Exception as e:
            logger.error(f"Error in WAV->MP4 conversion: {e}")
            raise

    def cleanup(self):
        """Clean up resources"""
        try:
            if self.is_recording:
                self.stop_recording()

            if self.stream:
                try:
                    self.stream.stop()
                    self.stream.close()
                except Exception as e:
                    logger.warning(f"Error closing stream: {e}")
                finally:
                    self.stream = None

            self.frames = []
            logger.info("AudioCaptureProcess cleanup completed")

        except Exception as e:
            logger.error(f"Error in cleanup: {e}")

class AudioRecorder:
    """Simplified audio recorder using PyAudio as fallback"""

    def __init__(self):
        self.recording = False
        self.paused = False
        self.audio_data = []
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 1024
        self.format = pyaudio.paInt16 if PYAUDIO_AVAILABLE else 1

        # Audio stream components
        self.audio_stream = None
        self.pyaudio_instance = None
        self.recording_thread = None

        # Safety and monitoring
        self.max_recording_seconds = 300
        self.audio_level_callback = None
        self.start_time = None
        self.total_duration = 0.0

        logger.info("AudioRecorder initialized")

    def set_audio_level_callback(self, callback):
        """Set callback for real-time audio level monitoring"""
        self.audio_level_callback = callback

    def _calculate_audio_level(self, audio_chunk: bytes) -> float:
        """Calculate audio level from bytes"""
        try:
            audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
            rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
            level = min(100.0, (rms / 1000.0) * 100)
            return level
        except Exception as e:
            logger.error(f"Audio level calculation error: {e}")
            return 0.0

    async def start_recording(self, session_id: str) -> bool:
        """Start PyAudio recording"""
        try:
            if not PYAUDIO_AVAILABLE:
                logger.error("Cannot start recording: PyAudio not available")
                return False

            if self.recording:
                return False

            self.recording = True
            self.paused = False
            self.audio_data = []
            self.start_time = time.time()

            self.recording_thread = threading.Thread(
                target=self._recording_worker,
                daemon=True
            )
            self.recording_thread.start()

            return True

        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            self.recording = False
            return False

    def _recording_worker(self):
        """PyAudio recording worker thread"""
        try:
            if not PYAUDIO_AVAILABLE:
                return

            self.pyaudio_instance = pyaudio.PyAudio()
            self.audio_stream = self.pyaudio_instance.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )

            while self.recording:
                if not self.paused:
                    try:
                        audio_chunk = self.audio_stream.read(
                            self.chunk_size,
                            exception_on_overflow=False
                        )
                        self.audio_data.append(audio_chunk)

                        # Audio level callback
                        if self.audio_level_callback:
                            level = self._calculate_audio_level(audio_chunk)
                            try:
                                self.audio_level_callback(level)
                            except Exception as e:
                                logger.error(f"Audio level callback error: {e}")

                        # Safety check
                        if self.start_time and (time.time() - self.start_time) > self.max_recording_seconds:
                            logger.warning("Recording stopped: time limit reached")
                            self.recording = False
                            break

                    except Exception as e:
                        logger.error(f"Recording chunk error: {e}")
                        break
                else:
                    time.sleep(0.1)

        except Exception as e:
            logger.error(f"Recording worker error: {e}")
        finally:
            try:
                if self.audio_stream:
                    self.audio_stream.stop_stream()
                    self.audio_stream.close()
                if self.pyaudio_instance:
                    self.pyaudio_instance.terminate()
            except Exception as e:
                logger.error(f"Audio cleanup error: {e}")

    async def stop_recording(self, session_id: str, output_dir: str = "data/recordings"):
        """Stop recording and save file"""
        try:
            if not self.recording:
                return None

            self.recording = False
            self.paused = False

            if self.recording_thread and self.recording_thread.is_alive():
                self.recording_thread.join(timeout=5.0)

            if self.start_time:
                self.total_duration = time.time() - self.start_time

            if self.audio_data:
                return await self._save_audio_file(session_id, output_dir)

            return None

        except Exception as e:
            logger.error(f"Failed to stop recording: {e}")
            return None

    async def _save_audio_file(self, session_id: str, output_dir: str) -> str:
        """Save audio to WAV file"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            timestamp = int(time.time())
            filename = f"recording_{session_id}_{timestamp}.wav"
            file_path = output_path / filename

            with wave.open(str(file_path), 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 16-bit audio
                wav_file.setframerate(self.sample_rate)
                audio_bytes = b''.join(self.audio_data)
                wav_file.writeframes(audio_bytes)

            return str(file_path)

        except Exception as e:
            logger.error(f"Failed to save audio file: {e}")
            raise

# Enhanced robust audio preprocessing (consolidated from robust_audio_loader.py)
def preprocess_audio_for_whisper(audio_data: Union[str, np.ndarray], target_sr: int = 16000) -> np.ndarray:
    """
    Robust audio preprocessing for Whisper models
    Enhanced version consolidated from robust_audio_loader.py
    """
    try:
        if isinstance(audio_data, str):
            # Load from file using existing robust loader
            loader = RobustAudioLoader()
            audio_array, sr = loader.load_audio(audio_data, target_sr)
        elif isinstance(audio_data, np.ndarray):
            # Already loaded audio
            audio_array = audio_data
            sr = target_sr  # Assume it's already at target sample rate
        else:
            raise ValueError(f"Unsupported audio data type: {type(audio_data)}")

        # Convert to mono if stereo
        if audio_array.ndim > 1:
            audio_array = np.mean(audio_array, axis=1)

        # Ensure float32 dtype
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)

        # Normalize audio
        if np.max(np.abs(audio_array)) > 0:
            audio_array = audio_array / np.max(np.abs(audio_array))

        return audio_array

    except Exception as e:
        logger.error(f"Error in robust audio preprocessing: {e}")
        # Return silence as fallback
        return np.zeros(target_sr, dtype=np.float32)  # 1 second of silence

# Compatibility aliases for backward compatibility
def audio_capture_worker(process_id: int):
    """Simplified worker function for audio capture process"""
    try:
        logger.info(f"Audio capture worker {process_id} iniciado")
        # Implementação simplificada para evitar pickle errors
        for i in range(10):
            logger.debug(f"Audio capture worker {process_id} - ciclo {i+1}")
            time.sleep(0.1)
        logger.info(f"Audio capture worker {process_id} finalizado")
    except Exception as e:
        logger.error(f"Fatal error in audio capture process: {e}")
        raise

# Global instances
streaming_processor = StreamingAudioProcessor()
audio_recorder = AudioRecorder()  # Global audio recorder instance
robust_audio_loader = RobustAudioLoader()