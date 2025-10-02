"""
TranscrevAI Optimized - Transcription Engine
Sistema de transcrição PT-BR otimizado com integração ao model cache e resource manager
"""

import asyncio
import gc
import os
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
import warnings

# Import our optimized modules
from logging_setup import get_logger, log_performance
from resource_manager import get_resource_manager, ResourceStatus
from model_cache import get_model_cache, load_whisper_model
from config import CONFIG, PT_BR_CONFIG

logger = get_logger("transcrevai.transcription")

# Suppress warnings from ML libraries
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Lazy imports for heavy dependencies
_whisper = None
_librosa = None

def get_whisper():
    """Lazy import whisper"""
    global _whisper
    if _whisper is None:
        try:
            import whisper
            _whisper = whisper
            logger.info("OpenAI Whisper loaded successfully")
        except ImportError as e:
            logger.error(f"Failed to import whisper: {e}")
            raise ImportError(f"OpenAI Whisper not available: {e}")
    return _whisper

def get_librosa():
    """Lazy import librosa"""
    global _librosa
    if _librosa is None:
        try:
            import librosa
            _librosa = librosa
            logger.info("librosa loaded successfully")
        except ImportError as e:
            logger.warning(f"librosa not available: {e}")
            _librosa = None
    return _librosa


class TranscriptionError(Exception):
    """Custom exception for transcription errors"""
    def __init__(self, message: str, error_type: str = "unknown"):
        self.error_type = error_type
        super().__init__(f"[{error_type}] {message}")


class PTBRTranscriptionOptimizer:
    """
    PT-BR specific optimizations for transcription
    Applies contextual corrections and improvements specific to Brazilian Portuguese
    """
    
    def __init__(self):
        self.corrections = PT_BR_CONFIG["corrections"]
        self.common_replacements = self._build_replacement_dict()
        
    def _build_replacement_dict(self) -> Dict[str, str]:
        """Build comprehensive replacement dictionary for PT-BR"""
        replacements = self.corrections.copy()
        
        # Add common transcription fixes
        replacements.update({
            # Contractions and informal speech
            "pra": "para",
            "pro": "para o",
            "pras": "para as",
            "pros": "para os",
            "numa": "em uma", 
            "numas": "em umas",
            "nuns": "em uns",
            "dum": "de um",
            "duma": "de uma",
            
            # Common words often mistranscribed
            "ta bom": "tá bom",
            "ta certo": "tá certo", 
            "voce": "você",
            "nao": "não",
            "tambem": "também",
            "so": "só",
            "ja": "já",
            "la": "lá",
            
            # Question words
            "onde que": "onde",
            "como que": "como",
            "quando que": "quando",
            "porque que": "por que",
            
            # Common expressions
            "ai meu deus": "ai, meu Deus",
            "nossa senhora": "nossa Senhora",
            "valeu cara": "valeu, cara",
            "beleza entao": "beleza, então",
        })
        
        return replacements
    
    def optimize_text(self, text: str) -> str:
        """
        Apply PT-BR specific optimizations to transcribed text
        
        Args:
            text: Original transcribed text
            
        Returns:
            str: Optimized text with PT-BR corrections
        """
        if not text or not isinstance(text, str):
            return text
        
        optimized_text = text.lower().strip()
        
        # Apply word-level replacements
        for wrong, correct in self.common_replacements.items():
            optimized_text = optimized_text.replace(wrong, correct)
        
        # Apply contextual corrections
        optimized_text = self._apply_contextual_corrections(optimized_text)
        
        # Apply capitalization rules
        optimized_text = self._apply_capitalization_rules(optimized_text)
        
        return optimized_text
    
    def _apply_contextual_corrections(self, text: str) -> str:
        """Apply context-aware corrections"""
        # Fix common patterns
        patterns = [
            # Question patterns
            (r'\bonde esta\b', 'onde está'),
            (r'\bcomo esta\b', 'como está'),
            (r'\bquem e\b', 'quem é'),
            (r'\bo que e\b', 'o que é'),
            
            # Contractions with pronouns
            (r'\bvou te\b', 'vou te'),
            (r'\bvamos nos\b', 'vamos nos'),
            
            # Common phrases
            (r'\ba gente vai\b', 'a gente vai'),
            (r'\btudo bem\b', 'tudo bem'),
            (r'\btudo bom\b', 'tudo bom'),
        ]
        
        import re
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def _apply_capitalization_rules(self, text: str) -> str:
        """Apply proper capitalization for PT-BR"""
        if not text:
            return text
        
        # Capitalize first letter of sentences
        sentences = text.split('. ')
        capitalized_sentences = []
        
        for sentence in sentences:
            if sentence:
                # Capitalize first letter
                sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
                capitalized_sentences.append(sentence)
        
        return '. '.join(capitalized_sentences)


class BrowserSafeTranscriber:
    """
    Browser-safe transcription with progressive processing and memory management
    """
    
    def __init__(self):
        self.resource_manager = get_resource_manager()
        self.model_cache = get_model_cache()
        self.optimizer = PTBRTranscriptionOptimizer()
        
        # Processing settings
        self.chunk_duration = 30.0  # Process in 30-second chunks
        self.overlap_duration = 1.0  # 1-second overlap for continuity
        
        logger.info("BrowserSafeTranscriber initialized")
    
    async def transcribe_audio(self, 
                             audio_file: str, 
                             progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Transcribe audio file with browser-safe progressive processing
        
        Args:
            audio_file: Path to audio file
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dict containing transcription results and metadata
        """
        transcribe_start = time.time()
        
        try:
            # Validate input file
            if not os.path.exists(audio_file):
                raise TranscriptionError(f"Audio file not found: {audio_file}", "file_not_found")
            
            # Load audio and get duration
            audio_data, duration = await self._load_audio_safe(audio_file)
            if audio_data is None:
                raise TranscriptionError("Failed to load audio file", "audio_load_error")
            
            # Update progress
            if progress_callback:
                await progress_callback(5, "Áudio carregado, iniciando transcrição...")
            
            # Load Whisper model
            model = await self._ensure_model_loaded()
            if model is None:
                raise TranscriptionError("Failed to load Whisper model", "model_load_error")
            
            # Update progress
            if progress_callback:
                await progress_callback(15, "Modelo carregado, processando áudio...")
            
            # Process audio in chunks for browser safety
            segments = await self._process_audio_chunks(
                audio_data, 
                model, 
                duration,
                progress_callback
            )
            
            # Apply PT-BR optimizations
            if progress_callback:
                await progress_callback(90, "Aplicando otimizações PT-BR...")
            
            optimized_segments = self._optimize_transcription_results(segments)
            
            # Calculate metrics
            processing_time = time.time() - transcribe_start
            processing_ratio = processing_time / max(duration, 1.0)
            
            # Log performance
            log_performance(
                "Transcription completed",
                duration=processing_time,
                audio_duration=duration,
                processing_ratio=processing_ratio,
                segments_count=len(optimized_segments),
                model="medium",
                language="pt"
            )
            
            if progress_callback:
                await progress_callback(100, "Transcrição concluída!")
            
            return {
                "success": True,
                "segments": optimized_segments,
                "metadata": {
                    "audio_duration": duration,
                    "processing_time": processing_time,
                    "processing_ratio": processing_ratio,
                    "model": "medium",
                    "language": "pt",
                    "segments_count": len(optimized_segments)
                }
            }
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            if progress_callback:
                await progress_callback(0, f"Erro na transcrição: {str(e)}")
            
            return {
                "success": False,
                "error": str(e),
                "error_type": getattr(e, 'error_type', 'unknown')
            }
    
    async def _load_audio_safe(self, audio_file: str) -> tuple[Optional[np.ndarray], float]:
        """Safely load audio file with memory management"""
        try:
            # Check memory before loading
            file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
            if not self.resource_manager.can_allocate(file_size_mb * 2):
                logger.warning("Insufficient memory for audio loading, attempting cleanup")
                await self.resource_manager.perform_cleanup(aggressive=False)
            
            # Load audio using librosa (preferred) or fallback
            librosa = get_librosa()
            if librosa:
                audio_data, sample_rate = librosa.load(
                    audio_file,
                    sr=CONFIG["audio"]["sample_rate"],
                    mono=True
                )
                duration = len(audio_data) / sample_rate
            else:
                # Fallback to soundfile
                import soundfile as sf
                audio_data, sample_rate = sf.read(audio_file)
                
                # Convert to mono if stereo
                if audio_data.ndim > 1:
                    audio_data = np.mean(audio_data, axis=1)
                
                # Resample if needed (simplified)
                target_sr = CONFIG["audio"]["sample_rate"]
                if sample_rate != target_sr:
                    # Simple resampling (for production, use proper resampling)
                    audio_data = audio_data[::int(sample_rate / target_sr)]
                
                duration = len(audio_data) / target_sr
            
            logger.info(f"Audio loaded: {duration:.2f}s, {sample_rate}Hz")
            return audio_data.astype(np.float32), duration
            
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            return None, 0.0
    
    async def _ensure_model_loaded(self):
        """Ensure Whisper model is loaded and cached"""
        try:
            # Use our model cache system
            model = await load_whisper_model("medium")
            logger.info("Whisper model ready for transcription")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            return None
    
    async def _process_audio_chunks(self, 
                                  audio_data: np.ndarray, 
                                  model: Any,
                                  duration: float,
                                  progress_callback: Optional[Callable] = None) -> List[Dict]:
        """
        Process audio in chunks to prevent browser freezing
        """
        segments = []
        sample_rate = CONFIG["audio"]["sample_rate"]
        chunk_samples = int(self.chunk_duration * sample_rate)
        overlap_samples = int(self.overlap_duration * sample_rate)
        
        total_samples = len(audio_data)
        processed_samples = 0
        chunk_index = 0
        
        while processed_samples < total_samples:
            # Calculate chunk boundaries
            start_sample = max(0, processed_samples - overlap_samples)
            end_sample = min(total_samples, processed_samples + chunk_samples)
            
            # Extract chunk
            chunk = audio_data[start_sample:end_sample]
            chunk_start_time = start_sample / sample_rate
            
            # Check memory pressure before processing
            if self.resource_manager.is_memory_pressure_high():
                logger.warning("High memory pressure during transcription")
                await self.resource_manager.perform_cleanup(aggressive=False)
                await asyncio.sleep(0.1)  # Brief pause to let system recover
            
            try:
                # Process chunk
                chunk_segments = await self._transcribe_chunk(
                    chunk, 
                    model, 
                    chunk_start_time,
                    chunk_index
                )
                
                # Add segments with proper timing adjustment
                segments.extend(chunk_segments)
                
                # Update progress
                progress = 20 + int((processed_samples / total_samples) * 65)  # 20-85%
                if progress_callback:
                    chunk_duration = (end_sample - start_sample) / sample_rate
                    await progress_callback(
                        progress, 
                        f"Processando áudio... {chunk_duration:.1f}s de {duration:.1f}s"
                    )
                
                # Browser-safe: yield control periodically
                if chunk_index % 3 == 0:
                    await asyncio.sleep(0.01)  # Small yield to prevent blocking
                
            except Exception as e:
                logger.error(f"Failed to process chunk {chunk_index}: {e}")
                # Continue with next chunk instead of failing completely
            
            processed_samples = end_sample
            chunk_index += 1
        
        return segments
    
    async def _transcribe_chunk(self, 
                              chunk: np.ndarray, 
                              model: Any, 
                              offset_time: float,
                              chunk_index: int) -> List[Dict]:
        """Transcribe a single audio chunk"""
        try:
            # Prepare whisper config for PT-BR
            whisper_config = CONFIG["whisper"].copy()
            
            # Run transcription in executor to prevent blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: model.transcribe(
                    chunk,
                    **whisper_config
                )
            )
            
            # Process segments
            segments = []
            if "segments" in result:
                for segment in result["segments"]:
                    segments.append({
                        "start": segment["start"] + offset_time,
                        "end": segment["end"] + offset_time,
                        "text": segment["text"].strip(),
                        "confidence": segment.get("no_speech_prob", 0.0),
                        "chunk_index": chunk_index
                    })
            else:
                # Fallback: create single segment from full result
                segments.append({
                    "start": offset_time,
                    "end": offset_time + len(chunk) / CONFIG["audio"]["sample_rate"],
                    "text": result.get("text", "").strip(),
                    "confidence": 0.5,  # Default confidence
                    "chunk_index": chunk_index
                })
            
            return segments
            
        except Exception as e:
            logger.error(f"Chunk transcription failed: {e}")
            return []
    
    def _optimize_transcription_results(self, segments: List[Dict]) -> List[Dict]:
        """Apply PT-BR optimizations to transcription results"""
        optimized_segments = []
        
        for segment in segments:
            # Apply text optimization
            original_text = segment.get("text", "")
            if original_text:
                optimized_text = self.optimizer.optimize_text(original_text)
                
                # Create optimized segment
                optimized_segment = segment.copy()
                optimized_segment["text"] = optimized_text
                optimized_segment["original_text"] = original_text
                
                # Only add if text is meaningful
                if len(optimized_text.strip()) > 2:
                    optimized_segments.append(optimized_segment)
        
        # Merge consecutive segments from same chunk if needed
        merged_segments = self._merge_consecutive_segments(optimized_segments)
        
        return merged_segments
    
    def _merge_consecutive_segments(self, segments: List[Dict]) -> List[Dict]:
        """Merge consecutive segments to reduce fragmentation"""
        if not segments:
            return segments
        
        merged = []
        current_segment = None
        
        for segment in segments:
            if current_segment is None:
                current_segment = segment.copy()
            else:
                # Check if segments should be merged (same chunk, close timing)
                time_gap = segment["start"] - current_segment["end"]
                same_chunk = segment.get("chunk_index") == current_segment.get("chunk_index")
                
                if same_chunk and time_gap < 2.0:  # Less than 2 seconds gap
                    # Merge segments
                    current_segment["end"] = segment["end"]
                    current_segment["text"] += " " + segment["text"]
                    current_segment["confidence"] = min(
                        current_segment["confidence"], 
                        segment["confidence"]
                    )
                else:
                    # Start new segment
                    merged.append(current_segment)
                    current_segment = segment.copy()
        
        # Add final segment
        if current_segment:
            merged.append(current_segment)
        
        return merged


class TranscriptionEngine:
    """
    Main transcription engine with browser-safe processing and PT-BR optimization
    """
    
    def __init__(self):
        self.transcriber = BrowserSafeTranscriber()
        self.resource_manager = get_resource_manager()
        
        # Statistics
        self.total_transcriptions = 0
        self.total_processing_time = 0.0
        self.total_audio_duration = 0.0
        
        logger.info("TranscriptionEngine initialized")
    
    async def transcribe(self, 
                        audio_file: str, 
                        language: str = "pt",
                        progress_callback: Optional[Callable] = None) -> List[Dict]:
        """
        Main transcription interface
        
        Args:
            audio_file: Path to audio file
            language: Language code (fixed to 'pt' for PT-BR)
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of transcription segments
        """
        if language != "pt":
            logger.warning(f"Language {language} not supported, using PT-BR")
        
        transcription_start = time.time()
        
        try:
            # Reserve memory for transcription
            file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
            if not self.resource_manager.reserve_memory(
                f"transcription_{int(time.time())}", 
                file_size_mb * 1.5, 
                "transcription_process"
            ):
                logger.warning("Could not reserve memory for transcription")
            
            # Perform transcription
            result = await self.transcriber.transcribe_audio(audio_file, progress_callback)
            
            if result["success"]:
                segments = result["segments"]
                metadata = result["metadata"]
                
                # Update statistics
                self.total_transcriptions += 1
                self.total_processing_time += metadata["processing_time"]
                self.total_audio_duration += metadata["audio_duration"]
                
                # Log success
                logger.info(f"Transcription successful: {len(segments)} segments, "
                           f"{metadata['processing_ratio']:.2f}x processing ratio")
                
                return segments
            else:
                error_msg = result.get("error", "Unknown transcription error")
                logger.error(f"Transcription failed: {error_msg}")
                raise TranscriptionError(error_msg, result.get("error_type", "unknown"))
            
        except Exception as e:
            logger.error(f"TranscriptionEngine error: {e}")
            if progress_callback:
                await progress_callback(0, f"Erro: {str(e)}")
            raise
        
        finally:
            # Release memory reservation
            self.resource_manager.release_memory_reservation(f"transcription_{int(time.time())}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get transcription statistics"""
        avg_processing_ratio = (
            self.total_processing_time / max(self.total_audio_duration, 1.0)
            if self.total_audio_duration > 0 else 0.0
        )
        
        return {
            "total_transcriptions": self.total_transcriptions,
            "total_processing_time": self.total_processing_time,
            "total_audio_duration": self.total_audio_duration,
            "average_processing_ratio": avg_processing_ratio,
            "model": "medium",
            "language": "pt",
            "optimizations": "PT-BR contextual corrections enabled"
        }
    
    async def validate_audio_for_transcription(self, audio_file: str) -> Dict[str, Any]:
        """
        Validate audio file for transcription
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Dict with validation results
        """
        try:
            if not os.path.exists(audio_file):
                return {"valid": False, "error": "File not found"}
            
            # Check file size
            file_size = os.path.getsize(audio_file)
            max_size = CONFIG["audio"]["max_file_size_mb"] * 1024 * 1024
            if file_size > max_size:
                return {"valid": False, "error": f"File too large (max {max_size//1024//1024}MB)"}
            
            # Check format
            file_ext = Path(audio_file).suffix.lower()
            if file_ext not in CONFIG["audio"]["supported_formats"]:
                return {"valid": False, "error": f"Unsupported format: {file_ext}"}
            
            # Try to get duration
            try:
                import soundfile as sf
                info = sf.info(audio_file)
                duration = info.frames / info.samplerate
                
                if duration > CONFIG["audio"]["max_duration_minutes"] * 60:
                    return {"valid": False, "error": f"Audio too long (max {CONFIG['audio']['max_duration_minutes']} minutes)"}
                
                return {
                    "valid": True,
                    "duration": duration,
                    "sample_rate": info.samplerate,
                    "channels": info.channels,
                    "file_size": file_size
                }
                
            except Exception as e:
                return {"valid": False, "error": f"Cannot read audio file: {e}"}
            
        except Exception as e:
            return {"valid": False, "error": f"Validation error: {e}"}


# Utility functions for external use
async def quick_transcribe(audio_file: str, progress_callback: Optional[Callable] = None) -> List[Dict]:
    """Quick transcription function for simple use cases"""
    engine = TranscriptionEngine()
    return await engine.transcribe(audio_file, "pt", progress_callback)


def estimate_transcription_time(audio_duration: float) -> Dict[str, float]:
    """Estimate transcription time based on audio duration"""
    warm_start_time = audio_duration * CONFIG["performance"]["targets"]["processing_ratio_warm"]
    cold_start_time = audio_duration * CONFIG["performance"]["targets"]["processing_ratio_cold"]
    
    return {
        "warm_start_seconds": warm_start_time,
        "cold_start_seconds": cold_start_time,
        "audio_duration": audio_duration
    }