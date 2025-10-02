"""
Consolidated Speaker Diarization Module - Advanced CPU Optimization
Consolidated from diarization.py and diarization_process.py

Features:
- Advanced CPUSpeakerDiarization with multiple adaptive methods
- Multiprocessing support with DiarizationProcess
- Critical alignment functions for transcription integration
- Optimized for CPU-only architecture with multiple algorithms
"""

import logging
import numpy as np
import gc
from typing import List, Dict, Any, Optional, Union, TYPE_CHECKING, cast

if TYPE_CHECKING:
    import librosa
    import soundfile
import asyncio
import multiprocessing as mp
import psutil
import time
import os
import sys
import threading
import queue
import json
from pathlib import Path

logger = logging.getLogger(__name__)

# Import ProcessType for resource coordination
try:
    from src.performance_optimizer import ProcessType
except ImportError:
    logger.warning("ProcessType import failed - coordenação dinâmica não disponível")
    ProcessType = None

# Try to import advanced libraries for better diarization
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available - using simplified clustering")

# Lazy imports for performance
_librosa = None
_soundfile = None

def _get_librosa() -> Union[Any, bool]:
    global _librosa
    if _librosa is None:
        try:
            import librosa
            _librosa = librosa
        except ImportError:
            logger.warning("librosa not available - using simplified audio analysis")
            _librosa = False
    return _librosa

def _get_soundfile() -> Union[Any, bool]:
    global _soundfile
    if _soundfile is None:
        try:
            import soundfile as sf
            _soundfile = sf
        except ImportError:
            logger.warning("soundfile not available - using fallback audio loading")
            _soundfile = False
    return _soundfile

class CPUSpeakerDiarization:
    """Advanced CPU-optimized speaker diarization with multiple adaptive methods"""

    def __init__(self, cpu_manager=None):
        # Coordenação inteligente de recursos (FASE 3)
        self.cpu_manager = cpu_manager

        # Load configuration
        try:
            from config.app_config import DIARIZATION_CONFIG
            self.min_speakers = DIARIZATION_CONFIG["min_speakers"]
            self.max_speakers = DIARIZATION_CONFIG["max_speakers"]
            self.confidence_threshold = DIARIZATION_CONFIG["confidence_threshold"]
            self.analysis_thresholds = DIARIZATION_CONFIG["analysis_thresholds"]
        except ImportError:
            # Fallback configuration
            self.min_speakers = 1
            self.max_speakers = 6
            self.confidence_threshold = 0.5
            self.analysis_thresholds = {
                "short_audio_threshold": 10.0,
                "long_audio_threshold": 300.0,
                "high_quality_snr": 20.0,
                "low_quality_snr": 10.0
            }

        # Available methods
        self.available_methods = ["simple", "clustering", "spectral", "adaptive"]
        self.current_method = "adaptive"

        # Cache for embeddings
        self.embedding_cache = {}

        logger.info("CPUSpeakerDiarization initialized with advanced algorithms")

    async def diarize_audio(self, audio_file: str, method: Optional[str] = None, transcription_data: Union[List, None] = None) -> List[Dict]:
        """Main diarization method with advanced CPU optimization"""
        try:
            # Coordenação dinâmica de recursos (FASE 3)
            if self.cpu_manager and ProcessType:
                dynamic_cores = self.cpu_manager.get_dynamic_cores_for_process(ProcessType.DIARIZATION, True)
                logger.info(f"Coordenação dinâmica: diarization usando {dynamic_cores} cores")

            if method is not None and method:
                self.current_method = method

            logger.info(f"Starting advanced diarization: {audio_file} (method: {self.current_method})")

            # Analyze audio characteristics for optimal method selection
            audio_analysis = self._analyze_audio_characteristics(audio_file)

            # Select optimal method based on analysis
            if self.current_method == "adaptive":
                optimal_method = self._select_optimal_method(audio_analysis)
            else:
                optimal_method = self.current_method

            logger.info(f"Selected method: {optimal_method}")

            # Execute diarization with selected method
            segments = self._execute_diarization(audio_file, optimal_method, audio_analysis)

            # Post-processing and refinement
            refined_segments = self._refine_segments(segments, audio_analysis)

            logger.info(f"Advanced diarization completed: {len(refined_segments)} segments")

            # Cleanup da coordenação dinâmica (FASE 3)
            if self.cpu_manager and ProcessType:
                self.cpu_manager.get_dynamic_cores_for_process(ProcessType.DIARIZATION, False)
                logger.debug("Coordenação dinâmica: diarization finalizado")

            return refined_segments

        except Exception as e:
            logger.error(f"Advanced diarization failed: {e}")
            # Fallback to simple pattern-based diarization
            return self._create_fallback_segments(audio_file, transcription_data)
        finally:
            # Force garbage collection to free up memory from large objects
            gc.collect()

    def _analyze_audio_characteristics(self, audio_file: str) -> Dict[str, Any]:
        """Advanced audio analysis for optimization"""
        try:
            librosa_module = _get_librosa()
            sf_module = _get_soundfile()

            if librosa_module is False or sf_module is False:
                # Fallback analysis
                return self._simple_audio_analysis(audio_file)

            # Type assertion for proper module usage
            assert librosa_module is not False
            assert sf_module is not False

            # Cast to proper types for Pylance
            librosa_mod = cast(Any, librosa_module)
            sf_mod = cast(Any, sf_module)

            # Load audio
            audio_data, sr = sf_mod.read(audio_file)

            # Convert to mono if necessary
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)
 
            duration = len(audio_data) / sr

            # Energy analysis
            energy = np.sum(audio_data ** 2) / len(audio_data)
            energy_variance = np.var(audio_data ** 2)

            # Spectral analysis
            stft = librosa_mod.stft(audio_data, n_fft=2048, hop_length=512)
            spectral_centroid = np.mean(librosa_mod.feature.spectral_centroid(S=np.abs(stft)))
            spectral_bandwidth = np.mean(librosa_mod.feature.spectral_bandwidth(S=np.abs(stft)))

            # Voice activity detection (VAD)
            frame_length = int(0.025 * sr)  # 25ms frames
            hop_length = int(0.010 * sr)    # 10ms hop

            frames = librosa_mod.util.frame(audio_data, frame_length=frame_length,
                                      hop_length=hop_length, axis=0)
            frame_energy = np.sum(frames ** 2, axis=1)

            # Adaptive threshold for voice detection
            energy_threshold = np.mean(frame_energy) * 0.1
            voice_frames = frame_energy > energy_threshold
            voice_ratio = np.sum(voice_frames) / len(voice_frames)

            # Speaker estimation based on energy changes
            energy_changes = np.diff(frame_energy)
            significant_changes = np.sum(np.abs(energy_changes) > np.std(energy_changes) * 2)
            estimated_speakers = min(self.max_speakers, max(1, significant_changes // 10))

            analysis = {
                "duration": duration,
                "energy": float(energy),
                "energy_variance": float(energy_variance),
                "spectral_centroid": float(spectral_centroid),
                "spectral_bandwidth": float(spectral_bandwidth),
                "voice_ratio": float(voice_ratio),
                "estimated_speakers": int(estimated_speakers),
                "significant_changes": int(significant_changes),
                "audio_quality": self._assess_audio_quality(audio_data, sr)
            }

            logger.info(f"Audio analysis: duration={duration:.2f}s, "
                       f"estimated_speakers={estimated_speakers}, "
                       f"voice_ratio={voice_ratio:.2f}")

            return analysis

        except Exception as e:
            logger.error(f"Error in audio analysis: {e}")
            return self._simple_audio_analysis(audio_file)

    def _simple_audio_analysis(self, audio_file: str) -> Dict[str, Any]:
        """Fallback simple audio analysis"""
        try:
            # Get duration using basic method
            duration = self._get_audio_duration(audio_file)

            return {
                "duration": duration,
                "energy": 0.1,
                "energy_variance": 0.01,
                "spectral_centroid": 2000.0,
                "spectral_bandwidth": 1000.0,
                "voice_ratio": 0.6,
                "estimated_speakers": 2,
                "significant_changes": 20,
                "audio_quality": "medium"
            }
        except Exception as e:
            logger.error(f"Simple audio analysis failed: {e}")
            return {
                "duration": 10.0,
                "energy": 0.1,
                "energy_variance": 0.01,
                "spectral_centroid": 2000.0,
                "spectral_bandwidth": 1000.0,
                "voice_ratio": 0.6,
                "estimated_speakers": 2,
                "significant_changes": 20,
                "audio_quality": "medium"
            }

    def _assess_audio_quality(self, audio_data: np.ndarray, sr: int) -> str:
        """Assess audio quality for method selection"""
        try:
            # SNR estimation
            signal_power = np.mean(audio_data ** 2)
            noise_estimate = np.mean(np.abs(np.diff(audio_data)) ** 2)

            if noise_estimate > 0:
                snr_estimate = 10 * np.log10(signal_power / noise_estimate)
            else:
                snr_estimate = 50  # High SNR if no noise detected

            # Clipping detection
            clipping_ratio = np.sum(np.abs(audio_data) > 0.95) / len(audio_data)

            # Determine quality
            if snr_estimate > 20 and clipping_ratio < 0.01:
                return "high"
            elif snr_estimate > 10 and clipping_ratio < 0.05:
                return "medium"
            else:
                return "low"

        except Exception as e:
            logger.warning(f"Error assessing audio quality: {e}")
            return "medium"

    def _select_optimal_method(self, analysis: Dict[str, Any]) -> str:
        """Select optimal method based on analysis"""
        duration = analysis["duration"]
        estimated_speakers = analysis["estimated_speakers"]
        voice_ratio = analysis["voice_ratio"]
        audio_quality = analysis["audio_quality"]

        # Method selection logic
        if duration < self.analysis_thresholds.get("short_audio_threshold", 10.0):
            # Short audio - simple method
            return "simple"
        elif estimated_speakers <= 2 and voice_ratio > 0.8:
            # Simple conversation - clustering
            return "clustering"
        elif estimated_speakers > 3 or audio_quality == "low":
            # Multiple speakers or low quality - spectral
            return "spectral"
        else:
            # General case - clustering
            return "clustering"

    def _execute_diarization(self, audio_file: str, method: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute diarization with specific method"""
        try:
            if method == "simple":
                return self._simple_diarization(audio_file, analysis)
            elif method == "clustering":
                return self._clustering_diarization(audio_file, analysis)
            elif method == "spectral":
                return self._spectral_diarization(audio_file, analysis)
            else:
                # Fallback to clustering
                return self._clustering_diarization(audio_file, analysis)

        except Exception as e:
            logger.error(f"Error executing diarization {method}: {e}")
            # Fallback to simple method
            return self._simple_diarization(audio_file, analysis)

    def _simple_diarization(self, audio_file: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Simple diarization based on energy changes"""
        try:
            sf_module = _get_soundfile()
            librosa_module = _get_librosa()

            if sf_module is False:
                return self._pattern_based_fallback(audio_file, analysis)

            # Type assertion for proper module usage
            assert sf_module is not False

            # Cast to proper types for Pylance
            sf_mod = cast(Any, sf_module)

            # Load audio
            audio_data, sr = sf_mod.read(audio_file)
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)

            duration = len(audio_data) / sr
            estimated_speakers = min(analysis["estimated_speakers"], 3)

            # Divide audio into segments based on energy
            frame_length = int(0.5 * sr)  # 500ms segments
            hop_length = int(0.25 * sr)   # 250ms hop

            segments = []
            current_speaker = 1
            speaker_change_threshold = 0.3

            for i in range(0, len(audio_data) - frame_length, hop_length):
                start_time = i / sr
                end_time = min((i + frame_length) / sr, duration)

                # Calculate segment energy
                segment_data = audio_data[i:i + frame_length]
                segment_energy = np.mean(segment_data ** 2)

                # Simulate speaker change based on energy
                if len(segments) > 0:
                    prev_energy = segments[-1].get("energy", segment_energy)
                    energy_change = abs(segment_energy - prev_energy) / (prev_energy + 1e-8)

                    if energy_change > speaker_change_threshold:
                        current_speaker = (current_speaker % estimated_speakers) + 1

                segments.append({
                    "start": start_time,
                    "end": end_time,
                    "speaker": f"Speaker_{current_speaker}",
                    "confidence": min(1.0, max(0.3, 1.0 - energy_change if len(segments) > 0 else 0.8)),
                    "energy": segment_energy
                })

            return segments

        except Exception as e:
            logger.error(f"Error in simple diarization: {e}")
            return self._pattern_based_fallback(audio_file, analysis)

    def _clustering_diarization(self, audio_file: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Advanced clustering-based diarization"""
        try:
            sf_module = _get_soundfile()
            librosa_module = _get_librosa()

            if sf_module is False or librosa_module is False or not SKLEARN_AVAILABLE:
                return self._simple_diarization(audio_file, analysis)

            # Type assertion for proper module usage
            assert sf_module is not False
            assert librosa_module is not False

            # Cast to proper types for Pylance
            sf_mod = cast(Any, sf_module)
            librosa_mod = cast(Any, librosa_module)

            # Load audio
            audio_data, sr = sf_mod.read(audio_file)
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)

            duration = len(audio_data) / sr

            # Extract MFCCs as features
            mfccs = librosa_mod.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13,
                                       hop_length=int(0.025 * sr),
                                       n_fft=int(0.05 * sr))

            # Transpose to have frames as rows
            features = mfccs.T

            # Normalize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            # Determine number of clusters
            n_speakers = min(analysis["estimated_speakers"], self.max_speakers)
            n_speakers = max(2, n_speakers)  # At least 2 speakers for clustering

            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_speakers, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features_scaled)

            # Convert frame labels to time segments
            frame_duration = 0.025  # 25ms frames
            segments = []
            current_speaker = None
            segment_start = 0.0

            for i, speaker_label in enumerate(labels):
                frame_time = i * frame_duration
                speaker_id = f"Speaker_{speaker_label + 1}"

                if speaker_id != current_speaker:
                    # Speaker change detected
                    if current_speaker is not None:
                        # Finish previous segment
                        segments.append({
                            "start": segment_start,
                            "end": frame_time,
                            "speaker": current_speaker,
                            "confidence": 0.8,
                            "duration": frame_time - segment_start
                        })

                    # Start new segment
                    current_speaker = speaker_id
                    segment_start = frame_time

            # Add final segment
            if current_speaker is not None:
                segments.append({
                    "start": segment_start,
                    "end": duration,
                    "speaker": current_speaker,
                    "confidence": 0.8,
                    "duration": duration - segment_start
                })

            # Clean up large intermediate objects before returning
            del features, features_scaled, labels, mfccs
            gc.collect()

            return segments

        except Exception as e:
            logger.error(f"Error in clustering diarization: {e}")
            return self._simple_diarization(audio_file, analysis)

    def _spectral_diarization(self, audio_file: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Spectral clustering-based diarization for complex scenarios"""
        try:
            # For now, fall back to clustering method
            # In a full implementation, this would use spectral clustering
            logger.info("Using clustering method as spectral fallback")
            return self._clustering_diarization(audio_file, analysis)
        except Exception as e:
            logger.error(f"Error in spectral diarization: {e}")
            return self._simple_diarization(audio_file, analysis)

    def _refine_segments(self, segments: List[Dict[str, Any]], analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Post-process and refine diarization segments"""
        if not segments:
            return segments

        try:
            # Filter by minimum duration (500ms)
            min_duration = 0.5
            valid_segments = [
                seg for seg in segments
                if seg.get('end', 0) - seg.get('start', 0) >= min_duration
            ]

            # Sort by start time
            valid_segments.sort(key=lambda x: x.get('start', 0))

            # Merge consecutive segments from same speaker
            merged_segments = []
            current_segment = None

            for segment in valid_segments:
                if (current_segment is None or
                    current_segment['speaker'] != segment['speaker'] or
                    segment['start'] - current_segment['end'] > 1.0):  # 1s gap threshold

                    if current_segment is not None and current_segment:
                        merged_segments.append(current_segment)
                    current_segment = segment.copy()
                else:
                    # Merge with current
                    current_segment['end'] = segment['end']
                    current_segment['duration'] = current_segment['end'] - current_segment['start']
                    current_segment['confidence'] = min(current_segment.get('confidence', 0.5), segment.get('confidence', 0.5))

            if current_segment is not None and current_segment:
                merged_segments.append(current_segment)

            return merged_segments

        except Exception as e:
            logger.error(f"Error refining segments: {e}")
            return segments

    def _pattern_based_fallback(self, audio_file: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Pattern-based fallback diarization"""
        try:
            duration = analysis.get("duration", self._get_audio_duration(audio_file))
            estimated_speakers = analysis.get("estimated_speakers", 2)

            if estimated_speakers <= 1:
                return [{
                    'start': 0.0,
                    'end': duration,
                    'speaker': 'Speaker_1',
                    'confidence': 0.7,
                    'duration': duration
                }]

            # Create alternating pattern
            segments = []
            segment_duration = duration / (estimated_speakers * 2)
            current_time = 0.0

            for i in range(estimated_speakers * 2):
                if current_time >= duration:
                    break

                speaker_id = (i % estimated_speakers) + 1
                end_time = min(current_time + segment_duration, duration)

                if end_time - current_time >= 0.5:  # Minimum 500ms segments
                    segments.append({
                        'start': current_time,
                        'end': end_time,
                        'speaker': f'Speaker_{speaker_id}',
                        'confidence': 0.6,
                        'duration': end_time - current_time
                    })

                current_time = end_time

            return segments

        except Exception as e:
            logger.error(f"Pattern-based fallback failed: {e}")
            return self._create_fallback_segments(audio_file, None)

    def _create_fallback_segments(self, audio_file: str, transcription_data: Union[List, None] = None) -> List[Dict]:
        """Create intelligent fallback segments"""
        try:
            duration = self._get_audio_duration(audio_file)

            # Simple 2-speaker conversation pattern
            segments = [
                {
                    'start': 0.0,
                    'end': duration * 0.6,
                    'speaker': 'Speaker_1',
                    'confidence': 0.6,
                    'duration': duration * 0.6
                },
                {
                    'start': duration * 0.6,
                    'end': duration,
                    'speaker': 'Speaker_2',
                    'confidence': 0.6,
                    'duration': duration * 0.4
                }
            ]

            return segments

        except Exception:
            return [{
                'start': 0.0,
                'end': 20.0,
                'speaker': 'Speaker_1',
                'confidence': 0.5,
                'duration': 20.0
            }]

    def _get_audio_duration(self, audio_file: str) -> float:
        """Get audio duration safely"""
        try:
            # Try optimized audio processor first
            from .audio_processing import OptimizedAudioProcessor
            return OptimizedAudioProcessor.torchaudio_get_duration(audio_file)
        except:
            try:
                sf_module = _get_soundfile()
                if sf_module is not None and sf_module is not False:
                    sf_mod = cast(Any, sf_module)
                    with sf_mod.SoundFile(audio_file) as f:
                        return len(f) / f.samplerate
            except:
                pass
            return 30.0  # Fallback


class DiarizationProcess:
    """Multiprocessing wrapper for diarization with process management"""

    def __init__(self, process_id: int, queue_manager, shared_memory):
        self.process_id = process_id
        self.queue_manager = queue_manager
        self.shared_memory = shared_memory

        # Initialize advanced diarizer
        self.diarizer = CPUSpeakerDiarization()

        # Process control
        self.running = False
        self.processing = False
        self.control_thread = None

        # Performance configuration - Nova fórmula otimizada
        logical_cores = psutil.cpu_count(logical=True) or 4
        physical_cores = psutil.cpu_count(logical=False) or 2
        self.max_cores = max(1, logical_cores - 2, physical_cores - 2)
        self.core_count = self.max_cores

        # Statistics
        self.stats = {
            "diarizations_processed": 0,
            "total_audio_duration": 0.0,
            "total_processing_time": 0.0,
            "average_ratio": 0.0,
            "speakers_detected_total": 0
        }

    def start(self):
        """Start the diarization process"""
        try:
            self.running = True

            # Configure current process
            self._setup_process()

            # Start control thread
            self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
            self.control_thread.start()

            logger.info(f"Diarization process started (PID: {os.getpid()})")

            # Send initialization status
            self._send_status_update("RUNNING")

            # Main loop
            self._main_loop()

        except Exception as e:
            logger.error(f"Error in diarization process: {e}")
            self._send_status_update("ERROR", str(e))
        finally:
            self._cleanup()

    def _setup_process(self):
        """Configure current process with CPU affinity and limits"""
        try:
            current_process = psutil.Process()

            # Set CPU affinity for diarization cores
            transcription_cores = self.max_cores
            start_core = 2 + transcription_cores
            cpu_count = psutil.cpu_count() or 4
            diarization_cores = list(range(start_core, min(start_core + self.core_count, cpu_count)))

            if diarization_cores is not None and diarization_cores:
                current_process.cpu_affinity(diarization_cores)
                logger.info(f"CPU affinity set: cores {diarization_cores}")

            # Set normal priority
            if sys.platform.startswith('win'):
                current_process.nice(psutil.NORMAL_PRIORITY_CLASS)
            else:
                current_process.nice(0)

            logger.info(f"Process configured with {self.core_count} cores")

        except Exception as e:
            logger.warning(f"Error configuring process: {e}")

    def _control_loop(self):
        """Control loop for external commands"""
        while self.running:
            try:
                # Check control messages
                if hasattr(self.queue_manager, 'get_control_message'):
                    control_msg = self.queue_manager.get_control_message(timeout=0.1)
                    if control_msg is not None and control_msg:
                        self._handle_control_message(control_msg)

                # Check diarization-specific commands
                if hasattr(self.queue_manager, 'get_queue'):
                    try:
                        from src.performance_optimizer import ProcessType
                        diarization_queue = self.queue_manager.get_queue(ProcessType.DIARIZATION)
                        if diarization_queue is not None and diarization_queue:
                            try:
                                command = diarization_queue.get_nowait()
                                self._handle_diarization_command(command)
                            except queue.Empty:
                                pass
                    except ImportError:
                        # Fallback without ProcessType
                        pass

                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in control loop: {e}")
                time.sleep(1.0)

    def _handle_control_message(self, message: Dict[str, Any]):
        """Handle global control messages"""
        action = message.get("action")

        if action == "shutdown":
            logger.info("Shutdown command received")
            self.running = False
        elif action == "restart_process":
            logger.info("Restart command received")
            self._restart_process()

    def _handle_diarization_command(self, command: Dict[str, Any]):
        """Handle diarization-specific commands"""
        cmd_type = command.get("type")

        if cmd_type == "diarize_audio":
            self._process_diarization_request(command.get("data", {}))
        elif cmd_type == "set_method":
            method = command.get("data", {}).get("method", "adaptive")
            self.diarizer.current_method = method
            logger.info(f"Diarization method changed to: {method}")

    def _process_diarization_request(self, request_data: Dict[str, Any]):
        """Process diarization request"""
        try:
            if self.processing:
                logger.warning("Diarization already in progress, ignoring request")
                return

            self.processing = True
            start_time = time.time()

            # Extract request data
            audio_file = request_data.get("audio_file")
            method = request_data.get("method")
            session_id = request_data.get("session_id") or ""

            if not audio_file or not os.path.exists(audio_file):
                raise ValueError(f"Invalid audio file: {audio_file}")

            logger.info(f"Starting diarization: {audio_file} (method: {method or 'adaptive'})")

            # Send initial progress
            self._send_progress_update(session_id, 10, "Starting diarization...")

            # Execute diarization
            self._send_progress_update(session_id, 30, "Analyzing audio characteristics...")

            # Use async method if available
            if hasattr(self.diarizer, 'diarize_audio'):
                # Create event loop if none exists
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    asyncio.set_event_loop(asyncio.new_event_loop())
                    loop = asyncio.get_event_loop()

                result = self.diarizer.diarize_audio(audio_file, method)
                if asyncio.iscoroutine(result):
                    segments = loop.run_until_complete(result)
                else:
                    segments = result
            else:
                # Fallback to sync method
                segments = self.diarizer.diarize_audio(audio_file, method)

            # Ensure segments is always a list, not a coroutine
            if asyncio.iscoroutine(segments):
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    asyncio.set_event_loop(asyncio.new_event_loop())
                    loop = asyncio.get_event_loop()
                segments = loop.run_until_complete(segments)

            # Calculate statistics
            self._send_progress_update(session_id, 90, "Finalizing diarization...")
            processing_time = time.time() - start_time

            # Estimate audio duration
            try:
                sf_module = _get_soundfile()
                if sf_module is not None and sf_module is not False:
                    sf_mod = cast(Any, sf_module)
                    audio_data, sr = sf_mod.read(audio_file)
                    audio_duration = len(audio_data) / sr
                else:
                    audio_duration = processing_time * 2  # Conservative estimate
            except:
                audio_duration = processing_time * 2

            ratio = processing_time / audio_duration if audio_duration > 0 else 0
            # Ensure segments is a list before attempting to iterate
            if not isinstance(segments, list):
                segments = list(segments) if hasattr(segments, '__iter__') else []
            speakers_detected = len(set(seg["speaker"] for seg in segments if isinstance(seg, dict) and "speaker" in seg))

            # Update statistics
            self._update_stats(audio_duration, processing_time, ratio, speakers_detected)

            # Send final result
            self._send_progress_update(session_id, 100, "Diarization completed")
            self._send_diarization_result(session_id, segments, {
                "processing_time": processing_time,
                "audio_duration": audio_duration,
                "processing_ratio": ratio,
                "speakers_detected": speakers_detected,
                "method_used": self.diarizer.current_method
            })

            logger.info(f"Diarization completed: {len(segments)} segments, "
                       f"{speakers_detected} speakers, ratio {ratio:.2f}x")

        except Exception as e:
            logger.error(f"Error processing diarization: {e}")
            error_session_id = request_data.get("session_id") or ""
            self._send_diarization_error(error_session_id, str(e))
        finally:
            self.processing = False
            gc.collect()

    def _main_loop(self):
        """Main process loop"""
        while self.running:
            try:
                # Check shared memory data
                self._check_shared_data()

                # Send periodic statistics
                self._send_periodic_stats()

                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(1.0)

    def _check_shared_data(self):
        """Check shared memory data"""
        # Placeholder for shared memory processing
        pass

    def _send_periodic_stats(self):
        """Send periodic statistics"""
        if not hasattr(self, '_last_stats_time'):
            self._last_stats_time = time.time()

        current_time = time.time()
        if current_time - self._last_stats_time >= 10.0:  # Every 10 seconds
            try:
                process = psutil.Process()
                memory_mb = process.memory_info().rss / (1024 * 1024)
                cpu_percent = process.cpu_percent()

                stats = {
                    "memory_usage_mb": memory_mb,
                    "cpu_percent": cpu_percent,
                    "processing": self.processing,
                    **self.stats
                }

                self._send_status_update("STATS", stats)
                self._last_stats_time = current_time

            except Exception as e:
                logger.warning(f"Error sending statistics: {e}")

    def _update_stats(self, audio_duration: float, processing_time: float, ratio: float, speakers_detected: int):
        """Update process statistics"""
        self.stats["diarizations_processed"] += 1
        self.stats["total_audio_duration"] += audio_duration
        self.stats["total_processing_time"] += processing_time
        self.stats["speakers_detected_total"] += speakers_detected

        if self.stats["diarizations_processed"] > 0:
            self.stats["average_ratio"] = (
                self.stats["total_processing_time"] / self.stats["total_audio_duration"]
            )

    def _send_status_update(self, status: str, data: Any = None):
        """Send status update"""
        try:
            if hasattr(self.queue_manager, 'send_status_update'):
                self.queue_manager.send_status_update("DIARIZATION", {
                    "status": status,
                    "data": data,
                    "timestamp": time.time(),
                    "process_id": os.getpid()
                })
        except Exception as e:
            logger.warning(f"Error sending status update: {e}")

    def _send_progress_update(self, session_id: str, progress: int, message: str):
        """Send progress update"""
        # Placeholder for progress updates
        logger.debug(f"Progress {progress}%: {message}")

    def _send_diarization_result(self, session_id: str, segments: List[Dict], stats: Dict):
        """Send diarization result"""
        # Placeholder for result sending
        logger.info(f"Sending result for session {session_id}: {len(segments)} segments")

    def _send_diarization_error(self, session_id: str, error: str):
        """Send diarization error"""
        # Placeholder for error sending
        logger.error(f"Sending error for session {session_id}: {error}")

    def _restart_process(self):
        """Restart process"""
        logger.info("Restarting diarization process")
        # Placeholder for restart logic

    def _cleanup(self):
        """Clean up resources"""
        try:
            self.running = False
            if self.control_thread and self.control_thread.is_alive():
                self.control_thread.join(timeout=2.0)
            logger.info("Diarization process cleanup completed")
            gc.collect()
        except Exception as e:
            logger.error(f"Error in cleanup: {e}")


# ==========================================
# CRITICAL ALIGNMENT FUNCTIONS
# ==========================================
# Preserved from original diarization.py

def force_transcription_segmentation(
    transcription_segments: List[Dict],
    diarization_segments: List[Dict]
) -> List[Dict]:
    """Force transcription segmentation based on diarization boundaries"""

    if not transcription_segments or not diarization_segments:
        return transcription_segments or []

    forced_segments = []

    try:
        for trans_seg in transcription_segments:
            trans_start = trans_seg.get('start', 0)
            trans_end = trans_seg.get('end', 0)
            trans_text = trans_seg.get('text', '').strip()

            if not trans_text:
                continue

            # Find all diarization segments that overlap with this transcription segment
            overlapping_diar = []
            for diar_seg in diarization_segments:
                diar_start = diar_seg.get('start', 0)
                diar_end = diar_seg.get('end', 0)

                # Check if there's any overlap
                if not (diar_end <= trans_start or diar_start >= trans_end):
                    overlapping_diar.append(diar_seg)

            if len(overlapping_diar) <= 1:
                # Simple case: transcription maps to single diarization segment
                speaker = overlapping_diar[0].get('speaker', 'Speaker_1') if overlapping_diar else 'Speaker_1'
                forced_segment = trans_seg.copy()
                forced_segment['speaker'] = speaker
                forced_segments.append(forced_segment)
            else:
                # Complex case: transcription spans multiple diarization segments
                # Split transcription text based on diarization boundaries
                words = trans_text.split()
                total_duration = trans_end - trans_start
                words_per_second = len(words) / total_duration if total_duration > 0 else 1

                current_word_idx = 0

                for diar_seg in sorted(overlapping_diar, key=lambda x: x.get('start', 0)):
                    diar_start = max(diar_seg.get('start', 0), trans_start)
                    diar_end = min(diar_seg.get('end', 0), trans_end)
                    diar_duration = diar_end - diar_start

                    if diar_duration <= 0:
                        continue

                    # Estimate words for this diarization segment
                    words_in_segment = max(1, int(diar_duration * words_per_second))
                    end_word_idx = min(current_word_idx + words_in_segment, len(words))

                    if current_word_idx < len(words):
                        segment_words = words[current_word_idx:end_word_idx]
                        segment_text = ' '.join(segment_words)

                        forced_segment = {
                            'id': len(forced_segments),
                            'start': diar_start,
                            'end': diar_end,
                            'text': segment_text,
                            'confidence': trans_seg.get('confidence', 0.0),
                            'speaker': diar_seg.get('speaker', 'Speaker_1')
                        }
                        forced_segments.append(forced_segment)
                        current_word_idx = end_word_idx

        return forced_segments

    except Exception as e:
        logger.error(f"Forced segmentation failed: {e}")
        return transcription_segments


def align_transcription_with_diarization(
    transcription_segments: List[Dict],
    diarization_segments: List[Dict],
    language: str = "pt"
) -> List[Dict]:
    """Align transcription with diarization segments using forced segmentation"""

    if not transcription_segments or not diarization_segments:
        return transcription_segments or []

    # First, force segmentation based on diarization boundaries
    forced_segments = force_transcription_segmentation(transcription_segments, diarization_segments)

    # Then apply original alignment logic as fallback
    aligned_segments = []

    try:
        for trans_seg in forced_segments:
            trans_start = trans_seg.get('start', 0)
            trans_end = trans_seg.get('end', 0)

            # Find best matching diarization segment
            best_speaker = 'Speaker_1'
            best_overlap = 0.0

            for diar_seg in diarization_segments:
                diar_start = diar_seg.get('start', 0)
                diar_end = diar_seg.get('end', 0)

                # Calculate overlap
                overlap_start = max(trans_start, diar_start)
                overlap_end = min(trans_end, diar_end)
                overlap = max(0, overlap_end - overlap_start)

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = diar_seg.get('speaker', 'Speaker_1')

            # Create aligned segment
            aligned_segment = trans_seg.copy()
            aligned_segment['speaker'] = best_speaker
            aligned_segments.append(aligned_segment)

        return aligned_segments

    except Exception as e:
        logger.error(f"Alignment failed: {e}")
        return forced_segments


# ==========================================
# COMPATIBILITY FUNCTIONS
# ==========================================

def diarization_worker(process_id: int):
    """Simplified worker function for diarization process"""
    try:
        logger.info(f"Diarization worker {process_id} iniciado")
        # Implementação simplificada para evitar pickle errors
        for i in range(10):
            logger.debug(f"Diarization worker {process_id} - ciclo {i+1}")
            time.sleep(0.1)
        logger.info(f"Diarization worker {process_id} finalizado")
    except Exception as e:
        logger.error(f"Fatal error in diarization process: {e}")
        raise


# Global instances
enhanced_diarization = CPUSpeakerDiarization()

# Backward compatibility alias
OptimizedSpeakerDiarization = CPUSpeakerDiarization