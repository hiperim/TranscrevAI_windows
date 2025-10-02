"""
Phase 6.1 - Simple Hardware Optimization
Basic CPU threading and memory optimizations
"""

import os
import logging
import psutil
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class SimpleHardwareOptimizer:
    """Simple hardware optimization for Phase 6.1"""

    def __init__(self):
        self.cpu_count = os.cpu_count() or 4
        self.memory_gb = psutil.virtual_memory().total / (1024 ** 3)
        self.optimal_threads = self._calculate_optimal_threads()
        self.cuda_streams = None  # Initialize cuda_streams attribute

        # Apply initial optimizations
        self.apply_cpu_optimizations()

        logger.info(f"Hardware detected: {self.cpu_count} cores, {self.memory_gb:.1f}GB RAM")
        logger.info(f"Optimal threads: {self.optimal_threads}")

    def _calculate_optimal_threads(self) -> int:
        """Calculate optimal thread count for CPU processing"""
        if self.cpu_count >= 8:
            return 4  # Use 50% of cores on high-end systems
        elif self.cpu_count >= 4:
            return 2  # Use 50% of cores on mid-range systems
        else:
            return 1  # Conservative on low-end systems

    def apply_cpu_optimizations(self):
        """Apply basic CPU optimizations"""
        try:
            # Set environment variables for optimal CPU performance
            os.environ['OMP_NUM_THREADS'] = str(self.optimal_threads)
            os.environ['MKL_NUM_THREADS'] = str(self.optimal_threads)
            os.environ['NUMEXPR_NUM_THREADS'] = str(self.optimal_threads)
            os.environ['OPENBLAS_NUM_THREADS'] = str(self.optimal_threads)

            # Disable TensorFlow parallelism warnings
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

            logger.info(f"Applied CPU optimizations with {self.optimal_threads} threads")

        except Exception as e:
            logger.error(f"Failed to apply CPU optimizations: {e}")

    def get_system_info(self) -> Dict[str, Any]:
        """Get current system information"""
        try:
            memory = psutil.virtual_memory()
            cpu_freq = psutil.cpu_freq()

            return {
                "cpu_cores": self.cpu_count,
                "cpu_frequency_mhz": cpu_freq.current if cpu_freq else 0,
                "memory_total_gb": self.memory_gb,
                "memory_available_gb": memory.available / (1024 ** 3),
                "memory_percent_used": memory.percent,
                "optimal_threads": self.optimal_threads,
                "platform": os.name
            }

        except Exception as e:
            logger.error(f"Failed to get system info: {e}")
            return {"error": str(e)}

    def check_memory_pressure(self) -> Dict[str, Any]:
        """Check if system is under memory pressure"""
        try:
            memory = psutil.virtual_memory()
            current_process = psutil.Process()
            process_memory = current_process.memory_info().rss / (1024 ** 3)

            return {
                "system_memory_percent": memory.percent,
                "system_available_gb": memory.available / (1024 ** 3),
                "process_memory_gb": process_memory,
                "memory_pressure": memory.percent > 80,
                "process_memory_high": process_memory > 2.0,
                "recommendations": self._get_memory_recommendations(memory.percent, process_memory)
            }

        except Exception as e:
            logger.error(f"Failed to check memory pressure: {e}")
            return {"error": str(e)}

    def _get_memory_recommendations(self, system_percent: float, process_gb: float) -> List[str]:
        """Get memory optimization recommendations"""
        recommendations = []

        if system_percent > 85:
            recommendations.append("System memory usage is high - consider closing other applications")

        if process_gb > 2.0:
            recommendations.append("Process memory usage exceeds 2GB limit - aggressive cleanup needed")

        if system_percent > 90:
            recommendations.append("Critical memory pressure - consider using smaller audio chunks")

        if len(recommendations) == 0:
            recommendations.append("Memory usage is within normal limits")

        return recommendations

    def optimize_for_task(self, task_type: str) -> Dict[str, Any]:
        """Optimize system for specific task"""
        try:
            if task_type == "model_loading":
                # Conservative settings for model loading
                threads = max(1, self.optimal_threads - 1)

            elif task_type == "transcription":
                # Use full optimal threads for transcription
                threads = self.optimal_threads

            elif task_type == "audio_processing":
                # Moderate threads for audio processing
                threads = min(2, self.optimal_threads)

            else:
                threads = self.optimal_threads

            # Apply thread settings
            os.environ['OMP_NUM_THREADS'] = str(threads)
            os.environ['MKL_NUM_THREADS'] = str(threads)

            return {
                "task_type": task_type,
                "threads_used": threads,
                "optimization_applied": True
            }

        except Exception as e:
            logger.error(f"Failed to optimize for task {task_type}: {e}")
            return {"error": str(e)}

    def get_performance_profile(self) -> str:
        """Get system performance profile"""
        if self.cpu_count >= 8 and self.memory_gb >= 8:
            return "high_performance"
        elif self.cpu_count >= 4 and self.memory_gb >= 4:
            return "medium_performance"
        else:
            return "low_performance"

# Global instances
hardware_optimizer = SimpleHardwareOptimizer()
phase3_optimizer = SimpleHardwareOptimizer()