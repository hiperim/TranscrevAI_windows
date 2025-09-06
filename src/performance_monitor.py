# Performance Monitor for TranscrevAI
# Real-time performance monitoring and optimization system

"""
PerformanceMonitor

Comprehensive performance monitoring system that:
- Tracks real-time processing ratios (must be < 1.0)
- Monitors memory usage and model loading times
- Auto-adjusts parameters based on performance metrics
- Provides alerts for performance degradation
- Maintains detailed performance analytics
"""

import asyncio
import time
import logging
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import deque
import json
from pathlib import Path

from config.whisper_optimization import validate_real_time_performance, PERFORMANCE_CONSTRAINTS

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    timestamp: float
    metric_type: str
    value: float
    metadata: Dict[str, Any]

@dataclass
class ProcessingStats:
    """Statistics for a processing session"""
    session_id: str
    audio_duration: float
    transcription_time: float
    diarization_time: float
    total_processing_time: float
    real_time_ratio: float
    memory_usage_mb: float
    model_load_time: float
    language: str
    model_name: str
    chunk_count: int
    start_time: float
    end_time: float

class PerformanceAlert:
    """Performance alert system"""
    
    def __init__(self, threshold_type: str, threshold_value: float, message: str):
        self.threshold_type = threshold_type
        self.threshold_value = threshold_value 
        self.message = message
        self.timestamp = time.time()
        self.acknowledged = False

class PerformanceMonitor:
    """
    Monitor and optimize system performance in real-time
    
    Features:
    - Real-time ratio monitoring (processing_time / audio_duration)
    - Memory usage tracking and alerts
    - Model loading performance analysis
    - Automatic parameter optimization
    - Performance degradation detection
    - Historical performance analytics
    """
    
    def __init__(self, max_history_size: int = 1000):
        """
        Initialize performance monitor
        
        Args:
            max_history_size: Maximum number of historical metrics to keep
        """
        # Metrics storage
        self.metrics_history = deque(maxlen=max_history_size)
        self.processing_stats: List[ProcessingStats] = []
        self.active_sessions: Dict[str, Dict] = {}
        
        # Performance tracking
        self.real_time_ratios = deque(maxlen=100)  # Last 100 processing ratios
        self.memory_usage_history = deque(maxlen=100)
        self.model_load_times = deque(maxlen=50)
        
        # Alert system
        self.active_alerts: List[PerformanceAlert] = []
        self.alert_callbacks: List[Callable[[PerformanceAlert], None]] = []
        
        # Configuration
        self.real_time_threshold = PERFORMANCE_CONSTRAINTS["max_processing_ratio"]  # 1.0
        self.memory_warning_threshold = PERFORMANCE_CONSTRAINTS["memory_limit_mb"] * 0.8  # 80% of limit
        self.memory_critical_threshold = PERFORMANCE_CONSTRAINTS["memory_limit_mb"] * 0.95  # 95% of limit
        
        # Background monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
        self._monitoring_active = False
        
        # Thread safety
        self._metrics_lock = asyncio.Lock()
        
        logger.info("PerformanceMonitor initialized with real-time monitoring")
    
    async def start_monitoring(self, monitoring_interval: float = 5.0):
        """Start background performance monitoring"""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitoring_task = asyncio.create_task(self._monitor_loop(monitoring_interval))
        logger.info(f"Performance monitoring started (interval: {monitoring_interval}s)")
    
    async def stop_monitoring(self):
        """Stop background performance monitoring"""
        self._monitoring_active = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Performance monitoring stopped")
    
    async def _monitor_loop(self, interval: float):
        """Background monitoring loop"""
        while self._monitoring_active:
            try:
                await self._collect_system_metrics()
                await self._check_performance_thresholds()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(interval)
    
    async def track_processing_session(self, session_id: str, audio_duration: float, language: str, model_name: str):
        """Start tracking a processing session"""
        async with self._metrics_lock:
            self.active_sessions[session_id] = {
                "start_time": time.time(),
                "audio_duration": audio_duration,
                "language": language,
                "model_name": model_name,
                "transcription_start": None,
                "transcription_end": None,
                "diarization_start": None,
                "diarization_end": None,
                "model_load_time": 0.0,
                "memory_start": await self._get_memory_usage(),
                "chunk_count": 0
            }
        
        logger.debug(f"Started tracking session {session_id}: {audio_duration:.2f}s audio, {language} language")
    
    async def record_transcription_timing(self, session_id: str, start_time: float, end_time: float):
        """Record transcription timing for a session"""
        async with self._metrics_lock:
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["transcription_start"] = start_time
                self.active_sessions[session_id]["transcription_end"] = end_time
    
    async def record_diarization_timing(self, session_id: str, start_time: float, end_time: float):
        """Record diarization timing for a session"""
        async with self._metrics_lock:
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["diarization_start"] = start_time
                self.active_sessions[session_id]["diarization_end"] = end_time
    
    async def record_model_load_time(self, session_id: str, load_time: float):
        """Record model loading time"""
        async with self._metrics_lock:
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["model_load_time"] = load_time
            self.model_load_times.append(load_time)
    
    async def record_chunk_processed(self, session_id: str):
        """Record that a chunk was processed"""
        async with self._metrics_lock:
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["chunk_count"] += 1
    
    async def complete_processing_session(self, session_id: str) -> Optional[ProcessingStats]:
        """Complete tracking and calculate final statistics"""
        async with self._metrics_lock:
            if session_id not in self.active_sessions:
                return None
            
            session = self.active_sessions[session_id]
            end_time = time.time()
            
            # Calculate timing statistics
            transcription_time = 0.0
            if session["transcription_start"] and session["transcription_end"]:
                transcription_time = session["transcription_end"] - session["transcription_start"]
            
            diarization_time = 0.0
            if session["diarization_start"] and session["diarization_end"]:
                diarization_time = session["diarization_end"] - session["diarization_start"]
            
            total_processing_time = end_time - session["start_time"]
            real_time_ratio = total_processing_time / session["audio_duration"] if session["audio_duration"] > 0 else 0
            
            # Create statistics
            stats = ProcessingStats(
                session_id=session_id,
                audio_duration=session["audio_duration"],
                transcription_time=transcription_time,
                diarization_time=diarization_time,
                total_processing_time=total_processing_time,
                real_time_ratio=real_time_ratio,
                memory_usage_mb=await self._get_memory_usage(),
                model_load_time=session["model_load_time"],
                language=session["language"],
                model_name=session["model_name"],
                chunk_count=session["chunk_count"],
                start_time=session["start_time"],
                end_time=end_time
            )
            
            # Store statistics
            self.processing_stats.append(stats)
            self.real_time_ratios.append(real_time_ratio)
            
            # Clean up active session
            del self.active_sessions[session_id]
            
            # Check for performance issues
            await self._check_session_performance(stats)
            
            logger.info(f"Session {session_id} completed: {real_time_ratio:.2f}x real-time ratio")
            return stats
    
    async def _collect_system_metrics(self):
        """Collect system-wide performance metrics"""
        try:
            # Memory usage
            memory_mb = await self._get_memory_usage()
            self.memory_usage_history.append(memory_mb)
            
            await self._record_metric("system_memory", memory_mb, {"unit": "MB"})
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            await self._record_metric("system_cpu", cpu_percent, {"unit": "percent"})
            
            # Active sessions count
            sessions_count = len(self.active_sessions)
            await self._record_metric("active_sessions", sessions_count, {"unit": "count"})
            
        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")
    
    async def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
    
    async def _record_metric(self, metric_type: str, value: float, metadata: Dict[str, Any]):
        """Record a performance metric"""
        metric = PerformanceMetric(
            timestamp=time.time(),
            metric_type=metric_type,
            value=value,
            metadata=metadata
        )
        self.metrics_history.append(metric)
    
    async def _check_performance_thresholds(self):
        """Check performance metrics against thresholds"""
        # Check real-time ratio
        if self.real_time_ratios:
            recent_ratios = list(self.real_time_ratios)[-10:]  # Last 10 ratios
            avg_ratio = sum(recent_ratios) / len(recent_ratios)
            
            if avg_ratio > self.real_time_threshold:
                await self._create_alert(
                    "real_time_ratio", 
                    avg_ratio,
                    f"Real-time ratio exceeded: {avg_ratio:.2f} > {self.real_time_threshold}"
                )
        
        # Check memory usage
        if self.memory_usage_history:
            current_memory = self.memory_usage_history[-1]
            
            if current_memory > self.memory_critical_threshold:
                await self._create_alert(
                    "memory_critical",
                    current_memory,
                    f"Critical memory usage: {current_memory:.1f}MB"
                )
            elif current_memory > self.memory_warning_threshold:
                await self._create_alert(
                    "memory_warning", 
                    current_memory,
                    f"High memory usage: {current_memory:.1f}MB"
                )
    
    async def _check_session_performance(self, stats: ProcessingStats):
        """Check individual session performance"""
        # Check if session maintained real-time performance
        if not validate_real_time_performance(stats.total_processing_time, stats.audio_duration):
            await self._create_alert(
                "session_real_time_violation",
                stats.real_time_ratio,
                f"Session {stats.session_id} violated real-time: {stats.real_time_ratio:.2f}x ratio"
            )
        
        # Check for unusually long model load times
        if stats.model_load_time > 30.0:  # 30 seconds threshold
            await self._create_alert(
                "slow_model_load",
                stats.model_load_time,
                f"Slow model loading: {stats.model_load_time:.1f}s for {stats.model_name}"
            )
    
    async def _create_alert(self, alert_type: str, value: float, message: str):
        """Create and manage performance alerts"""
        # Check if similar alert already exists
        for alert in self.active_alerts:
            if alert.threshold_type == alert_type and not alert.acknowledged:
                return  # Don't create duplicate alerts
        
        alert = PerformanceAlert(alert_type, value, message)
        self.active_alerts.append(alert)
        
        logger.warning(f"Performance Alert: {message}")
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """Add callback for performance alerts"""
        self.alert_callbacks.append(callback)
    
    async def acknowledge_alert(self, alert_timestamp: float):
        """Acknowledge an alert"""
        for alert in self.active_alerts:
            if abs(alert.timestamp - alert_timestamp) < 0.001:  # Close enough
                alert.acknowledged = True
                logger.info(f"Alert acknowledged: {alert.message}")
                break
    
    async def optimize_based_on_metrics(self) -> Dict[str, Any]:
        """Auto-adjust parameters based on performance metrics"""
        optimizations = {}
        
        try:
            # Analyze recent performance
            if len(self.real_time_ratios) >= 10:
                recent_ratios = list(self.real_time_ratios)[-10:]
                avg_ratio = sum(recent_ratios) / len(recent_ratios)
                
                # If consistently over real-time, suggest optimizations
                if avg_ratio > 1.2:  # 20% over real-time threshold
                    optimizations["suggest_smaller_model"] = True
                    optimizations["suggest_reduce_beam_size"] = True
                    optimizations["current_avg_ratio"] = avg_ratio
                
                # If consistently fast, can increase quality
                elif avg_ratio < 0.7:  # 30% under real-time threshold
                    optimizations["suggest_larger_model"] = False  # Keep small for stability
                    optimizations["suggest_increase_best_of"] = True
                    optimizations["current_avg_ratio"] = avg_ratio
            
            # Memory optimization suggestions
            if self.memory_usage_history:
                avg_memory = sum(self.memory_usage_history) / len(self.memory_usage_history)
                
                if avg_memory > self.memory_warning_threshold:
                    optimizations["suggest_reduce_cache_size"] = True
                    optimizations["suggest_cleanup_models"] = True
                    optimizations["current_memory_mb"] = avg_memory
            
            # Model loading optimization
            if self.model_load_times:
                avg_load_time = sum(self.model_load_times) / len(self.model_load_times)
                
                if avg_load_time > 10.0:  # 10 seconds threshold
                    optimizations["suggest_model_preloading"] = True
                    optimizations["current_load_time"] = avg_load_time
            
            if optimizations:
                logger.info(f"Performance optimizations suggested: {list(optimizations.keys())}")
            
            return optimizations
            
        except Exception as e:
            logger.error(f"Auto-optimization failed: {e}")
            return {}
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            report = {
                "timestamp": time.time(),
                "system_info": await self._get_system_info(),
                "current_status": await self._get_current_status(),
                "historical_performance": await self._get_historical_performance(),
                "active_alerts": [
                    {
                        "type": alert.threshold_type,
                        "value": alert.threshold_value,
                        "message": alert.message,
                        "timestamp": alert.timestamp,
                        "acknowledged": alert.acknowledged
                    }
                    for alert in self.active_alerts
                ],
                "optimizations": await self.optimize_based_on_metrics()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate performance report: {e}")
            return {"error": str(e), "timestamp": time.time()}
    
    async def _get_system_info(self) -> Dict[str, Any]:
        """Get current system information"""
        try:
            return {
                "memory_total_mb": psutil.virtual_memory().total / 1024 / 1024,
                "memory_available_mb": psutil.virtual_memory().available / 1024 / 1024,
                "cpu_count": psutil.cpu_count(),
                "cpu_percent": psutil.cpu_percent(interval=None)
            }
        except Exception:
            return {}
    
    async def _get_current_status(self) -> Dict[str, Any]:
        """Get current performance status"""
        current_memory = await self._get_memory_usage()
        
        return {
            "active_sessions": len(self.active_sessions),
            "current_memory_mb": current_memory,
            "memory_usage_percent": (current_memory / PERFORMANCE_CONSTRAINTS["memory_limit_mb"]) * 100,
            "average_real_time_ratio": sum(self.real_time_ratios) / len(self.real_time_ratios) if self.real_time_ratios else 0,
            "total_sessions_processed": len(self.processing_stats)
        }
    
    async def _get_historical_performance(self) -> Dict[str, Any]:
        """Get historical performance statistics"""
        if not self.processing_stats:
            return {}
        
        # Calculate statistics
        ratios = [s.real_time_ratio for s in self.processing_stats]
        processing_times = [s.total_processing_time for s in self.processing_stats]
        memory_usage = [s.memory_usage_mb for s in self.processing_stats]
        
        return {
            "total_sessions": len(self.processing_stats),
            "real_time_ratio": {
                "average": sum(ratios) / len(ratios),
                "min": min(ratios),
                "max": max(ratios),
                "under_threshold_count": sum(1 for r in ratios if r <= self.real_time_threshold)
            },
            "processing_time": {
                "average": sum(processing_times) / len(processing_times),
                "min": min(processing_times),
                "max": max(processing_times)
            },
            "memory_usage": {
                "average": sum(memory_usage) / len(memory_usage),
                "min": min(memory_usage),
                "max": max(memory_usage)
            },
            "language_distribution": self._get_language_distribution()
        }
    
    def _get_language_distribution(self) -> Dict[str, int]:
        """Get distribution of languages processed"""
        distribution = {}
        for stats in self.processing_stats:
            distribution[stats.language] = distribution.get(stats.language, 0) + 1
        return distribution
    
    async def export_metrics(self, file_path: str):
        """Export performance metrics to file"""
        try:
            export_data = {
                "export_timestamp": time.time(),
                "metrics": [asdict(metric) for metric in self.metrics_history],
                "processing_stats": [asdict(stats) for stats in self.processing_stats],
                "performance_report": await self.get_performance_report()
            }
            
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Performance metrics exported to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")

# Global performance monitor instance
_global_performance_monitor: Optional[PerformanceMonitor] = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get or create the global performance monitor instance"""
    global _global_performance_monitor
    if _global_performance_monitor is None:
        _global_performance_monitor = PerformanceMonitor()
    return _global_performance_monitor

async def start_global_monitoring():
    """Start global performance monitoring"""
    monitor = get_performance_monitor()
    await monitor.start_monitoring()

async def stop_global_monitoring():
    """Stop global performance monitoring"""
    monitor = get_performance_monitor()
    await monitor.stop_monitoring()