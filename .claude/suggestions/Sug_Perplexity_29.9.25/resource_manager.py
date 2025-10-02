"""
TranscrevAI Optimized - Resource Manager
Sistema unificado de gerenciamento de recursos com monitoramento de memória e CPU
"""

import asyncio
import gc
import os
import psutil
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Callable, Any, List
import logging

from logging_setup import get_logger, log_resource_usage

logger = get_logger("transcrevai.resource_manager")


class ResourceStatus(Enum):
    """Status levels for system resources"""
    NORMAL = "normal"          # <75% usage - everything OK
    WARNING = "warning"        # 75-85% usage - start optimizations
    EMERGENCY = "emergency"    # 85-90% usage - aggressive cleanup
    CRITICAL = "critical"      # >90% usage - emergency measures


class StreamingMode(Enum):
    """Modes for memory pressure handling"""
    DISABLED = "disabled"      # Normal operation
    ENABLED = "enabled"        # Streaming mode active
    AGGRESSIVE = "aggressive"  # Emergency streaming


@dataclass
class ResourceMetrics:
    """Current system resource metrics"""
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_available_mb: float = 0.0
    cpu_percent: float = 0.0
    cpu_count: int = 0
    disk_usage_percent: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    def __post_init__(self):
        if self.cpu_count == 0:
            self.cpu_count = os.cpu_count() or 4


@dataclass
class MemoryReservation:
    """Memory reservation for components"""
    component: str
    reserved_mb: float
    purpose: str
    timestamp: float = field(default_factory=time.time)
    
    
class ResourceManager:
    """
    Unified Resource Manager - Critical Implementation #5
    
    Monitors system resources and prevents browser crashes through:
    - Memory pressure detection
    - Automatic streaming mode activation
    - Emergency cleanup procedures
    - Resource coordination between components
    """
    
    def __init__(self, 
                 max_memory_mb: int = 2048,
                 warning_threshold: float = 0.75,
                 emergency_threshold: float = 0.85,
                 critical_threshold: float = 0.90,
                 streaming_threshold: float = 0.80):
        
        self.max_memory_mb = max_memory_mb
        self.warning_threshold = warning_threshold
        self.emergency_threshold = emergency_threshold
        self.critical_threshold = critical_threshold
        self.streaming_threshold = streaming_threshold
        
        # Current state
        self.current_status = ResourceStatus.NORMAL
        self.streaming_mode = StreamingMode.DISABLED
        self.last_cleanup_time = 0.0
        
        # Monitoring
        self.monitoring_active = False
        self.monitor_interval = 2.0  # seconds
        self.monitor_task: Optional[asyncio.Task] = None
        self._stop_monitoring = threading.Event()
        
        # Resource tracking
        self.memory_reservations: Dict[str, MemoryReservation] = {}
        self.cleanup_callbacks: List[Callable] = []
        self.warning_callbacks: List[Callable] = []
        
        # Metrics history for trend analysis
        self.metrics_history: List[ResourceMetrics] = []
        self.max_history_size = 60  # Keep 60 measurements (2 minutes at 2s intervals)
        
        # Thread safety
        self._lock = threading.Lock()
        
        logger.info(f"ResourceManager initialized - Max Memory: {max_memory_mb}MB")
        logger.info(f"Thresholds - Warning: {warning_threshold*100:.1f}%, Emergency: {emergency_threshold*100:.1f}%, Critical: {critical_threshold*100:.1f}%")
    
    def get_current_metrics(self) -> ResourceMetrics:
        """Get current system resource metrics"""
        try:
            # Memory metrics
            virtual_memory = psutil.virtual_memory()
            memory_percent = virtual_memory.percent / 100.0
            memory_used_mb = virtual_memory.used / (1024 * 1024)
            memory_available_mb = virtual_memory.available / (1024 * 1024)
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None) / 100.0  # Non-blocking
            cpu_count = psutil.cpu_count()
            
            # Disk metrics
            disk_usage = psutil.disk_usage('/')
            disk_usage_percent = disk_usage.percent / 100.0
            
            return ResourceMetrics(
                memory_percent=memory_percent,
                memory_used_mb=memory_used_mb,
                memory_available_mb=memory_available_mb,
                cpu_percent=cpu_percent,
                cpu_count=cpu_count,
                disk_usage_percent=disk_usage_percent
            )
            
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return ResourceMetrics()  # Return default metrics
    
    def update_status(self, metrics: ResourceMetrics) -> ResourceStatus:
        """Update resource status based on current metrics"""
        old_status = self.current_status
        
        if metrics.memory_percent >= self.critical_threshold:
            self.current_status = ResourceStatus.CRITICAL
        elif metrics.memory_percent >= self.emergency_threshold:
            self.current_status = ResourceStatus.EMERGENCY  
        elif metrics.memory_percent >= self.warning_threshold:
            self.current_status = ResourceStatus.WARNING
        else:
            self.current_status = ResourceStatus.NORMAL
        
        # Update streaming mode
        old_streaming = self.streaming_mode
        if metrics.memory_percent >= self.emergency_threshold:
            self.streaming_mode = StreamingMode.AGGRESSIVE
        elif metrics.memory_percent >= self.streaming_threshold:
            self.streaming_mode = StreamingMode.ENABLED
        else:
            self.streaming_mode = StreamingMode.DISABLED
        
        # Log status changes
        if old_status != self.current_status:
            logger.warning(f"Resource status changed: {old_status.value} -> {self.current_status.value}")
            logger.warning(f"Memory usage: {metrics.memory_percent*100:.1f}%, CPU: {metrics.cpu_percent*100:.1f}%")
            
            # Trigger warning callbacks
            for callback in self.warning_callbacks:
                try:
                    callback(self.current_status, metrics)
                except Exception as e:
                    logger.error(f"Warning callback failed: {e}")
        
        if old_streaming != self.streaming_mode:
            logger.info(f"Streaming mode changed: {old_streaming.value} -> {self.streaming_mode.value}")
        
        return self.current_status
    
    async def perform_cleanup(self, aggressive: bool = False) -> bool:
        """
        Perform memory cleanup based on current pressure level
        
        Args:
            aggressive: If True, perform more aggressive cleanup
            
        Returns:
            True if cleanup was successful
        """
        cleanup_start = time.time()
        
        try:
            logger.info(f"Starting {'aggressive' if aggressive else 'normal'} cleanup")
            
            # Get metrics before cleanup
            before_metrics = self.get_current_metrics()
            
            # Force garbage collection
            gc.collect()
            
            # Call registered cleanup callbacks
            cleanup_success = True
            for callback in self.cleanup_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(aggressive)
                    else:
                        callback(aggressive)
                except Exception as e:
                    logger.error(f"Cleanup callback failed: {e}")
                    cleanup_success = False
            
            # Additional aggressive measures
            if aggressive:
                # Clear metrics history (keep only recent)
                if len(self.metrics_history) > 10:
                    self.metrics_history = self.metrics_history[-10:]
                
                # Force another GC pass
                gc.collect()
                
                # Clear unused memory reservations
                self._cleanup_old_reservations()
            
            # Get metrics after cleanup
            await asyncio.sleep(0.5)  # Let system settle
            after_metrics = self.get_current_metrics()
            
            # Calculate cleanup effectiveness
            memory_freed_mb = before_metrics.memory_used_mb - after_metrics.memory_used_mb
            cleanup_duration = time.time() - cleanup_start
            
            # Log cleanup results
            logger.info(f"Cleanup completed in {cleanup_duration:.2f}s")
            logger.info(f"Memory freed: {memory_freed_mb:.1f}MB")
            logger.info(f"Memory usage: {before_metrics.memory_percent*100:.1f}% -> {after_metrics.memory_percent*100:.1f}%")
            
            # Log performance metrics
            log_resource_usage(
                "cleanup",
                after_metrics.memory_used_mb,
                after_metrics.cpu_percent * 100,
                freed_mb=memory_freed_mb,
                duration=cleanup_duration,
                aggressive=aggressive,
                success=cleanup_success
            )
            
            self.last_cleanup_time = time.time()
            return cleanup_success
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return False
    
    def _cleanup_old_reservations(self) -> None:
        """Remove old memory reservations"""
        current_time = time.time()
        old_reservations = [
            component for component, reservation in self.memory_reservations.items()
            if current_time - reservation.timestamp > 300  # 5 minutes old
        ]
        
        for component in old_reservations:
            del self.memory_reservations[component]
            logger.debug(f"Removed old reservation for {component}")
    
    def reserve_memory(self, component: str, amount_mb: float, purpose: str = "") -> bool:
        """
        Reserve memory for a component
        
        Args:
            component: Component name
            amount_mb: Amount of memory to reserve in MB
            purpose: Description of what the memory is for
            
        Returns:
            True if reservation was successful
        """
        with self._lock:
            current_metrics = self.get_current_metrics()
            total_reserved = sum(r.reserved_mb for r in self.memory_reservations.values())
            
            # Check if reservation would exceed limits
            if (current_metrics.memory_used_mb + total_reserved + amount_mb) > (self.max_memory_mb * 0.9):
                logger.warning(f"Memory reservation denied for {component}: {amount_mb:.1f}MB would exceed limit")
                return False
            
            # Create reservation
            reservation = MemoryReservation(
                component=component,
                reserved_mb=amount_mb,
                purpose=purpose
            )
            
            self.memory_reservations[component] = reservation
            logger.debug(f"Memory reserved for {component}: {amount_mb:.1f}MB ({purpose})")
            
            return True
    
    def release_memory_reservation(self, component: str) -> bool:
        """Release memory reservation for a component"""
        with self._lock:
            if component in self.memory_reservations:
                reservation = self.memory_reservations.pop(component)
                logger.debug(f"Memory reservation released for {component}: {reservation.reserved_mb:.1f}MB")
                return True
            return False
    
    def get_total_reserved_memory(self) -> float:
        """Get total reserved memory in MB"""
        return sum(r.reserved_mb for r in self.memory_reservations.values())
    
    def can_allocate(self, amount_mb: float) -> bool:
        """Check if a memory allocation is safe"""
        current_metrics = self.get_current_metrics()
        total_reserved = self.get_total_reserved_memory()
        
        # Conservative check - ensure we don't exceed warning threshold
        projected_usage = (current_metrics.memory_used_mb + total_reserved + amount_mb) / (current_metrics.memory_available_mb + current_metrics.memory_used_mb)
        
        return projected_usage < self.warning_threshold
    
    def is_memory_pressure_high(self) -> bool:
        """Check if system is under memory pressure"""
        return self.current_status in [ResourceStatus.WARNING, ResourceStatus.EMERGENCY, ResourceStatus.CRITICAL]
    
    def should_use_streaming_mode(self) -> bool:
        """Check if streaming mode should be active"""
        return self.streaming_mode in [StreamingMode.ENABLED, StreamingMode.AGGRESSIVE]
    
    def get_recommended_chunk_size(self, default_chunk_mb: float) -> float:
        """Get recommended chunk size based on memory pressure"""
        if self.streaming_mode == StreamingMode.AGGRESSIVE:
            return default_chunk_mb * 0.25  # 25% of normal
        elif self.streaming_mode == StreamingMode.ENABLED:
            return default_chunk_mb * 0.5   # 50% of normal
        else:
            return default_chunk_mb
    
    def register_cleanup_callback(self, callback: Callable) -> None:
        """Register a cleanup callback"""
        self.cleanup_callbacks.append(callback)
        logger.debug(f"Registered cleanup callback: {callback.__name__}")
    
    def register_warning_callback(self, callback: Callable) -> None:
        """Register a warning callback"""
        self.warning_callbacks.append(callback)
        logger.debug(f"Registered warning callback: {callback.__name__}")
    
    async def start_monitoring(self) -> None:
        """Start resource monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self._stop_monitoring.clear()
        
        logger.info("Starting resource monitoring")
        
        # Start monitoring task
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self) -> None:
        """Stop resource monitoring"""
        if not self.monitoring_active:
            return
        
        logger.info("Stopping resource monitoring")
        
        self.monitoring_active = False
        self._stop_monitoring.set()
        
        if self.monitor_task:
            try:
                await asyncio.wait_for(self.monitor_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Monitor task did not stop gracefully")
                self.monitor_task.cancel()
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        logger.info(f"Resource monitoring started (interval: {self.monitor_interval}s)")
        
        try:
            while self.monitoring_active and not self._stop_monitoring.is_set():
                # Get current metrics
                metrics = self.get_current_metrics()
                
                # Update history
                self.metrics_history.append(metrics)
                if len(self.metrics_history) > self.max_history_size:
                    self.metrics_history.pop(0)
                
                # Update status
                old_status = self.current_status
                self.update_status(metrics)
                
                # Log resource usage periodically
                log_resource_usage(
                    "system",
                    metrics.memory_used_mb,
                    metrics.cpu_percent * 100,
                    memory_percent=metrics.memory_percent * 100,
                    status=self.current_status.value,
                    streaming_mode=self.streaming_mode.value,
                    reserved_mb=self.get_total_reserved_memory()
                )
                
                # Automatic cleanup based on status
                if self.current_status == ResourceStatus.CRITICAL:
                    # Emergency cleanup
                    await self.perform_cleanup(aggressive=True)
                elif self.current_status == ResourceStatus.EMERGENCY:
                    # Check if enough time has passed since last cleanup
                    if time.time() - self.last_cleanup_time > 30:  # 30 seconds
                        await self.perform_cleanup(aggressive=True)
                elif self.current_status == ResourceStatus.WARNING:
                    # Gentle cleanup if needed
                    if time.time() - self.last_cleanup_time > 120:  # 2 minutes
                        await self.perform_cleanup(aggressive=False)
                
                # Wait for next check
                await asyncio.sleep(self.monitor_interval)
                
        except asyncio.CancelledError:
            logger.info("Resource monitoring cancelled")
        except Exception as e:
            logger.error(f"Resource monitoring error: {e}")
        finally:
            self.monitoring_active = False
            logger.info("Resource monitoring stopped")
    
    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report"""
        metrics = self.get_current_metrics()
        
        return {
            "timestamp": time.time(),
            "status": self.current_status.value,
            "streaming_mode": self.streaming_mode.value,
            "memory": {
                "used_mb": metrics.memory_used_mb,
                "available_mb": metrics.memory_available_mb,
                "usage_percent": metrics.memory_percent * 100,
                "reserved_mb": self.get_total_reserved_memory(),
                "thresholds": {
                    "warning": self.warning_threshold * 100,
                    "emergency": self.emergency_threshold * 100,
                    "critical": self.critical_threshold * 100,
                }
            },
            "cpu": {
                "usage_percent": metrics.cpu_percent * 100,
                "cores": metrics.cpu_count,
            },
            "disk": {
                "usage_percent": metrics.disk_usage_percent * 100,
            },
            "monitoring": {
                "active": self.monitoring_active,
                "interval": self.monitor_interval,
                "history_size": len(self.metrics_history),
            },
            "reservations": [
                {
                    "component": comp,
                    "amount_mb": res.reserved_mb,
                    "purpose": res.purpose,
                    "age_seconds": time.time() - res.timestamp
                }
                for comp, res in self.memory_reservations.items()
            ]
        }


# Global instance for easy access
_global_resource_manager: Optional[ResourceManager] = None


def get_resource_manager() -> ResourceManager:
    """Get global resource manager instance"""
    global _global_resource_manager
    
    if _global_resource_manager is None:
        # Import config values
        try:
            from config import CONFIG
            hardware_config = CONFIG["hardware"]
            
            _global_resource_manager = ResourceManager(
                max_memory_mb=hardware_config["max_memory_mb"],
                warning_threshold=hardware_config["memory_thresholds"]["warning"],
                emergency_threshold=hardware_config["memory_thresholds"]["emergency"],
                critical_threshold=hardware_config["memory_thresholds"]["critical"],
                streaming_threshold=hardware_config["memory_thresholds"]["streaming"]
            )
        except ImportError:
            # Fallback configuration
            logger.warning("Config not available, using default ResourceManager settings")
            _global_resource_manager = ResourceManager()
    
    return _global_resource_manager


# Convenience functions
async def check_memory_pressure() -> ResourceStatus:
    """Check current memory pressure level"""
    rm = get_resource_manager()
    metrics = rm.get_current_metrics()
    return rm.update_status(metrics)


def can_allocate_memory(amount_mb: float) -> bool:
    """Check if memory allocation is safe"""
    rm = get_resource_manager()
    return rm.can_allocate(amount_mb)


async def request_cleanup(aggressive: bool = False) -> bool:
    """Request immediate cleanup"""
    rm = get_resource_manager()
    return await rm.perform_cleanup(aggressive)