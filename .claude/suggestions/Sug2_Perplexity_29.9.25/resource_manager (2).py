"""
TranscrevAI Optimized - Advanced Resource Manager
Sistema crítico de gerenciamento de recursos com prevenção de crashes
"""

import asyncio
import gc
import os
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set
import psutil
import logging

logger = logging.getLogger("transcrevai.resource_manager")


class ResourceStatus(Enum):
    """Resource status levels"""
    NORMAL = "normal"           # <70% usage
    WARNING = "warning"         # 70-80% usage
    CRITICAL = "critical"       # 80-90% usage  
    EMERGENCY = "emergency"     # >90% usage


class MemoryPressureLevel(Enum):
    """Memory pressure intensity levels"""
    LOW = "low"                 # <70%
    MODERATE = "moderate"       # 70-80%
    HIGH = "high"              # 80-90%
    EXTREME = "extreme"         # >90%


@dataclass
class ResourceMetrics:
    """Resource usage metrics snapshot"""
    timestamp: float
    cpu_percent: float
    cpu_count: int
    memory_total_mb: float
    memory_used_mb: float
    memory_available_mb: float
    memory_percent: float
    swap_total_mb: float
    swap_used_mb: float
    process_memory_mb: float
    process_cpu_percent: float
    thread_count: int
    open_files: int
    status: ResourceStatus
    pressure_level: MemoryPressureLevel


@dataclass  
class ResourceThresholds:
    """Resource usage thresholds"""
    memory_warning: float = 75.0      # 75%
    memory_critical: float = 85.0     # 85%
    memory_emergency: float = 95.0    # 95%
    cpu_warning: float = 70.0         # 70%
    cpu_critical: float = 85.0        # 85%
    swap_warning: float = 50.0        # 50%
    swap_critical: float = 80.0       # 80%


class ResourceAlert:
    """Resource alert notification"""
    
    def __init__(self, alert_type: str, level: ResourceStatus, message: str, metrics: ResourceMetrics):
        self.alert_type = alert_type
        self.level = level
        self.message = message
        self.metrics = metrics
        self.timestamp = time.time()


class MemoryReservationManager:
    """Manage memory reservations to prevent over-allocation"""
    
    def __init__(self):
        self.reservations: Dict[str, float] = {}  # reservation_id -> MB
        self.descriptions: Dict[str, str] = {}    # reservation_id -> description
        self._lock = threading.RLock()
    
    def reserve(self, reservation_id: str, memory_mb: float, description: str = "") -> bool:
        """
        Reserve memory for an operation
        
        Args:
            reservation_id: Unique identifier for reservation
            memory_mb: Memory to reserve in MB
            description: Description of what needs memory
            
        Returns:
            bool: True if reservation successful
        """
        with self._lock:
            # Check if we have enough memory
            current_usage = psutil.virtual_memory().percent
            current_reserved = sum(self.reservations.values())
            system_memory_mb = psutil.virtual_memory().total / (1024 * 1024)
            
            # Calculate projected usage
            projected_usage = ((current_reserved + memory_mb) / system_memory_mb) * 100 + current_usage
            
            # Conservative threshold - don't allow reservations that would push us over 80%
            if projected_usage > 80.0:
                logger.warning(f"Memory reservation denied: {reservation_id} ({memory_mb}MB) "
                             f"would push usage to {projected_usage:.1f}%")
                return False
            
            # Store reservation
            self.reservations[reservation_id] = memory_mb
            self.descriptions[reservation_id] = description
            
            logger.debug(f"Memory reserved: {reservation_id} ({memory_mb}MB) - {description}")
            return True
    
    def release(self, reservation_id: str) -> bool:
        """Release a memory reservation"""
        with self._lock:
            if reservation_id in self.reservations:
                memory_mb = self.reservations.pop(reservation_id)
                self.descriptions.pop(reservation_id, None)
                logger.debug(f"Memory released: {reservation_id} ({memory_mb}MB)")
                return True
            return False
    
    def get_total_reserved(self) -> float:
        """Get total reserved memory in MB"""
        with self._lock:
            return sum(self.reservations.values())
    
    def get_reservations(self) -> Dict[str, Dict[str, Any]]:
        """Get all current reservations"""
        with self._lock:
            return {
                res_id: {
                    "memory_mb": memory_mb,
                    "description": self.descriptions.get(res_id, "")
                }
                for res_id, memory_mb in self.reservations.items()
            }
    
    def cleanup_expired(self, max_age_minutes: float = 30) -> int:
        """Clean up old reservations (emergency fallback)"""
        with self._lock:
            # This is a simplified cleanup - in production you'd track timestamps
            old_count = len(self.reservations)
            
            # Clear reservations that look like they might be stuck
            temp_reservations = {
                k: v for k, v in self.reservations.items()
                if not (k.startswith("temp_") or k.startswith("old_"))
            }
            
            self.reservations = temp_reservations
            self.descriptions = {
                k: v for k, v in self.descriptions.items()
                if k in temp_reservations
            }
            
            cleaned = old_count - len(self.reservations)
            if cleaned > 0:
                logger.info(f"Cleaned up {cleaned} expired memory reservations")
            
            return cleaned


class ResourceMonitor:
    """Comprehensive resource monitoring system"""
    
    def __init__(self, thresholds: Optional[ResourceThresholds] = None):
        self.thresholds = thresholds or ResourceThresholds()
        self.memory_reservations = MemoryReservationManager()
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.monitoring_interval = 5.0  # seconds
        
        # Metrics storage
        self.current_metrics: Optional[ResourceMetrics] = None
        self.metrics_history: List[ResourceMetrics] = []
        self.max_history_size = 100
        
        # Alert system
        self.alert_callbacks: List[Callable[[ResourceAlert], None]] = []
        self.last_alert_time: Dict[str, float] = {}
        self.alert_cooldown = 30.0  # seconds
        
        # Process reference
        try:
            self.process = psutil.Process()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            self.process = None
            logger.warning("Unable to get process reference for monitoring")
        
        # Thread safety
        self._lock = threading.RLock()
    
    async def start_monitoring(self) -> bool:
        """Start resource monitoring"""
        if self.monitoring_active:
            logger.warning("Resource monitoring already active")
            return False
        
        try:
            self.monitoring_active = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Resource monitoring started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start resource monitoring: {e}")
            self.monitoring_active = False
            return False
    
    async def stop_monitoring(self) -> bool:
        """Stop resource monitoring"""
        if not self.monitoring_active:
            return True
        
        try:
            self.monitoring_active = False
            
            if self.monitoring_task and not self.monitoring_task.done():
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            self.monitoring_task = None
            logger.info("Resource monitoring stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop resource monitoring: {e}")
            return False
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        logger.info("Resource monitoring loop started")
        
        try:
            while self.monitoring_active:
                try:
                    # Take metrics snapshot
                    metrics = self._take_metrics_snapshot()
                    
                    if metrics:
                        # Update current metrics
                        with self._lock:
                            self.current_metrics = metrics
                            self.metrics_history.append(metrics)
                            
                            # Limit history size
                            if len(self.metrics_history) > self.max_history_size:
                                self.metrics_history = self.metrics_history[-self.max_history_size:]
                        
                        # Check for alerts
                        await self._check_and_send_alerts(metrics)
                    
                    # Wait for next monitoring cycle
                    await asyncio.sleep(self.monitoring_interval)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    await asyncio.sleep(1)  # Brief pause on error
                    
        except Exception as e:
            logger.error(f"Monitoring loop failed: {e}")
        finally:
            logger.info("Resource monitoring loop stopped")
    
    def _take_metrics_snapshot(self) -> Optional[ResourceMetrics]:
        """Take a comprehensive metrics snapshot"""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Process metrics
            process_memory_mb = 0.0
            process_cpu_percent = 0.0
            thread_count = 0
            open_files = 0
            
            if self.process:
                try:
                    process_info = self.process.memory_info()
                    process_memory_mb = process_info.rss / (1024 * 1024)
                    process_cpu_percent = self.process.cpu_percent()
                    thread_count = self.process.num_threads()
                    
                    # Open files (may fail on some systems)
                    try:
                        open_files = len(self.process.open_files())
                    except (psutil.AccessDenied, AttributeError):
                        pass
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # Determine status and pressure level
            status = self._determine_status(memory.percent, cpu_percent, swap.percent)
            pressure_level = self._determine_pressure_level(memory.percent)
            
            return ResourceMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                cpu_count=cpu_count,
                memory_total_mb=memory.total / (1024 * 1024),
                memory_used_mb=memory.used / (1024 * 1024),
                memory_available_mb=memory.available / (1024 * 1024),
                memory_percent=memory.percent,
                swap_total_mb=swap.total / (1024 * 1024),
                swap_used_mb=swap.used / (1024 * 1024),
                process_memory_mb=process_memory_mb,
                process_cpu_percent=process_cpu_percent,
                thread_count=thread_count,
                open_files=open_files,
                status=status,
                pressure_level=pressure_level
            )
            
        except Exception as e:
            logger.error(f"Failed to take metrics snapshot: {e}")
            return None
    
    def _determine_status(self, memory_percent: float, cpu_percent: float, swap_percent: float) -> ResourceStatus:
        """Determine overall resource status"""
        # Memory is most critical for our use case
        if memory_percent >= self.thresholds.memory_emergency:
            return ResourceStatus.EMERGENCY
        elif memory_percent >= self.thresholds.memory_critical:
            return ResourceStatus.CRITICAL
        elif memory_percent >= self.thresholds.memory_warning:
            return ResourceStatus.WARNING
        
        # Check CPU and swap as secondary factors
        if cpu_percent >= self.thresholds.cpu_critical or swap_percent >= self.thresholds.swap_critical:
            return ResourceStatus.CRITICAL
        elif cpu_percent >= self.thresholds.cpu_warning or swap_percent >= self.thresholds.swap_warning:
            return ResourceStatus.WARNING
        
        return ResourceStatus.NORMAL
    
    def _determine_pressure_level(self, memory_percent: float) -> MemoryPressureLevel:
        """Determine memory pressure level"""
        if memory_percent >= 90.0:
            return MemoryPressureLevel.EXTREME
        elif memory_percent >= 80.0:
            return MemoryPressureLevel.HIGH
        elif memory_percent >= 70.0:
            return MemoryPressureLevel.MODERATE
        else:
            return MemoryPressureLevel.LOW
    
    async def _check_and_send_alerts(self, metrics: ResourceMetrics):
        """Check for alert conditions and send notifications"""
        current_time = time.time()
        alerts_to_send = []
        
        # Memory alerts
        if metrics.memory_percent >= self.thresholds.memory_emergency:
            alert_key = "memory_emergency"
            if self._should_send_alert(alert_key, current_time):
                alert = ResourceAlert(
                    "memory", ResourceStatus.EMERGENCY,
                    f"EMERGENCY: Memory usage at {metrics.memory_percent:.1f}% "
                    f"({metrics.memory_used_mb:.0f}MB/{metrics.memory_total_mb:.0f}MB)",
                    metrics
                )
                alerts_to_send.append(alert)
                
        elif metrics.memory_percent >= self.thresholds.memory_critical:
            alert_key = "memory_critical"
            if self._should_send_alert(alert_key, current_time):
                alert = ResourceAlert(
                    "memory", ResourceStatus.CRITICAL,
                    f"CRITICAL: Memory usage at {metrics.memory_percent:.1f}%",
                    metrics
                )
                alerts_to_send.append(alert)
        
        # CPU alerts
        if metrics.cpu_percent >= self.thresholds.cpu_critical:
            alert_key = "cpu_critical"
            if self._should_send_alert(alert_key, current_time):
                alert = ResourceAlert(
                    "cpu", ResourceStatus.CRITICAL,
                    f"CRITICAL: CPU usage at {metrics.cpu_percent:.1f}%",
                    metrics
                )
                alerts_to_send.append(alert)
        
        # Swap alerts
        if metrics.swap_total_mb > 0 and (metrics.swap_used_mb / metrics.swap_total_mb * 100) >= self.thresholds.swap_critical:
            alert_key = "swap_critical"
            swap_percent = (metrics.swap_used_mb / metrics.swap_total_mb * 100)
            if self._should_send_alert(alert_key, current_time):
                alert = ResourceAlert(
                    "swap", ResourceStatus.CRITICAL,
                    f"CRITICAL: Swap usage at {swap_percent:.1f}%",
                    metrics
                )
                alerts_to_send.append(alert)
        
        # Send alerts
        for alert in alerts_to_send:
            await self._send_alert(alert)
    
    def _should_send_alert(self, alert_key: str, current_time: float) -> bool:
        """Check if alert should be sent based on cooldown"""
        last_alert = self.last_alert_time.get(alert_key, 0)
        return (current_time - last_alert) >= self.alert_cooldown
    
    async def _send_alert(self, alert: ResourceAlert):
        """Send alert to all registered callbacks"""
        self.last_alert_time[alert.alert_type + "_" + alert.level.value] = alert.timestamp
        
        # Log alert
        log_level = logging.CRITICAL if alert.level == ResourceStatus.EMERGENCY else logging.ERROR
        logger.log(log_level, f"RESOURCE ALERT: {alert.message}")
        
        # Send to callbacks
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def add_alert_callback(self, callback: Callable[[ResourceAlert], None]):
        """Add alert callback"""
        self.alert_callbacks.append(callback)
    
    def get_current_metrics(self) -> Optional[ResourceMetrics]:
        """Get current resource metrics"""
        with self._lock:
            return self.current_metrics
    
    def get_metrics_history(self, last_n: int = 10) -> List[ResourceMetrics]:
        """Get recent metrics history"""
        with self._lock:
            return self.metrics_history[-last_n:] if self.metrics_history else []
    
    def is_memory_pressure_high(self) -> bool:
        """Check if memory pressure is high"""
        metrics = self.get_current_metrics()
        if metrics:
            return metrics.pressure_level in [MemoryPressureLevel.HIGH, MemoryPressureLevel.EXTREME]
        
        # Fallback check
        try:
            return psutil.virtual_memory().percent > 80.0
        except:
            return False
    
    def can_allocate(self, memory_mb: float) -> bool:
        """Check if we can safely allocate memory"""
        try:
            current_usage = psutil.virtual_memory().percent
            reserved = self.memory_reservations.get_total_reserved()
            total_memory_mb = psutil.virtual_memory().total / (1024 * 1024)
            
            # Calculate projected usage
            projected_usage = ((reserved + memory_mb) / total_memory_mb) * 100 + current_usage
            
            # Conservative threshold
            return projected_usage <= 75.0  # Don't exceed 75%
            
        except Exception as e:
            logger.error(f"Error checking memory allocation: {e}")
            return False
    
    def reserve_memory(self, reservation_id: str, memory_mb: float, description: str = "") -> bool:
        """Reserve memory for an operation"""
        return self.memory_reservations.reserve(reservation_id, memory_mb, description)
    
    def release_memory_reservation(self, reservation_id: str) -> bool:
        """Release a memory reservation"""
        return self.memory_reservations.release(reservation_id)
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Get comprehensive memory status"""
        try:
            memory = psutil.virtual_memory()
            reserved = self.memory_reservations.get_total_reserved()
            
            return {
                "total_mb": memory.total / (1024 * 1024),
                "used_mb": memory.used / (1024 * 1024),
                "available_mb": memory.available / (1024 * 1024),
                "usage_percent": memory.percent,
                "reserved_mb": reserved,
                "pressure_level": self._determine_pressure_level(memory.percent).value,
                "reservations_count": len(self.memory_reservations.reservations),
                "can_allocate_100mb": self.can_allocate(100.0),
                "can_allocate_500mb": self.can_allocate(500.0),
            }
        except Exception as e:
            logger.error(f"Error getting memory status: {e}")
            return {}
    
    async def perform_cleanup(self, aggressive: bool = False) -> Dict[str, Any]:
        """Perform resource cleanup"""
        cleanup_start = time.time()
        
        try:
            actions_performed = []
            
            # Garbage collection
            collected_before = gc.collect()
            actions_performed.append(f"gc_collected_{collected_before}")
            
            if aggressive:
                # Multiple GC passes
                for i in range(3):
                    collected = gc.collect()
                    await asyncio.sleep(0.01)  # Brief pause
                actions_performed.append("aggressive_gc")
                
                # Clean up expired reservations
                cleaned = self.memory_reservations.cleanup_expired()
                if cleaned > 0:
                    actions_performed.append(f"cleaned_reservations_{cleaned}")
            
            cleanup_duration = time.time() - cleanup_start
            
            return {
                "success": True,
                "duration": cleanup_duration,
                "actions": actions_performed,
                "aggressive": aggressive
            }
            
        except Exception as e:
            logger.error(f"Resource cleanup failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "duration": time.time() - cleanup_start
            }


# Global resource manager instance
_global_resource_manager: Optional[ResourceMonitor] = None


def get_resource_manager() -> ResourceMonitor:
    """Get global resource manager instance"""
    global _global_resource_manager
    
    if _global_resource_manager is None:
        _global_resource_manager = ResourceMonitor()
        logger.info("Global resource manager created")
    
    return _global_resource_manager


async def initialize_resource_manager() -> ResourceMonitor:
    """Initialize and start resource manager"""
    manager = get_resource_manager()
    await manager.start_monitoring()
    return manager


# Convenience functions
def get_current_memory_usage() -> float:
    """Get current memory usage percentage"""
    try:
        return psutil.virtual_memory().percent
    except:
        return 0.0


def get_current_cpu_usage() -> float:
    """Get current CPU usage percentage"""
    try:
        return psutil.cpu_percent(interval=0.1)
    except:
        return 0.0


def is_system_under_pressure() -> bool:
    """Check if system is under resource pressure"""
    try:
        memory_percent = psutil.virtual_memory().percent
        return memory_percent > 75.0
    except:
        return False


# Export main functions and classes
__all__ = [
    "ResourceMonitor", 
    "ResourceStatus", 
    "MemoryPressureLevel",
    "ResourceMetrics",
    "ResourceAlert",
    "get_resource_manager",
    "initialize_resource_manager",
    "get_current_memory_usage",
    "get_current_cpu_usage", 
    "is_system_under_pressure"
]