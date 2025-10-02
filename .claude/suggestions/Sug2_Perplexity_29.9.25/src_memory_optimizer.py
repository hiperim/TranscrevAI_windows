"""
TranscrevAI Optimized - Memory Optimizer Module  
Sistema avançado de otimização de memória com cleanup adaptativo
"""

import asyncio
import gc
import os
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import psutil

# Import our optimized modules
from logging_setup import get_logger, log_performance, log_resource_usage
from resource_manager import get_resource_manager, ResourceStatus
from config import CONFIG

logger = get_logger("transcrevai.memory_optimizer")


class CleanupLevel(Enum):
    """Memory cleanup intensity levels"""
    LIGHT = "light"          # Basic cleanup
    MODERATE = "moderate"    # Standard cleanup  
    AGGRESSIVE = "aggressive"  # Deep cleanup
    EMERGENCY = "emergency"   # Critical cleanup


@dataclass
class MemorySnapshot:
    """Memory usage snapshot"""
    timestamp: float
    total_mb: float
    used_mb: float
    available_mb: float
    usage_percent: float
    process_memory_mb: float
    reserved_mb: float
    threat_level: str


class AdaptiveMemoryCleanup:
    """
    Adaptive memory cleanup system that adjusts strategy based on memory pressure
    """
    
    def __init__(self):
        self.resource_manager = get_resource_manager()
        
        # Cleanup history and patterns
        self.cleanup_history: List[Dict[str, Any]] = []
        self.effectiveness_scores: Dict[str, float] = {}
        
        # Cleanup strategies
        self.cleanup_strategies = {
            CleanupLevel.LIGHT: self._light_cleanup,
            CleanupLevel.MODERATE: self._moderate_cleanup,
            CleanupLevel.AGGRESSIVE: self._aggressive_cleanup,
            CleanupLevel.EMERGENCY: self._emergency_cleanup
        }
        
        # Thresholds for adaptive cleanup
        self.thresholds = {
            "light": 70.0,      # 70% memory usage
            "moderate": 80.0,   # 80% memory usage
            "aggressive": 90.0, # 90% memory usage  
            "emergency": 95.0   # 95% memory usage
        }
        
        logger.info("AdaptiveMemoryCleanup initialized")
    
    async def perform_adaptive_cleanup(self) -> Dict[str, Any]:
        """
        Perform adaptive memory cleanup based on current memory pressure
        
        Returns:
            Dict with cleanup results and metrics
        """
        cleanup_start = time.time()
        before_snapshot = self._take_memory_snapshot()
        
        try:
            # Determine cleanup level based on memory usage
            cleanup_level = self._determine_cleanup_level(before_snapshot.usage_percent)
            
            logger.info(f"Starting {cleanup_level.value} memory cleanup (usage: {before_snapshot.usage_percent:.1f}%)")
            
            # Execute cleanup strategy
            cleanup_results = await self.cleanup_strategies[cleanup_level]()
            
            # Take after snapshot
            after_snapshot = self._take_memory_snapshot()
            
            # Calculate effectiveness
            memory_freed = before_snapshot.used_mb - after_snapshot.used_mb
            effectiveness = memory_freed / max(before_snapshot.used_mb, 1.0)
            
            # Record cleanup in history
            cleanup_record = {
                "timestamp": cleanup_start,
                "level": cleanup_level.value,
                "before_usage": before_snapshot.usage_percent,
                "after_usage": after_snapshot.usage_percent,
                "memory_freed_mb": memory_freed,
                "effectiveness": effectiveness,
                "duration": time.time() - cleanup_start,
                "results": cleanup_results
            }
            
            self.cleanup_history.append(cleanup_record)
            self._update_effectiveness_scores(cleanup_level.value, effectiveness)
            
            # Log results
            log_performance(
                f"Memory cleanup completed ({cleanup_level.value})",
                duration=cleanup_record["duration"],
                memory_freed_mb=memory_freed,
                effectiveness=effectiveness,
                before_usage=before_snapshot.usage_percent,
                after_usage=after_snapshot.usage_percent
            )
            
            logger.info(f"Memory cleanup freed {memory_freed:.1f}MB "
                       f"({before_snapshot.usage_percent:.1f}% → {after_snapshot.usage_percent:.1f}%)")
            
            return {
                "success": True,
                "level": cleanup_level.value,
                "memory_freed_mb": memory_freed,
                "effectiveness": effectiveness,
                "before_snapshot": before_snapshot,
                "after_snapshot": after_snapshot,
                "duration": cleanup_record["duration"]
            }
            
        except Exception as e:
            logger.error(f"Adaptive cleanup failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "level": "failed",
                "memory_freed_mb": 0.0,
                "effectiveness": 0.0
            }
    
    def _determine_cleanup_level(self, usage_percent: float) -> CleanupLevel:
        """Determine appropriate cleanup level based on memory usage"""
        if usage_percent >= self.thresholds["emergency"]:
            return CleanupLevel.EMERGENCY
        elif usage_percent >= self.thresholds["aggressive"]:
            return CleanupLevel.AGGRESSIVE
        elif usage_percent >= self.thresholds["moderate"]:
            return CleanupLevel.MODERATE
        else:
            return CleanupLevel.LIGHT
    
    def _take_memory_snapshot(self) -> MemorySnapshot:
        """Take current memory usage snapshot"""
        try:
            # System memory
            system_memory = psutil.virtual_memory()
            
            # Process memory
            process = psutil.Process()
            process_memory = process.memory_info()
            
            # Reserved memory from resource manager
            reserved_mb = sum(self.resource_manager.memory_reservations.values())
            
            # Determine threat level
            usage_percent = system_memory.percent
            if usage_percent >= 95:
                threat_level = "CRITICAL"
            elif usage_percent >= 85:
                threat_level = "WARNING" 
            elif usage_percent >= 75:
                threat_level = "CAUTION"
            else:
                threat_level = "NORMAL"
            
            return MemorySnapshot(
                timestamp=time.time(),
                total_mb=system_memory.total / (1024 * 1024),
                used_mb=system_memory.used / (1024 * 1024),
                available_mb=system_memory.available / (1024 * 1024),
                usage_percent=usage_percent,
                process_memory_mb=process_memory.rss / (1024 * 1024),
                reserved_mb=reserved_mb,
                threat_level=threat_level
            )
            
        except Exception as e:
            logger.warning(f"Failed to take memory snapshot: {e}")
            return MemorySnapshot(
                timestamp=time.time(),
                total_mb=0.0, used_mb=0.0, available_mb=0.0,
                usage_percent=0.0, process_memory_mb=0.0,
                reserved_mb=0.0, threat_level="UNKNOWN"
            )
    
    async def _light_cleanup(self) -> Dict[str, Any]:
        """Light cleanup: basic garbage collection"""
        try:
            # Force garbage collection
            collected_objects = gc.collect()
            
            # Clear small caches if available
            self._clear_small_caches()
            
            return {
                "method": "light",
                "actions": ["garbage_collection", "small_cache_cleanup"],
                "objects_collected": collected_objects
            }
            
        except Exception as e:
            logger.error(f"Light cleanup failed: {e}")
            return {"method": "light", "error": str(e)}
    
    async def _moderate_cleanup(self) -> Dict[str, Any]:
        """Moderate cleanup: garbage collection + cache clearing"""
        try:
            actions_performed = []
            objects_collected = 0
            
            # Multiple garbage collection passes
            for i in range(3):
                collected = gc.collect()
                objects_collected += collected
                if i < 2:  # Brief pause between passes
                    await asyncio.sleep(0.01)
            
            actions_performed.append("enhanced_garbage_collection")
            
            # Clear various caches
            self._clear_small_caches()
            self._clear_medium_caches()
            actions_performed.append("cache_cleanup")
            
            # Release unused memory reservations
            released_reservations = self._release_unused_reservations()
            if released_reservations > 0:
                actions_performed.append("reservation_cleanup")
            
            return {
                "method": "moderate",
                "actions": actions_performed,
                "objects_collected": objects_collected,
                "released_reservations": released_reservations
            }
            
        except Exception as e:
            logger.error(f"Moderate cleanup failed: {e}")
            return {"method": "moderate", "error": str(e)}
    
    async def _aggressive_cleanup(self) -> Dict[str, Any]:
        """Aggressive cleanup: comprehensive memory cleanup"""
        try:
            actions_performed = []
            objects_collected = 0
            
            # Enhanced garbage collection with all generations
            for generation in range(3):
                collected = gc.collect(generation)
                objects_collected += collected
                await asyncio.sleep(0.01)  # Browser-safe pause
            
            actions_performed.append("full_garbage_collection")
            
            # Clear all available caches
            self._clear_small_caches()
            self._clear_medium_caches()
            self._clear_large_caches()
            actions_performed.append("comprehensive_cache_cleanup")
            
            # Release all unused reservations
            released_reservations = self._release_unused_reservations(aggressive=True)
            if released_reservations > 0:
                actions_performed.append("aggressive_reservation_cleanup")
            
            # Clear model cache partially if memory pressure is high
            if self.resource_manager.is_memory_pressure_high():
                try:
                    from model_cache import get_model_cache
                    cache = get_model_cache()
                    cache.cleanup_expired_models()
                    actions_performed.append("model_cache_cleanup")
                except Exception as e:
                    logger.warning(f"Model cache cleanup failed: {e}")
            
            # Force Python to release memory back to OS (Linux/Mac)
            try:
                import ctypes
                if hasattr(ctypes, 'CDLL'):
                    libc = ctypes.CDLL("libc.so.6")
                    libc.malloc_trim(0)
                    actions_performed.append("malloc_trim")
            except:
                pass  # Not critical, ignore on Windows or if unavailable
            
            return {
                "method": "aggressive",
                "actions": actions_performed,
                "objects_collected": objects_collected,
                "released_reservations": released_reservations
            }
            
        except Exception as e:
            logger.error(f"Aggressive cleanup failed: {e}")
            return {"method": "aggressive", "error": str(e)}
    
    async def _emergency_cleanup(self) -> Dict[str, Any]:
        """Emergency cleanup: maximum memory recovery"""
        try:
            logger.warning("Performing emergency memory cleanup")
            actions_performed = []
            objects_collected = 0
            
            # Maximum garbage collection
            for i in range(5):  # Multiple passes
                for generation in range(3):
                    collected = gc.collect(generation)
                    objects_collected += collected
                await asyncio.sleep(0.01)  # Browser-safe
            
            actions_performed.append("emergency_garbage_collection")
            
            # Clear ALL caches aggressively
            self._clear_small_caches()
            self._clear_medium_caches()
            self._clear_large_caches()
            self._emergency_cache_clear()
            actions_performed.append("emergency_cache_cleanup")
            
            # Force release all reservations
            self.resource_manager.memory_reservations.clear()
            actions_performed.append("force_reservation_clear")
            
            # Emergency model cache cleanup
            try:
                from model_cache import get_model_cache
                cache = get_model_cache()
                await cache.emergency_cleanup()
                actions_performed.append("emergency_model_cache_cleanup")
            except Exception as e:
                logger.warning(f"Emergency model cache cleanup failed: {e}")
            
            # Try to force memory release at OS level
            try:
                import ctypes
                if hasattr(ctypes, 'CDLL'):
                    libc = ctypes.CDLL("libc.so.6")
                    libc.malloc_trim(0)
                    actions_performed.append("force_malloc_trim")
            except:
                pass
            
            # Final garbage collection
            final_collected = gc.collect()
            objects_collected += final_collected
            
            return {
                "method": "emergency",
                "actions": actions_performed,
                "objects_collected": objects_collected,
                "warning": "Emergency cleanup performed - some cached data may be lost"
            }
            
        except Exception as e:
            logger.error(f"Emergency cleanup failed: {e}")
            return {"method": "emergency", "error": str(e)}
    
    def _clear_small_caches(self):
        """Clear small internal caches"""
        try:
            # Clear logging formatters cache
            import logging
            logging._handlers.clear()
            logging._handlerList[:] = []
        except:
            pass
    
    def _clear_medium_caches(self):
        """Clear medium-sized caches"""
        try:
            # Clear import caches
            import sys
            if hasattr(sys, '_clear_type_cache'):
                sys._clear_type_cache()
            
            # Clear regex cache
            import re
            re.purge()
        except:
            pass
    
    def _clear_large_caches(self):
        """Clear large caches"""
        try:
            # Clear module caches if safe to do so
            import sys
            modules_to_clear = []
            for module_name, module in sys.modules.items():
                if hasattr(module, '__dict__') and hasattr(module, '_cache'):
                    try:
                        module._cache.clear()
                    except:
                        pass
        except:
            pass
    
    def _emergency_cache_clear(self):
        """Emergency cache clearing for critical situations"""
        try:
            # Clear function caches
            import functools
            for obj in gc.get_objects():
                if hasattr(obj, 'cache_clear') and callable(getattr(obj, 'cache_clear')):
                    try:
                        obj.cache_clear()
                    except:
                        pass
        except:
            pass
    
    def _release_unused_reservations(self, aggressive: bool = False) -> int:
        """Release unused memory reservations"""
        try:
            reservations_before = len(self.resource_manager.memory_reservations)
            
            if aggressive:
                # In aggressive mode, clear all reservations older than 5 minutes
                current_time = time.time()
                to_remove = []
                
                for reservation_id in self.resource_manager.memory_reservations:
                    # Check if reservation is old (simplified check)
                    if reservation_id.startswith("temp_") or reservation_id.startswith("old_"):
                        to_remove.append(reservation_id)
                
                for reservation_id in to_remove:
                    self.resource_manager.release_memory_reservation(reservation_id)
                
            else:
                # Normal mode: just clear expired reservations
                # This would need actual timestamp tracking in ResourceManager
                pass
            
            reservations_after = len(self.resource_manager.memory_reservations)
            return reservations_before - reservations_after
            
        except Exception as e:
            logger.warning(f"Failed to release unused reservations: {e}")
            return 0
    
    def _update_effectiveness_scores(self, cleanup_method: str, effectiveness: float):
        """Update effectiveness scores for cleanup methods"""
        if cleanup_method not in self.effectiveness_scores:
            self.effectiveness_scores[cleanup_method] = effectiveness
        else:
            # Moving average
            current_score = self.effectiveness_scores[cleanup_method]
            self.effectiveness_scores[cleanup_method] = (current_score * 0.8) + (effectiveness * 0.2)
    
    def get_cleanup_statistics(self) -> Dict[str, Any]:
        """Get cleanup statistics and effectiveness metrics"""
        if not self.cleanup_history:
            return {"no_data": "No cleanup history available"}
        
        recent_cleanups = self.cleanup_history[-10:]  # Last 10 cleanups
        
        total_memory_freed = sum(c.get("memory_freed_mb", 0) for c in recent_cleanups)
        avg_effectiveness = sum(c.get("effectiveness", 0) for c in recent_cleanups) / len(recent_cleanups)
        
        cleanup_counts = {}
        for cleanup in recent_cleanups:
            level = cleanup.get("level", "unknown")
            cleanup_counts[level] = cleanup_counts.get(level, 0) + 1
        
        return {
            "total_cleanups": len(self.cleanup_history),
            "recent_cleanups": len(recent_cleanups),
            "total_memory_freed_mb": total_memory_freed,
            "average_effectiveness": avg_effectiveness,
            "cleanup_counts": cleanup_counts,
            "effectiveness_scores": self.effectiveness_scores.copy(),
            "last_cleanup": self.cleanup_history[-1] if self.cleanup_history else None
        }


class MemoryOptimizer:
    """
    Main memory optimizer that coordinates all memory management features
    """
    
    def __init__(self):
        self.resource_manager = get_resource_manager()
        self.adaptive_cleanup = AdaptiveMemoryCleanup()
        
        # Optimization settings
        self.auto_cleanup_enabled = True
        self.cleanup_threshold = 80.0  # Trigger cleanup at 80% memory usage
        self.monitoring_interval = 30.0  # Check memory every 30 seconds
        
        # Background monitoring
        self.monitoring_task: Optional[asyncio.Task] = None
        self.optimization_callbacks: List[Callable] = []
        
        logger.info("MemoryOptimizer initialized")
    
    async def start_optimization(self) -> bool:
        """Start automatic memory optimization"""
        try:
            if self.monitoring_task is not None:
                logger.warning("Memory optimization already running")
                return False
            
            self.monitoring_task = asyncio.create_task(self._optimization_loop())
            logger.info("Memory optimization started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start memory optimization: {e}")
            return False
    
    async def stop_optimization(self) -> bool:
        """Stop automatic memory optimization"""
        try:
            if self.monitoring_task is None:
                return True
            
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            
            self.monitoring_task = None
            logger.info("Memory optimization stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop memory optimization: {e}")
            return False
    
    async def optimize_now(self, level: Optional[CleanupLevel] = None) -> Dict[str, Any]:
        """
        Perform immediate memory optimization
        
        Args:
            level: Specific cleanup level (if None, auto-determined)
            
        Returns:
            Dict with optimization results
        """
        try:
            if level is None:
                return await self.adaptive_cleanup.perform_adaptive_cleanup()
            else:
                # Force specific cleanup level
                strategy = self.adaptive_cleanup.cleanup_strategies[level]
                result = await strategy()
                
                return {
                    "success": True,
                    "level": level.value,
                    "forced": True,
                    "results": result
                }
                
        except Exception as e:
            logger.error(f"Manual optimization failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _optimization_loop(self):
        """Background optimization monitoring loop"""
        logger.info("Starting memory optimization monitoring loop")
        
        try:
            while True:
                try:
                    # Check if auto cleanup is enabled
                    if not self.auto_cleanup_enabled:
                        await asyncio.sleep(self.monitoring_interval)
                        continue
                    
                    # Check memory status
                    memory_status = self.resource_manager.get_memory_status()
                    
                    # Trigger cleanup if threshold exceeded
                    if memory_status.usage_percent >= self.cleanup_threshold:
                        logger.info(f"Memory threshold exceeded ({memory_status.usage_percent:.1f}%), "
                                   f"triggering automatic cleanup")
                        
                        cleanup_result = await self.adaptive_cleanup.perform_adaptive_cleanup()
                        
                        # Notify callbacks
                        await self._notify_optimization_callbacks("auto_cleanup", cleanup_result)
                    
                    # Wait before next check
                    await asyncio.sleep(self.monitoring_interval)
                    
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"Error in optimization loop: {e}")
                    await asyncio.sleep(5)  # Short sleep on error
                    
        except asyncio.CancelledError:
            logger.info("Memory optimization monitoring stopped")
        except Exception as e:
            logger.error(f"Optimization loop failed: {e}")
    
    async def _notify_optimization_callbacks(self, event_type: str, data: Any):
        """Notify registered optimization callbacks"""
        for callback in self.optimization_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event_type, data)
                else:
                    callback(event_type, data)
            except Exception as e:
                logger.warning(f"Optimization callback failed: {e}")
    
    def add_optimization_callback(self, callback: Callable):
        """Add callback for optimization events"""
        self.optimization_callbacks.append(callback)
    
    def configure_optimization(self,
                             auto_cleanup: bool = None,
                             cleanup_threshold: float = None,
                             monitoring_interval: float = None):
        """Configure optimization parameters"""
        if auto_cleanup is not None:
            self.auto_cleanup_enabled = auto_cleanup
            logger.info(f"Auto cleanup {'enabled' if auto_cleanup else 'disabled'}")
        
        if cleanup_threshold is not None:
            self.cleanup_threshold = max(50.0, min(95.0, cleanup_threshold))
            logger.info(f"Cleanup threshold set to {self.cleanup_threshold}%")
        
        if monitoring_interval is not None:
            self.monitoring_interval = max(10.0, monitoring_interval)
            logger.info(f"Monitoring interval set to {self.monitoring_interval}s")
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status and statistics"""
        return {
            "auto_cleanup_enabled": self.auto_cleanup_enabled,
            "cleanup_threshold": self.cleanup_threshold,
            "monitoring_interval": self.monitoring_interval,
            "monitoring_active": self.monitoring_task is not None,
            "cleanup_statistics": self.adaptive_cleanup.get_cleanup_statistics(),
            "current_memory": self.resource_manager.get_memory_status().__dict__,
            "callbacks_registered": len(self.optimization_callbacks)
        }


# Global instance for easy access
_global_memory_optimizer: Optional[MemoryOptimizer] = None


def get_memory_optimizer() -> MemoryOptimizer:
    """Get global memory optimizer instance"""
    global _global_memory_optimizer
    
    if _global_memory_optimizer is None:
        _global_memory_optimizer = MemoryOptimizer()
    
    return _global_memory_optimizer


# Convenience functions for external use
async def optimize_memory_now(level: Optional[str] = None) -> Dict[str, Any]:
    """Perform immediate memory optimization"""
    optimizer = get_memory_optimizer()
    
    if level:
        cleanup_level = CleanupLevel(level.lower())
        return await optimizer.optimize_now(cleanup_level)
    else:
        return await optimizer.optimize_now()


async def start_auto_optimization() -> bool:
    """Start automatic memory optimization"""
    optimizer = get_memory_optimizer()
    return await optimizer.start_optimization()


async def stop_auto_optimization() -> bool:
    """Stop automatic memory optimization"""
    optimizer = get_memory_optimizer()
    return await optimizer.stop_optimization()


def get_memory_stats() -> Dict[str, Any]:
    """Get comprehensive memory statistics"""
    optimizer = get_memory_optimizer()
    return optimizer.get_optimization_status()