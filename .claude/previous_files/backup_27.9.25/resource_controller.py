"""
 Resource CMinimalontroller for Testing
"""
import logging
import psutil

logger = logging.getLogger(__name__)

class MockResourceController:
    """Minimal resource controller for testing"""

    def __init__(self):
        self.memory_limit_mb = 2048
        self.emergency_threshold = 0.85
        self.conservative_mode = False
        self.memory_reservations = {}
        self.next_reservation_id = 1
        # PHASE 2.5: Model streaming mode support
        self.streaming_mode = False
        self.streaming_threshold = 0.80  # Enable streaming at 80% memory usage
        logger.info("Unified Resource Controller initialized")
        logger.info(f"   Memory: {psutil.virtual_memory().total / (1024**3):.1f}GB, CPU: {psutil.cpu_count()} cores")
        logger.info(f"   Emergency threshold: {self.emergency_threshold*100:.1f}% RAM")
        logger.info(f"   Streaming threshold: {self.streaming_threshold*100:.1f}% RAM")

    def get_memory_usage(self):
        return psutil.virtual_memory().percent

    def is_emergency_mode(self):
        return self.get_memory_usage() > self.emergency_threshold * 100

    def is_conservative_mode(self):
        return self.conservative_mode

    def should_use_streaming_mode(self):
        """Check if should use model streaming mode based on memory pressure"""
        memory_usage = self.get_memory_usage()
        return memory_usage > (self.streaming_threshold * 100)

    def enable_streaming_mode(self):
        """Enable streaming mode for memory optimization"""
        self.streaming_mode = True
        logger.info("Streaming mode ENABLED - Sequential model loading activated")

    def disable_streaming_mode(self):
        """Disable streaming mode (rollback capability)"""
        self.streaming_mode = False
        logger.info("Streaming mode DISABLED - Concurrent model loading restored")

    def is_streaming_mode(self):
        """Check if streaming mode is active"""
        return self.streaming_mode

    def reserve_memory(self, amount_mb):
        """Reserve memory and return reservation ID"""
        reservation_id = self.next_reservation_id
        self.memory_reservations[reservation_id] = amount_mb
        self.next_reservation_id += 1
        return reservation_id

    def release_memory(self, reservation_id):
        """Release memory reservation"""
        if reservation_id in self.memory_reservations:
            del self.memory_reservations[reservation_id]

    def get_memory_status(self):
        """Get current memory status"""
        vm = psutil.virtual_memory()
        return {
            'total_mb': vm.total / (1024 * 1024),
            'available_mb': vm.available / (1024 * 1024),
            'used_percent': vm.percent,
            'reserved_mb': sum(self.memory_reservations.values())
        }

    def get_available_memory_mb(self):
        """Get available memory in MB"""
        return psutil.virtual_memory().available / (1024 * 1024)

    def can_safely_allocate(self, amount_mb):
        """Check if can safely allocate given amount of memory"""
        available = self.get_available_memory_mb()
        return available > amount_mb * 1.2  # 20% safety margin

    def get_system_state(self):
        """Get current system state"""
        return {
            'memory_status': self.get_memory_status(),
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'emergency_mode': self.is_emergency_mode(),
            'conservative_mode': self.is_conservative_mode(),
            'streaming_mode': self.is_streaming_mode(),
            'should_stream': self.should_use_streaming_mode()
        }

    def get_recommended_settings(self):
        """Get recommended settings based on current system state"""
        available_memory = self.get_available_memory_mb()
        cpu_count = psutil.cpu_count() or 4

        if available_memory < 1024:  # Less than 1GB
            return {
                'max_concurrent_sessions': 1,
                'batch_size': 1,
                'num_threads': 1
            }
        elif available_memory < 4096:  # Less than 4GB
            return {
                'max_concurrent_sessions': 2,
                'batch_size': 2,
                'num_threads': min(2, cpu_count)
            }
        else:  # 4GB or more
            return {
                'max_concurrent_sessions': min(4, cpu_count),
                'batch_size': 4,
                'num_threads': min(4, cpu_count)
            }

    def get_cpu_config(self):
        """Get CPU configuration"""
        return {
            'cpu_count': psutil.cpu_count() or 4,
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'recommended_threads': min(4, psutil.cpu_count() or 4)
        }

_controller = None

def get_unified_resource_controller():
    """Get unified resource controller instance"""
    global _controller
    if _controller is None:
        _controller = MockResourceController()
    return _controller

def get_unified_controller():
    """Alias for compatibility"""
    return get_unified_resource_controller()