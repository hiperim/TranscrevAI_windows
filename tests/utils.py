
import threading
import time
import psutil
import os
from collections import deque
from typing import List, Tuple

class MemoryMonitor:
    """
    A context manager to monitor the peak memory usage (RSS) of the current process
    in a background thread.
    
    Usage:
        with MemoryMonitor() as monitor:
            # Code to be monitored
            ...
        peak_memory_mb = monitor.peak_memory_mb
    """
    def __init__(self, sampling_interval: float = 0.01): # Sample every 10ms
        self._sampling_interval = sampling_interval
        self._peak_memory = 0
        self._stop_event = threading.Event()
        self._monitor_thread = None
        self._process = psutil.Process(os.getpid())

    def __enter__(self):
        self._peak_memory = self._process.memory_info().rss
        self._monitor_thread = threading.Thread(target=self._monitor)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join()

    def _monitor(self):
        while not self._stop_event.is_set():
            try:
                current_memory = self._process.memory_info().rss
                if current_memory > self._peak_memory:
                    self._peak_memory = current_memory
            except psutil.NoSuchProcess:
                break
            time.sleep(self._sampling_interval)

    @property
    def peak_memory_bytes(self) -> int:
        return self._peak_memory

    @property
    def peak_memory_mb(self) -> float:
        return self._peak_memory / (1024 * 1024)
