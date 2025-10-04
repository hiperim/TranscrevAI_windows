# hardware_validator.py - FINAL AND CORRECTED
"""
Advanced Hardware Compatibility Validator for TranscrevAI
"""

import os
import platform
import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import threading

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)

class HardwareCompatibility(Enum):
    OPTIMAL = "optimal"
    COMPATIBLE = "compatible"
    LIMITED = "limited"
    INCOMPATIBLE = "incompatible"

class ComponentStatus(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    ADEQUATE = "adequate"
    INSUFFICIENT = "insufficient"
    MISSING = "missing"

@dataclass
class HardwareRequirements:
    min_cpu_cores: int = 4
    optimal_cpu_cores: int = 8
    min_ram_gb: float = 4.0
    optimal_ram_gb: float = 8.0
    min_free_space_gb: float = 5.0

@dataclass 
class HardwareReport:
    compatibility_level: HardwareCompatibility
    overall_score: float
    cpu_status: ComponentStatus
    memory_status: ComponentStatus
    storage_status: ComponentStatus
    cpu_info: Dict[str, Any] = field(default_factory=dict)
    memory_info: Dict[str, Any] = field(default_factory=dict)
    storage_info: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggested_config: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

class HardwareValidator:
    """Advanced hardware compatibility validator with auto-configuration."""
    
    def __init__(self, requirements: Optional[HardwareRequirements] = None):
        self.requirements = requirements or HardwareRequirements()
        self.cached_report: Optional[HardwareReport] = None
        self.cache_duration = 300
        self._lock = threading.Lock()
        
    def validate_system(self, force_refresh: bool = False) -> HardwareReport:
        with self._lock:
            if (self.cached_report and not force_refresh and time.time() - self.cached_report.timestamp < self.cache_duration):
                return self.cached_report
            
            report = HardwareReport(HardwareCompatibility.INCOMPATIBLE, 0.0, ComponentStatus.MISSING, ComponentStatus.MISSING, ComponentStatus.MISSING)
            self._validate_cpu(report)
            self._validate_memory(report)
            self._validate_storage(report)
            self._calculate_compatibility(report)
            self._generate_recommendations(report)
            self._generate_auto_config(report)
            self.cached_report = report
            return report
    
    def _validate_cpu(self, report: HardwareReport):
        if not PSUTIL_AVAILABLE: 
            report.warnings.append("psutil not found, cannot validate CPU.")
            report.cpu_status = ComponentStatus.MISSING
            return
        report.cpu_info = {'logical_cores': psutil.cpu_count(logical=True), 'physical_cores': psutil.cpu_count(logical=False)}
        if report.cpu_info['logical_cores'] >= self.requirements.optimal_cpu_cores:
            report.cpu_status = ComponentStatus.EXCELLENT
        elif report.cpu_info['logical_cores'] >= self.requirements.min_cpu_cores:
            report.cpu_status = ComponentStatus.ADEQUATE
        else:
            report.cpu_status = ComponentStatus.INSUFFICIENT

    def _validate_memory(self, report: HardwareReport):
        if not PSUTIL_AVAILABLE: 
            report.warnings.append("psutil not found, cannot validate Memory.")
            report.memory_status = ComponentStatus.MISSING
            return
        mem = psutil.virtual_memory()
        report.memory_info = {'total_gb': mem.total / (1024**3), 'available_gb': mem.available / (1024**3)}
        if report.memory_info['total_gb'] >= self.requirements.optimal_ram_gb:
            report.memory_status = ComponentStatus.EXCELLENT
        elif report.memory_info['total_gb'] >= self.requirements.min_ram_gb:
            report.memory_status = ComponentStatus.ADEQUATE
        else:
            report.memory_status = ComponentStatus.INSUFFICIENT

    def _validate_storage(self, report: HardwareReport):
        if not PSUTIL_AVAILABLE: 
            report.warnings.append("psutil not found, cannot validate Storage.")
            report.storage_status = ComponentStatus.MISSING
            return
        usage = psutil.disk_usage('/')
        report.storage_info = {'free_gb': usage.free / (1024**3)}
        if report.storage_info['free_gb'] >= self.requirements.min_free_space_gb:
            report.storage_status = ComponentStatus.ADEQUATE
        else:
            report.storage_status = ComponentStatus.INSUFFICIENT

    def _calculate_compatibility(self, report: HardwareReport):
        scores = {ComponentStatus.EXCELLENT: 1.0, ComponentStatus.ADEQUATE: 0.6, ComponentStatus.INSUFFICIENT: 0.1, ComponentStatus.MISSING: 0.0}
        weights = {'cpu': 0.5, 'memory': 0.4, 'storage': 0.1}
        report.overall_score = sum(scores.get(s, 0) * w for s, w in [(report.cpu_status, weights['cpu']), (report.memory_status, weights['memory']), (report.storage_status, weights['storage'])])
        if report.overall_score >= 0.8: report.compatibility_level = HardwareCompatibility.OPTIMAL
        elif report.overall_score >= 0.5: report.compatibility_level = HardwareCompatibility.COMPATIBLE
        elif report.overall_score > 0.1: report.compatibility_level = HardwareCompatibility.LIMITED
        else: report.compatibility_level = HardwareCompatibility.INCOMPATIBLE

    def _generate_recommendations(self, report: HardwareReport):
        if report.cpu_status == ComponentStatus.INSUFFICIENT: report.recommendations.append("Upgrade CPU to at least 4 cores.")
        if report.memory_status == ComponentStatus.INSUFFICIENT: report.recommendations.append("Upgrade RAM to at least 4GB.")
        if report.storage_status == ComponentStatus.INSUFFICIENT: report.recommendations.append("Free up disk space (at least 5GB recommended).")

    def _generate_auto_config(self, report: HardwareReport):
        cpu_cores = report.cpu_info.get('logical_cores', 1)
        report.suggested_config['max_workers'] = max(1, cpu_cores - 2)
        if report.compatibility_level == HardwareCompatibility.LIMITED: report.suggested_config['compute_type'] = 'int8'
        else: report.suggested_config['compute_type'] = 'float16'

# Moved factory function to be defined after the class
_global_validator: Optional[HardwareValidator] = None

def get_hardware_validator() -> HardwareValidator:
    global _global_validator
    if _global_validator is None:
        _global_validator = HardwareValidator()
    return _global_validator
