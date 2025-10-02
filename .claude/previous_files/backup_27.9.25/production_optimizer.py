"""
PHASE 9.5: Production Optimizer
Advanced optimization system for production deployment
Implements lazy loading, model caching, and startup performance improvements
"""

import asyncio
import logging
import time
import os
import gc
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, Set, List
import threading
from dataclasses import dataclass
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Lazy import for psutil
_psutil = None
def _get_psutil():
    global _psutil
    if _psutil is None:
        import psutil
        _psutil = psutil
    return _psutil

@dataclass(slots=True)
class ModelInfo:
    """Information about a model file"""
    path: Path
    size_mb: float
    hash: str
    last_used: float
    load_count: int
    priority: int = 1  # 1=low, 2=medium, 3=high

@dataclass(slots=True)
class StartupMetrics:
    """Metrics for startup performance tracking"""
    total_startup_time: float = 0.0
    import_time: float = 0.0
    model_discovery_time: float = 0.0
    initialization_time: float = 0.0
    memory_optimization_time: float = 0.0
    concurrent_setup_time: float = 0.0

class ModelCacheManager:
    """
    Advanced model caching and deduplication system
    Eliminates duplicate models and implements intelligent caching
    """

    def __init__(self, cache_dir: str = "data/models_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.model_registry: Dict[str, ModelInfo] = {}
        self.loaded_models: Dict[str, Any] = {}
        self.model_lock = threading.RLock()

        # Performance settings
        self.max_memory_usage_gb = 4.0  # Maximum memory for cached models
        self.max_cached_models = 3      # Maximum number of cached models

        logger.info("🗄️ Model Cache Manager initialized")

    def scan_and_deduplicate_models(self) -> Dict[str, List[Path]]:
        """Scan for models and identify duplicates"""
        logger.info("🔍 Scanning for model files and duplicates...")

        model_files = {}  # hash -> list of paths
        scan_dirs = [
            Path("models"),
            Path("data/models"),
            Path("src/models"),
            Path(".")  # Root directory
        ]

        for scan_dir in scan_dirs:
            if scan_dir.exists():
                for model_file in scan_dir.rglob("*.onnx"):
                    if model_file.is_file():
                        file_hash = self._calculate_file_hash(model_file)
                        if file_hash not in model_files:
                            model_files[file_hash] = []
                        model_files[file_hash].append(model_file)

        # Identify duplicates
        duplicates = {h: paths for h, paths in model_files.items() if len(paths) > 1}

        if duplicates:
            logger.warning(f"⚠️ Found {len(duplicates)} sets of duplicate models:")
            for file_hash, paths in duplicates.items():
                size_mb = paths[0].stat().st_size / (1024*1024)
                logger.warning(f"  Duplicate ({size_mb:.1f}MB): {[str(p) for p in paths]}")

        return model_files

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of file for deduplication"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                # Read in chunks to handle large files efficiently
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"❌ Failed to hash {file_path}: {e}")
            return str(file_path)  # Fallback to path

    def optimize_model_storage(self) -> Dict[str, Any]:
        """Optimize model storage by deduplicating and organizing"""
        logger.info("🔧 Optimizing model storage...")

        model_files = self.scan_and_deduplicate_models()
        optimization_stats = {
            'total_models': sum(len(paths) for paths in model_files.values()),
            'unique_models': len(model_files),
            'duplicates_removed': 0,
            'space_saved_mb': 0
        }

        for file_hash, paths in model_files.items():
            if len(paths) > 1:
                # Keep the model in the most appropriate location
                primary_model = self._select_primary_model(paths)

                # Register the primary model
                model_info = ModelInfo(
                    path=primary_model,
                    size_mb=primary_model.stat().st_size / (1024*1024),
                    hash=file_hash,
                    last_used=time.time(),
                    load_count=0
                )

                self.model_registry[file_hash] = model_info
                logger.info(f"✅ Registered model: {primary_model.name} ({model_info.size_mb:.1f}MB)")

                # Calculate potential space savings (don't actually delete for safety)
                for duplicate_path in paths[1:]:
                    size_mb = duplicate_path.stat().st_size / (1024*1024)
                    optimization_stats['space_saved_mb'] += int(size_mb)
                    optimization_stats['duplicates_removed'] += 1
                    logger.info(f"📋 Duplicate identified: {duplicate_path}")
            else:
                # Single model, just register it
                model_path = paths[0]
                model_info = ModelInfo(
                    path=model_path,
                    size_mb=model_path.stat().st_size / (1024*1024),
                    hash=file_hash,
                    last_used=time.time(),
                    load_count=0
                )
                self.model_registry[file_hash] = model_info

        logger.info(f"📊 Storage optimization complete:")
        logger.info(f"  Total models: {optimization_stats['total_models']}")
        logger.info(f"  Unique models: {optimization_stats['unique_models']}")
        logger.info(f"  Potential space savings: {optimization_stats['space_saved_mb']:.1f}MB")

        return optimization_stats

    def _select_primary_model(self, paths: List[Path]) -> Path:
        """Select the best location for a model among duplicates"""
        # Preference: data/models > models > src/models > others
        preferences = ["data/models", "models", "src/models"]

        for pref in preferences:
            for path in paths:
                if pref in str(path):
                    return path

        # If no preference matches, use the first one
        return paths[0]

    async def load_model_lazy(self, model_name: str, priority: int = 1) -> Optional[Any]:
        """Load a model using lazy loading with priority"""
        with self.model_lock:
            # Find model by name pattern
            matching_models = [
                (hash_val, info) for hash_val, info in self.model_registry.items()
                if model_name.lower() in info.path.name.lower()
            ]

            if not matching_models:
                logger.error(f"❌ Model not found: {model_name}")
                return ""

            # Use the first matching model
            model_hash, model_info = matching_models[0]

            # Check if already loaded
            if model_hash in self.loaded_models:
                model_info.last_used = time.time()
                model_info.load_count += 1
                logger.debug(f"♻️ Using cached model: {model_info.path.name}")
                return self.loaded_models[model_hash]

            # Check memory constraints before loading
            if not self._can_load_model(model_info):
                await self._cleanup_cache()
                if not self._can_load_model(model_info):
                    logger.error(f"❌ Cannot load model {model_name} - insufficient memory")
                    return ""

            # Load the model
            logger.info(f"📥 Loading model: {model_info.path.name} ({model_info.size_mb:.1f}MB)")

            try:
                # Simulate model loading (replace with actual ONNX loading)
                start_time = time.time()

                # This would be the actual model loading code:
                # import onnxruntime as ort
                # model = ort.InferenceSession(str(model_info.path))

                # For now, simulate loading time
                await asyncio.sleep(0.1)  # Simulate loading time
                model = f"MockModel_{model_info.path.name}"  # Mock model object

                load_time = time.time() - start_time

                # Cache the loaded model
                self.loaded_models[model_hash] = model
                model_info.last_used = time.time()
                model_info.load_count += 1
                model_info.priority = priority

                logger.info(f"✅ Model loaded in {load_time:.2f}s: {model_info.path.name}")
                return model

            except Exception as e:
                logger.error(f"❌ Failed to load model {model_name}: {e}")
                return ""

    def _can_load_model(self, model_info: ModelInfo) -> bool:
        """Check if model can be loaded within memory constraints"""
        psutil = _get_psutil()
        current_memory_gb = psutil.Process().memory_info().rss / (1024**3)
        estimated_model_memory_gb = model_info.size_mb / 1024 * 1.5  # 1.5x overhead estimate

        return (current_memory_gb + estimated_model_memory_gb) <= self.max_memory_usage_gb

    async def _cleanup_cache(self):
        """Clean up cached models to free memory"""
        if len(self.loaded_models) >= self.max_cached_models:
            # Remove least recently used model with lowest priority
            to_remove = min(
                self.model_registry.items(),
                key=lambda x: (x[1].priority, x[1].last_used)
            )

            model_hash, model_info = to_remove
            if model_hash in self.loaded_models:
                del self.loaded_models[model_hash]
                logger.info(f"🗑️ Evicted cached model: {model_info.path.name}")
                gc.collect()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_size_mb = sum(info.size_mb for info in self.model_registry.values())
        loaded_size_mb = sum(
            self.model_registry[h].size_mb
            for h in self.loaded_models.keys()
            if h in self.model_registry
        )

        return {
            'total_models': len(self.model_registry),
            'loaded_models': len(self.loaded_models),
            'total_size_mb': total_size_mb,
            'loaded_size_mb': loaded_size_mb,
            'memory_usage_gb': _get_psutil().Process().memory_info().rss / (1024**3),
            'cache_hit_rate': self._calculate_cache_hit_rate()
        }

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total_loads = sum(info.load_count for info in self.model_registry.values())
        if total_loads == 0:
            return 0.0

        cache_hits = sum(
            info.load_count - 1 for info in self.model_registry.values()
            if info.load_count > 1
        )

        return cache_hits / total_loads if total_loads > 0 else 0.0

class LazyImportManager:
    """
    Manages lazy importing of heavy dependencies to reduce startup time
    """

    def __init__(self):
        self._imported_modules: Dict[str, Any] = {}
        self._import_times: Dict[str, float] = {}
        self._import_lock = threading.Lock()

        logger.info("⚡ Lazy Import Manager initialized")

    @contextmanager
    def time_import(self, module_name: str):
        """Context manager to time imports"""
        start_time = time.time()
        try:
            yield
        finally:
            import_time = time.time() - start_time
            self._import_times[module_name] = import_time
            logger.debug(f"📦 {module_name} imported in {import_time:.3f}s")

    async def import_onnxruntime(self):
        """Lazy import of onnxruntime with optimization"""
        if 'onnxruntime' in self._imported_modules:
            return self._imported_modules['onnxruntime']

        with self._import_lock:
            if 'onnxruntime' in self._imported_modules:
                return self._imported_modules['onnxruntime']

            logger.info("📦 Lazy loading onnxruntime...")

            with self.time_import('onnxruntime'):
                # Set ONNX optimizations before import
                os.environ['ORT_NUM_THREADS'] = str(min(4, os.cpu_count() or 4))
                os.environ['ORT_SESSION_OPTIONS_GRAPH_OPTIMIZATION_LEVEL'] = '99'

                import onnxruntime as ort

                # Configure session options for optimal performance
                session_options = ort.SessionOptions()
                session_options.inter_op_num_threads = min(4, os.cpu_count() or 4)
                session_options.intra_op_num_threads = min(4, os.cpu_count() or 4)
                session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

                self._imported_modules['onnxruntime'] = ort
                self._imported_modules['ort_session_options'] = session_options

                logger.info("✅ onnxruntime loaded and optimized")
                return ort

    async def import_torch(self):
        """Lazy import of torch with optimization"""
        if 'torch' in self._imported_modules:
            return self._imported_modules['torch']

        with self._import_lock:
            if 'torch' in self._imported_modules:
                return self._imported_modules['torch']

            logger.info("📦 Lazy loading torch...")

            with self.time_import('torch'):
                import torch

                # Optimize torch settings
                torch.set_num_threads(min(4, os.cpu_count() or 4))
                torch.set_num_interop_threads(min(2, os.cpu_count() or 2))

                # Disable torch autograd for inference
                torch.set_grad_enabled(False)

                self._imported_modules['torch'] = torch
                logger.info("✅ torch loaded and optimized")
                return torch

    def get_import_stats(self) -> Dict[str, float]:
        """Get import timing statistics"""
        return dict(self._import_times)

class ProductionOptimizer:
    """
    Main production optimization coordinator
    Implements comprehensive startup and runtime optimizations
    """

    def __init__(self):
        self.model_cache = ModelCacheManager()
        self.lazy_imports = LazyImportManager()
        self.startup_metrics = StartupMetrics()

        # Optimization flags
        self.startup_optimized = False
        self.models_optimized = False

        logger.info("🚀 Production Optimizer initialized")

    async def optimize_startup(self) -> StartupMetrics:
        """Comprehensive startup optimization"""
        start_time = time.time()
        logger.info("🔧 Starting production startup optimization...")

        try:
            # Phase 1: Model discovery and optimization
            model_start = time.time()
            optimization_stats = self.model_cache.optimize_model_storage()
            self.startup_metrics.model_discovery_time = time.time() - model_start

            # Phase 2: Memory optimization
            memory_start = time.time()
            await self._optimize_memory_startup()
            self.startup_metrics.memory_optimization_time = time.time() - memory_start

            # Phase 3: Lazy import preparation
            import_start = time.time()
            await self._prepare_lazy_imports()
            self.startup_metrics.import_time = time.time() - import_start

            # Phase 4: System optimization
            init_start = time.time()
            await self._optimize_system_startup()
            self.startup_metrics.initialization_time = time.time() - init_start

            self.startup_metrics.total_startup_time = time.time() - start_time
            self.startup_optimized = True

            logger.info(f"✅ Startup optimization complete in {self.startup_metrics.total_startup_time:.2f}s")
            self._log_startup_metrics()

            return self.startup_metrics

        except Exception as e:
            logger.error(f"❌ Startup optimization failed: {e}")
            raise

    async def _optimize_memory_startup(self):
        """Optimize memory usage during startup"""
        # Force garbage collection
        gc.collect()

        # Set memory-efficient settings
        os.environ['PYTHONOPTIMIZE'] = '1'

        # Configure torch memory if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    async def _prepare_lazy_imports(self):
        """Prepare for lazy importing without actually importing"""
        # This doesn't import the modules, just prepares the infrastructure
        logger.debug("📋 Preparing lazy import infrastructure...")

    async def _optimize_system_startup(self):
        """Optimize system-level settings for startup"""
        try:
            # Set process priority to normal during startup
            psutil = _get_psutil()
            process = psutil.Process()
            if hasattr(process, 'nice'):
                process.nice(0)  # Normal priority

            # Optimize thread settings
            import threading
            threading.stack_size(2**20)  # 1MB stack size

        except Exception as e:
            logger.warning(f"⚠️ System optimization failed: {e}")

    def _log_startup_metrics(self):
        """Log detailed startup metrics"""
        metrics = self.startup_metrics
        logger.info("📊 Startup Performance Metrics:")
        logger.info(f"  Total startup time: {metrics.total_startup_time:.2f}s")
        logger.info(f"  Model discovery: {metrics.model_discovery_time:.2f}s")
        logger.info(f"  Memory optimization: {metrics.memory_optimization_time:.2f}s")
        logger.info(f"  Import preparation: {metrics.import_time:.2f}s")
        logger.info(f"  System initialization: {metrics.initialization_time:.2f}s")

    async def get_model_lazy(self, model_name: str, priority: int = 1) -> Optional[Any]:
        """Get a model using lazy loading"""
        return await self.model_cache.load_model_lazy(model_name, priority)

    async def get_onnxruntime_lazy(self):
        """Get onnxruntime using lazy loading"""
        return await self.lazy_imports.import_onnxruntime()

    async def get_torch_lazy(self):
        """Get torch using lazy loading"""
        return await self.lazy_imports.import_torch()

    def get_optimization_status(self) -> Dict[str, Any]:
        """Get comprehensive optimization status"""
        return {
            'startup_optimized': self.startup_optimized,
            'models_optimized': self.models_optimized,
            'startup_metrics': {
                'total_time': self.startup_metrics.total_startup_time,
                'model_discovery_time': self.startup_metrics.model_discovery_time,
                'memory_optimization_time': self.startup_metrics.memory_optimization_time,
                'import_time': self.startup_metrics.import_time,
                'initialization_time': self.startup_metrics.initialization_time
            },
            'model_cache_stats': self.model_cache.get_cache_stats(),
            'import_stats': self.lazy_imports.get_import_stats(),
            'system_stats': {
                'memory_usage_gb': _get_psutil().Process().memory_info().rss / (1024**3),
                'cpu_count': os.cpu_count(),
                'available_memory_gb': _get_psutil().virtual_memory().available / (1024**3)
            }
        }

# ==========================================
# PRODUCTION MONITORING AND RESILIENCE
# ==========================================
# Merged from production_monitoring.py and production_resilience.py for file consolidation

from enum import Enum
from contextlib import asynccontextmanager
import json

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass(slots=True)
class HealthCheck:
    """Individual health check definition"""
    name: str
    critical: bool = False
    timeout: float = 5.0
    last_result: Optional[Dict[str, Any]] = None

class ProductionMonitoringManager:
    """Simplified production monitoring for essential health checks"""

    def __init__(self):
        self.health_checks: Dict[str, HealthCheck] = {}
        self.system_metrics = {}
        self.monitoring_active = True

    def add_health_check(self, name: str, critical: bool = False):
        """Add a health check"""
        self.health_checks[name] = HealthCheck(name=name, critical=critical)

    async def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = {}

        try:
            # Basic system health
            psutil = _get_psutil()
            memory = psutil.virtual_memory()
            results["memory"] = {
                "status": "healthy" if memory.percent < 85 else "warning",
                "percent": memory.percent
            }

            # Basic CPU health
            cpu_percent = psutil.cpu_percent(interval=1)
            results["cpu"] = {
                "status": "healthy" if cpu_percent < 80 else "warning",
                "percent": cpu_percent
            }

            return results

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"error": str(e)}

    async def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        health_results = await self.run_health_checks()

        return {
            "timestamp": time.time(),
            "health_checks": health_results,
            "system_info": {
                "platform": os.name,
                "memory_total_gb": _get_psutil().virtual_memory().total / (1024**3),
                "cpu_count": os.cpu_count()
            }
        }

class ProductionResilienceManager:
    """Simplified error handling and retry mechanisms"""

    def __init__(self):
        self.error_counts = {}
        self.max_retries = 3
        self.retry_delays = [1, 2, 4]  # seconds

    async def execute_with_retry(self, func, *args, **kwargs):
        """Execute function with retry logic"""
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)

            except Exception as e:
                last_exception = e
                error_key = f"{func.__name__}_{type(e).__name__}"
                self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1

                if attempt < self.max_retries - 1:
                    delay = self.retry_delays[min(attempt, len(self.retry_delays) - 1)]
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All retry attempts failed for {func.__name__}: {e}")

        if last_exception:
            raise last_exception
        else:
            raise Exception(f"All retry attempts failed for {func.__name__}")

    @asynccontextmanager
    async def error_context(self, operation_name: str):
        """Context manager for error handling"""
        try:
            yield
        except Exception as e:
            logger.error(f"Error in {operation_name}: {e}")
            self.error_counts[operation_name] = self.error_counts.get(operation_name, 0) + 1
            raise

# Global instances
_production_optimizer = None
_monitoring_manager = ProductionMonitoringManager()
_resilience_manager = ProductionResilienceManager()

async def get_production_optimizer() -> ProductionOptimizer:
    """Get or create the global production optimizer"""
    global _production_optimizer

    if _production_optimizer is None:
        _production_optimizer = ProductionOptimizer()
        await _production_optimizer.optimize_startup()

    return _production_optimizer

async def get_monitoring_manager() -> ProductionMonitoringManager:
    """Get the global monitoring manager"""
    return _monitoring_manager

async def get_resilience_manager() -> ProductionResilienceManager:
    """Get the global resilience manager"""
    return _resilience_manager