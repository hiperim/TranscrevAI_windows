"""
TranscrevAI Optimized - Progressive Loader Module
Sistema de carregamento progressivo browser-safe que previne travamentos
"""

import asyncio
import gc
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import threading
from dataclasses import dataclass
from enum import Enum

# Import our optimized modules
from logging_setup import get_logger, log_performance
from resource_manager import get_resource_manager, ResourceStatus
from config import CONFIG

logger = get_logger("transcrevai.progressive_loader")


class LoadingPhase(Enum):
    """Loading phases for progressive initialization"""
    ESSENTIAL = "essential"
    MODELS = "models"
    PROCESSING = "processing"
    OPTIMIZATION = "optimization"
    COMPLETE = "complete"


@dataclass
class LoadingTask:
    """Individual loading task definition"""
    name: str
    phase: LoadingPhase
    priority: int  # Lower = higher priority
    estimated_time: float  # Seconds
    memory_required: float  # MB
    callback: Callable
    dependencies: List[str] = None
    critical: bool = False  # If True, failure stops the loading process
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class BrowserSafeLoader:
    """
    Browser-safe loader that prevents UI freezing during heavy operations
    Implements progressive loading with yielding and memory management
    """
    
    def __init__(self):
        self.resource_manager = get_resource_manager()
        
        # Loading state
        self.current_phase = LoadingPhase.ESSENTIAL
        self.loading_tasks: Dict[str, LoadingTask] = {}
        self.completed_tasks: set = set()
        self.failed_tasks: set = set()
        self.loading_in_progress = False
        
        # Browser safety settings
        self.max_blocking_time = 50  # ms - maximum time before yielding
        self.yield_time = 10  # ms - how long to yield
        self.memory_check_interval = 5  # Check memory every N tasks
        
        # Progress tracking
        self.progress_callbacks: List[Callable] = []
        self.total_estimated_time = 0.0
        self.current_progress = 0.0
        
        logger.info("BrowserSafeLoader initialized")
    
    def register_task(self, task: LoadingTask) -> None:
        """Register a loading task"""
        self.loading_tasks[task.name] = task
        self.total_estimated_time += task.estimated_time
        logger.debug(f"Registered loading task: {task.name} (Phase: {task.phase.value})")
    
    def add_progress_callback(self, callback: Callable[[float, str], None]) -> None:
        """Add progress callback for UI updates"""
        self.progress_callbacks.append(callback)
    
    async def load_progressively(self, 
                                target_phases: List[LoadingPhase] = None) -> Dict[str, Any]:
        """
        Execute progressive loading up to target phases
        
        Args:
            target_phases: List of phases to complete (if None, load all)
            
        Returns:
            Dict with loading results and metrics
        """
        if self.loading_in_progress:
            logger.warning("Progressive loading already in progress")
            return {"success": False, "error": "Loading already in progress"}
        
        load_start = time.time()
        self.loading_in_progress = True
        
        try:
            # Set target phases
            if target_phases is None:
                target_phases = list(LoadingPhase)
            
            logger.info(f"Starting progressive loading for phases: {[p.value for p in target_phases]}")
            
            # Execute phases in order
            results = {}
            for phase in target_phases:
                if not await self._load_phase(phase):
                    results["last_completed_phase"] = phase.value
                    break
                results[f"phase_{phase.value}"] = "completed"
            
            load_duration = time.time() - load_start
            
            # Final progress update
            await self._update_progress(100.0, "Carregamento concluído!")
            
            # Log performance
            log_performance(
                "Progressive loading completed",
                duration=load_duration,
                phases_completed=len(results),
                tasks_completed=len(self.completed_tasks),
                tasks_failed=len(self.failed_tasks)
            )
            
            logger.info(f"Progressive loading completed in {load_duration:.2f}s")
            return {
                "success": True,
                "duration": load_duration,
                "phases_completed": results,
                "tasks_completed": len(self.completed_tasks),
                "tasks_failed": len(self.failed_tasks),
                "failed_tasks": list(self.failed_tasks)
            }
            
        except Exception as e:
            logger.error(f"Progressive loading failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "tasks_completed": len(self.completed_tasks),
                "tasks_failed": len(self.failed_tasks)
            }
        finally:
            self.loading_in_progress = False
    
    async def _load_phase(self, phase: LoadingPhase) -> bool:
        """Load all tasks for a specific phase"""
        phase_start = time.time()
        self.current_phase = phase
        
        # Get tasks for this phase
        phase_tasks = [
            task for task in self.loading_tasks.values() 
            if task.phase == phase and task.name not in self.completed_tasks
        ]
        
        if not phase_tasks:
            logger.info(f"No tasks found for phase {phase.value}")
            return True
        
        # Sort by priority (lower number = higher priority)
        phase_tasks.sort(key=lambda t: (t.priority, t.estimated_time))
        
        logger.info(f"Loading phase {phase.value}: {len(phase_tasks)} tasks")
        await self._update_progress(
            self.current_progress,
            f"Iniciando fase {phase.value}..."
        )
        
        # Execute tasks with browser-safe yielding
        phase_success = True
        task_count = 0
        
        for task in phase_tasks:
            try:
                # Check dependencies
                if not self._check_dependencies(task):
                    logger.warning(f"Task {task.name} dependencies not met, skipping")
                    continue
                
                # Check memory before task
                if task_count % self.memory_check_interval == 0:
                    if not await self._check_memory_safety(task):
                        logger.warning(f"Insufficient memory for task {task.name}, skipping")
                        continue
                
                # Execute task with browser-safe timing
                task_success = await self._execute_task_safe(task)
                
                if task_success:
                    self.completed_tasks.add(task.name)
                    logger.debug(f"Task {task.name} completed successfully")
                else:
                    self.failed_tasks.add(task.name)
                    if task.critical:
                        logger.error(f"Critical task {task.name} failed, stopping phase")
                        phase_success = False
                        break
                    else:
                        logger.warning(f"Non-critical task {task.name} failed, continuing")
                
                # Update progress
                progress_increment = (task.estimated_time / self.total_estimated_time) * 100
                self.current_progress = min(95.0, self.current_progress + progress_increment)
                await self._update_progress(
                    self.current_progress,
                    f"Concluído: {task.name}"
                )
                
                # Browser-safe yielding after each task
                await self._yield_control()
                
                task_count += 1
                
            except Exception as e:
                logger.error(f"Error executing task {task.name}: {e}")
                self.failed_tasks.add(task.name)
                if task.critical:
                    phase_success = False
                    break
        
        phase_duration = time.time() - phase_start
        logger.info(f"Phase {phase.value} completed in {phase_duration:.2f}s")
        
        return phase_success
    
    async def _execute_task_safe(self, task: LoadingTask) -> bool:
        """Execute a single task with browser-safe timing"""
        task_start = time.time()
        logger.debug(f"Executing task: {task.name}")
        
        try:
            # Reserve memory for task
            if task.memory_required > 0:
                if not self.resource_manager.reserve_memory(
                    f"loading_{task.name}",
                    task.memory_required,
                    "progressive_loading"
                ):
                    logger.warning(f"Could not reserve memory for task {task.name}")
            
            # Execute task with timeout
            try:
                # Run task callback
                if asyncio.iscoroutinefunction(task.callback):
                    result = await asyncio.wait_for(
                        task.callback(),
                        timeout=task.estimated_time * 3  # 3x estimated time as timeout
                    )
                else:
                    # Run sync function in executor
                    loop = asyncio.get_event_loop()
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, task.callback),
                        timeout=task.estimated_time * 3
                    )
                
                # Check if task took too long (browser safety)
                task_duration = time.time() - task_start
                if task_duration > self.max_blocking_time / 1000:  # Convert ms to seconds
                    logger.debug(f"Task {task.name} took {task_duration:.3f}s, yielding control")
                    await self._yield_control()
                
                return result is not False  # None or True = success, False = failure
                
            except asyncio.TimeoutError:
                logger.error(f"Task {task.name} timed out after {task.estimated_time * 3:.1f}s")
                return False
                
        except Exception as e:
            logger.error(f"Task {task.name} failed: {e}")
            return False
        finally:
            # Release memory reservation
            if task.memory_required > 0:
                self.resource_manager.release_memory_reservation(f"loading_{task.name}")
    
    def _check_dependencies(self, task: LoadingTask) -> bool:
        """Check if task dependencies are satisfied"""
        if not task.dependencies:
            return True
        
        for dependency in task.dependencies:
            if dependency not in self.completed_tasks:
                return False
        
        return True
    
    async def _check_memory_safety(self, task: LoadingTask) -> bool:
        """Check if there's enough memory for the task"""
        if task.memory_required <= 0:
            return True
        
        # Check current memory status
        if self.resource_manager.is_memory_pressure_high():
            logger.warning("High memory pressure detected, attempting cleanup")
            await self.resource_manager.perform_cleanup(aggressive=False)
        
        # Check if we can allocate the required memory
        can_allocate = self.resource_manager.can_allocate(task.memory_required)
        
        if not can_allocate:
            logger.warning(f"Cannot allocate {task.memory_required}MB for task {task.name}")
        
        return can_allocate
    
    async def _yield_control(self) -> None:
        """Yield control to prevent browser freezing"""
        await asyncio.sleep(self.yield_time / 1000)  # Convert ms to seconds
    
    async def _update_progress(self, progress: float, message: str) -> None:
        """Update progress for all registered callbacks"""
        for callback in self.progress_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(progress, message)
                else:
                    callback(progress, message)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")
    
    def get_loading_status(self) -> Dict[str, Any]:
        """Get current loading status"""
        return {
            "loading_in_progress": self.loading_in_progress,
            "current_phase": self.current_phase.value,
            "tasks_registered": len(self.loading_tasks),
            "tasks_completed": len(self.completed_tasks),
            "tasks_failed": len(self.failed_tasks),
            "current_progress": self.current_progress,
            "failed_tasks": list(self.failed_tasks)
        }


class ProgressiveLoader:
    """
    Main progressive loader that coordinates browser-safe initialization
    """
    
    def __init__(self):
        self.browser_loader = BrowserSafeLoader()
        self.resource_manager = get_resource_manager()
        
        # Default loading tasks
        self._register_default_tasks()
        
        logger.info("ProgressiveLoader initialized")
    
    def _register_default_tasks(self) -> None:
        """Register default loading tasks for TranscrevAI"""
        
        # Phase 1: Essential Services (highest priority, browser-safe)
        self.browser_loader.register_task(LoadingTask(
            name="core_directories",
            phase=LoadingPhase.ESSENTIAL,
            priority=1,
            estimated_time=0.1,
            memory_required=5,
            callback=self._create_core_directories,
            critical=True
        ))
        
        self.browser_loader.register_task(LoadingTask(
            name="logging_setup",
            phase=LoadingPhase.ESSENTIAL,
            priority=2,
            estimated_time=0.2,
            memory_required=10,
            callback=self._initialize_logging,
            critical=True
        ))
        
        self.browser_loader.register_task(LoadingTask(
            name="resource_monitoring",
            phase=LoadingPhase.ESSENTIAL,
            priority=3,
            estimated_time=0.3,
            memory_required=20,
            callback=self._start_resource_monitoring,
            dependencies=["logging_setup"],
            critical=True
        ))
        
        # Phase 2: Model Loading (background, non-blocking)
        self.browser_loader.register_task(LoadingTask(
            name="model_cache_init",
            phase=LoadingPhase.MODELS,
            priority=1,
            estimated_time=0.5,
            memory_required=50,
            callback=self._initialize_model_cache,
            dependencies=["resource_monitoring"]
        ))
        
        self.browser_loader.register_task(LoadingTask(
            name="whisper_model_preload",
            phase=LoadingPhase.MODELS,
            priority=2,
            estimated_time=5.0,  # Can be slow
            memory_required=800,
            callback=self._preload_whisper_model,
            dependencies=["model_cache_init"]
        ))
        
        # Phase 3: Processing Engines (lightweight initialization)
        self.browser_loader.register_task(LoadingTask(
            name="audio_processor_init",
            phase=LoadingPhase.PROCESSING,
            priority=1,
            estimated_time=0.3,
            memory_required=30,
            callback=self._initialize_audio_processor,
            dependencies=["resource_monitoring"]
        ))
        
        self.browser_loader.register_task(LoadingTask(
            name="transcription_engine_init",
            phase=LoadingPhase.PROCESSING,
            priority=2,
            estimated_time=0.4,
            memory_required=40,
            callback=self._initialize_transcription_engine,
            dependencies=["model_cache_init"]
        ))
        
        self.browser_loader.register_task(LoadingTask(
            name="diarization_engine_init",
            phase=LoadingPhase.PROCESSING,
            priority=3,
            estimated_time=0.5,
            memory_required=60,
            callback=self._initialize_diarization_engine,
            dependencies=["audio_processor_init"]
        ))
        
        self.browser_loader.register_task(LoadingTask(
            name="subtitle_generator_init",
            phase=LoadingPhase.PROCESSING,
            priority=4,
            estimated_time=0.2,
            memory_required=20,
            callback=self._initialize_subtitle_generator,
            dependencies=["transcription_engine_init"]
        ))
        
        # Phase 4: Optimizations (performance tuning)
        self.browser_loader.register_task(LoadingTask(
            name="hardware_optimization",
            phase=LoadingPhase.OPTIMIZATION,
            priority=1,
            estimated_time=0.3,
            memory_required=10,
            callback=self._apply_hardware_optimizations
        ))
        
        self.browser_loader.register_task(LoadingTask(
            name="memory_optimization",
            phase=LoadingPhase.OPTIMIZATION,
            priority=2,
            estimated_time=0.2,
            memory_required=5,
            callback=self._apply_memory_optimizations
        ))
    
    async def load_essential_services(self) -> Dict[str, Any]:
        """Load only essential services for immediate UI availability"""
        return await self.browser_loader.load_progressively([LoadingPhase.ESSENTIAL])
    
    async def load_all_phases(self) -> Dict[str, Any]:
        """Load all phases progressively"""
        return await self.browser_loader.load_progressively()
    
    async def load_up_to_phase(self, target_phase: LoadingPhase) -> Dict[str, Any]:
        """Load up to a specific phase"""
        phases = []
        for phase in LoadingPhase:
            phases.append(phase)
            if phase == target_phase:
                break
        
        return await self.browser_loader.load_progressively(phases)
    
    def add_progress_callback(self, callback: Callable[[float, str], None]) -> None:
        """Add progress callback for UI updates"""
        self.browser_loader.add_progress_callback(callback)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current loading status"""
        return self.browser_loader.get_loading_status()
    
    # Task implementation methods
    async def _create_core_directories(self) -> bool:
        """Create essential directories"""
        try:
            directories = [
                CONFIG["paths"]["data_dir"],
                CONFIG["paths"]["temp_dir"],
                CONFIG["paths"]["output_dir"],
                CONFIG["paths"]["recordings_dir"],
                CONFIG["paths"]["cache_dir"]
            ]
            
            for directory in directories:
                Path(directory).mkdir(parents=True, exist_ok=True)
            
            logger.info("Core directories created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create core directories: {e}")
            return False
    
    async def _initialize_logging(self) -> bool:
        """Initialize enhanced logging"""
        try:
            # Logging is already initialized by logging_setup module
            logger.info("Enhanced logging initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize logging: {e}")
            return False
    
    async def _start_resource_monitoring(self) -> bool:
        """Start resource monitoring"""
        try:
            await self.resource_manager.start_monitoring()
            logger.info("Resource monitoring started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start resource monitoring: {e}")
            return False
    
    async def _initialize_model_cache(self) -> bool:
        """Initialize model cache system"""
        try:
            from model_cache import get_model_cache
            cache = get_model_cache()
            
            # The cache is initialized on first access
            stats = cache.get_cache_stats()
            logger.info(f"Model cache initialized: {stats['cached_models_count']} models cached")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize model cache: {e}")
            return False
    
    async def _preload_whisper_model(self) -> bool:
        """Preload Whisper model in background"""
        try:
            from model_cache import preload_whisper_model
            
            # This is non-blocking and happens in background
            success = await preload_whisper_model("medium")
            if success:
                logger.info("Whisper model preloaded successfully")
            else:
                logger.warning("Whisper model preload failed, will load on demand")
            
            return True  # Non-critical, don't fail loading
            
        except Exception as e:
            logger.warning(f"Whisper model preload failed: {e}")
            return True  # Non-critical
    
    async def _initialize_audio_processor(self) -> bool:
        """Initialize audio processor"""
        try:
            # AudioProcessor is initialized on demand
            logger.info("Audio processor ready for initialization")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize audio processor: {e}")
            return False
    
    async def _initialize_transcription_engine(self) -> bool:
        """Initialize transcription engine"""
        try:
            # TranscriptionEngine is initialized on demand
            logger.info("Transcription engine ready for initialization")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize transcription engine: {e}")
            return False
    
    async def _initialize_diarization_engine(self) -> bool:
        """Initialize diarization engine"""
        try:
            # SpeakerDiarization is initialized on demand
            logger.info("Diarization engine ready for initialization")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize diarization engine: {e}")
            return False
    
    async def _initialize_subtitle_generator(self) -> bool:
        """Initialize subtitle generator"""
        try:
            # SubtitleGenerator is initialized on demand
            logger.info("Subtitle generator ready for initialization")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize subtitle generator: {e}")
            return False
    
    async def _apply_hardware_optimizations(self) -> bool:
        """Apply hardware-specific optimizations"""
        try:
            # CPU optimizations
            cpu_cores = CONFIG["hardware"]["cpu_cores"]
            
            # Set optimal thread counts
            os.environ["OMP_NUM_THREADS"] = str(min(4, cpu_cores))
            os.environ["OPENBLAS_NUM_THREADS"] = str(min(2, cpu_cores))
            os.environ["MKL_NUM_THREADS"] = str(min(4, cpu_cores))
            
            # Memory optimizations
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Disable for CPU-only
            
            logger.info(f"Hardware optimizations applied for {cpu_cores} cores")
            return True
            
        except Exception as e:
            logger.warning(f"Hardware optimization failed: {e}")
            return True  # Non-critical
    
    async def _apply_memory_optimizations(self) -> bool:
        """Apply memory optimizations"""
        try:
            # Force garbage collection
            gc.collect()
            
            # Set Python optimization flags
            os.environ["PYTHONOPTIMIZE"] = "1"
            
            logger.info("Memory optimizations applied")
            return True
            
        except Exception as e:
            logger.warning(f"Memory optimization failed: {e}")
            return True  # Non-critical


# Global instance for easy access
_global_progressive_loader: Optional[ProgressiveLoader] = None


def get_progressive_loader() -> ProgressiveLoader:
    """Get global progressive loader instance"""
    global _global_progressive_loader
    
    if _global_progressive_loader is None:
        _global_progressive_loader = ProgressiveLoader()
    
    return _global_progressive_loader


# Convenience functions for external use
async def load_essential() -> Dict[str, Any]:
    """Load essential services only"""
    loader = get_progressive_loader()
    return await loader.load_essential_services()


async def load_everything() -> Dict[str, Any]:
    """Load all components progressively"""
    loader = get_progressive_loader()
    return await loader.load_all_phases()


async def load_with_progress(progress_callback: Callable[[float, str], None]) -> Dict[str, Any]:
    """Load with progress updates"""
    loader = get_progressive_loader()
    loader.add_progress_callback(progress_callback)
    return await loader.load_all_phases()