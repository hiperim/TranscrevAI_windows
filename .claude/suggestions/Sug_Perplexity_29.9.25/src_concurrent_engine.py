"""
TranscrevAI Optimized - Concurrent Engine Module
Sistema de processamento concorrente otimizado para CPU com coordenação browser-safe
"""

import asyncio
import gc
import multiprocessing as mp
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
import psutil

# Import our optimized modules
from logging_setup import get_logger, log_performance, log_resource_usage
from resource_manager import get_resource_manager, ResourceStatus
from config import CONFIG

logger = get_logger("transcrevai.concurrent_engine")


class ProcessingMode(Enum):
    """Processing modes for concurrent engine"""
    SINGLE_THREAD = "single_thread"
    MULTI_THREAD = "multi_thread"
    MULTI_PROCESS = "multi_process"
    HYBRID = "hybrid"


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ProcessingTask:
    """Individual processing task definition"""
    task_id: str
    function: Callable
    args: tuple
    kwargs: dict
    priority: TaskPriority
    estimated_duration: float
    memory_required: float  # MB
    cpu_intensive: bool = True
    dependencies: List[str] = None
    callback: Optional[Callable] = None
    timeout: Optional[float] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class ProcessingResult:
    """Processing task result"""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    duration: float = 0.0
    memory_used: float = 0.0
    cpu_usage: float = 0.0


class BrowserSafeConcurrency:
    """
    Browser-safe concurrent processing that prevents UI freezing
    Implements smart scheduling and resource management
    """
    
    def __init__(self):
        self.resource_manager = get_resource_manager()
        
        # Processing configuration
        self.max_workers = min(4, max(1, mp.cpu_count() - 1))
        self.max_memory_per_worker = CONFIG["hardware"]["memory_per_worker_mb"]
        
        # Executors (initialized lazily)
        self._thread_executor: Optional[ThreadPoolExecutor] = None
        self._process_executor: Optional[ProcessPoolExecutor] = None
        
        # Task management
        self.pending_tasks: Dict[str, ProcessingTask] = {}
        self.running_tasks: Dict[str, ProcessingTask] = {}
        self.completed_tasks: Dict[str, ProcessingResult] = {}
        self.task_queue = asyncio.PriorityQueue()
        
        # Monitoring
        self.processing_active = False
        self.worker_stats: Dict[str, Dict] = {}
        
        # Browser safety
        self.max_blocking_time = 100  # ms
        self.yield_interval = 0.05  # seconds
        
        logger.info(f"BrowserSafeConcurrency initialized with {self.max_workers} max workers")
    
    async def submit_task(self, task: ProcessingTask) -> str:
        """
        Submit a task for concurrent processing
        
        Args:
            task: ProcessingTask to execute
            
        Returns:
            str: Task ID for tracking
        """
        try:
            # Validate task
            if not task.task_id:
                task.task_id = f"task_{int(time.time() * 1000)}_{id(task)}"
            
            # Check resource availability
            if not await self._check_resource_availability(task):
                raise RuntimeError(f"Insufficient resources for task {task.task_id}")
            
            # Add to pending tasks
            self.pending_tasks[task.task_id] = task
            
            # Add to priority queue
            await self.task_queue.put((
                -task.priority.value,  # Negative for priority queue (higher priority = lower number)
                time.time(),  # Submission time for tie-breaking
                task
            ))
            
            logger.info(f"Task {task.task_id} submitted for processing")
            
            # Start processing if not already active
            if not self.processing_active:
                asyncio.create_task(self._processing_loop())
            
            return task.task_id
            
        except Exception as e:
            logger.error(f"Failed to submit task: {e}")
            raise
    
    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> ProcessingResult:
        """
        Wait for a specific task to complete
        
        Args:
            task_id: ID of task to wait for
            timeout: Maximum time to wait in seconds
            
        Returns:
            ProcessingResult: Result of the task
        """
        start_time = time.time()
        
        while task_id not in self.completed_tasks:
            if timeout and (time.time() - start_time) > timeout:
                raise asyncio.TimeoutError(f"Task {task_id} did not complete within {timeout}s")
            
            # Browser-safe waiting
            await asyncio.sleep(0.1)
        
        return self.completed_tasks[task_id]
    
    async def wait_for_all_tasks(self, timeout: Optional[float] = None) -> Dict[str, ProcessingResult]:
        """
        Wait for all pending and running tasks to complete
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            Dict mapping task IDs to results
        """
        start_time = time.time()
        
        while self.pending_tasks or self.running_tasks:
            if timeout and (time.time() - start_time) > timeout:
                raise asyncio.TimeoutError(f"Not all tasks completed within {timeout}s")
            
            # Browser-safe waiting
            await asyncio.sleep(0.2)
        
        return self.completed_tasks.copy()
    
    async def _processing_loop(self):
        """Main processing loop for concurrent tasks"""
        if self.processing_active:
            return
        
        self.processing_active = True
        logger.info("Started concurrent processing loop")
        
        try:
            while not self.task_queue.empty() or self.running_tasks:
                try:
                    # Process available tasks
                    await self._process_available_tasks()
                    
                    # Check for completed tasks
                    await self._check_completed_tasks()
                    
                    # Browser-safe yielding
                    await asyncio.sleep(self.yield_interval)
                    
                except asyncio.CancelledError:
                    logger.info("Processing loop cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error in processing loop: {e}")
                    await asyncio.sleep(1)  # Brief pause on error
                    
        except Exception as e:
            logger.error(f"Processing loop failed: {e}")
        finally:
            self.processing_active = False
            logger.info("Concurrent processing loop stopped")
    
    async def _process_available_tasks(self):
        """Process tasks that can be started"""
        while not self.task_queue.empty() and len(self.running_tasks) < self.max_workers:
            try:
                # Get next task (non-blocking)
                try:
                    priority, submission_time, task = self.task_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                
                # Check if task can be started
                if await self._can_start_task(task):
                    await self._start_task(task)
                else:
                    # Put task back in queue
                    await self.task_queue.put((priority, submission_time, task))
                    break  # Try again later
                    
            except Exception as e:
                logger.error(f"Error processing available tasks: {e}")
    
    async def _can_start_task(self, task: ProcessingTask) -> bool:
        """Check if task can be started now"""
        try:
            # Check dependencies
            for dep_id in task.dependencies:
                if dep_id not in self.completed_tasks:
                    return False
                if not self.completed_tasks[dep_id].success:
                    logger.warning(f"Task {task.task_id} dependency {dep_id} failed")
                    return False
            
            # Check resource availability
            if not await self._check_resource_availability(task):
                return False
            
            # Check memory pressure
            if self.resource_manager.is_memory_pressure_high():
                logger.debug(f"High memory pressure, delaying task {task.task_id}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking task readiness: {e}")
            return False
    
    async def _start_task(self, task: ProcessingTask):
        """Start executing a task"""
        try:
            # Move to running tasks
            self.running_tasks[task.task_id] = task
            if task.task_id in self.pending_tasks:
                del self.pending_tasks[task.task_id]
            
            # Reserve resources
            if not self.resource_manager.reserve_memory(
                f"task_{task.task_id}",
                task.memory_required,
                "concurrent_processing"
            ):
                logger.warning(f"Could not reserve memory for task {task.task_id}")
            
            # Choose processing mode based on task characteristics
            processing_mode = self._determine_processing_mode(task)
            
            # Start task execution
            if processing_mode == ProcessingMode.MULTI_PROCESS:
                future = await self._execute_in_process(task)
            else:
                future = await self._execute_in_thread(task)
            
            # Store future for monitoring
            self.worker_stats[task.task_id] = {
                "start_time": time.time(),
                "future": future,
                "mode": processing_mode.value,
                "memory_reserved": task.memory_required
            }
            
            logger.debug(f"Started task {task.task_id} in {processing_mode.value} mode")
            
        except Exception as e:
            logger.error(f"Failed to start task {task.task_id}: {e}")
            # Mark as failed and clean up
            await self._complete_task(task.task_id, False, None, str(e))
    
    async def _execute_in_thread(self, task: ProcessingTask) -> asyncio.Future:
        """Execute task in thread pool"""
        if self._thread_executor is None:
            self._thread_executor = ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="TranscrevAI-"
            )
        
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(
            self._thread_executor,
            self._execute_task_wrapper,
            task
        )
        
        return future
    
    async def _execute_in_process(self, task: ProcessingTask) -> asyncio.Future:
        """Execute task in process pool"""
        if self._process_executor is None:
            # Configure process pool
            ctx = mp.get_context('spawn')  # Use spawn for cross-platform compatibility
            self._process_executor = ProcessPoolExecutor(
                max_workers=min(2, self.max_workers),  # Fewer processes than threads
                mp_context=ctx
            )
        
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(
            self._process_executor,
            _execute_task_in_process,  # Global function for pickling
            task.function,
            task.args,
            task.kwargs
        )
        
        return future
    
    def _execute_task_wrapper(self, task: ProcessingTask) -> Any:
        """Wrapper for task execution with monitoring"""
        start_time = time.time()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)
        
        try:
            # Execute task
            if task.timeout:
                # This is a simplified timeout - in production, use more robust timeout mechanism
                result = task.function(*task.args, **task.kwargs)
            else:
                result = task.function(*task.args, **task.kwargs)
            
            # Calculate metrics
            final_memory = process.memory_info().rss / (1024 * 1024)
            duration = time.time() - start_time
            memory_used = final_memory - initial_memory
            
            return {
                "success": True,
                "result": result,
                "duration": duration,
                "memory_used": memory_used,
                "cpu_usage": 0.0  # Would need more sophisticated monitoring
            }
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Task {task.task_id} failed: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "duration": duration,
                "memory_used": 0.0,
                "cpu_usage": 0.0
            }
    
    async def _check_completed_tasks(self):
        """Check for completed tasks and process results"""
        completed_task_ids = []
        
        for task_id, stats in self.worker_stats.items():
            future = stats["future"]
            
            if future.done():
                try:
                    result = future.result()
                    
                    if isinstance(result, dict) and "success" in result:
                        success = result["success"]
                        task_result = result.get("result")
                        error = result.get("error")
                        duration = result.get("duration", 0.0)
                        memory_used = result.get("memory_used", 0.0)
                        cpu_usage = result.get("cpu_usage", 0.0)
                    else:
                        # Simple result
                        success = True
                        task_result = result
                        error = None
                        duration = time.time() - stats["start_time"]
                        memory_used = 0.0
                        cpu_usage = 0.0
                    
                    await self._complete_task(task_id, success, task_result, error, duration, memory_used, cpu_usage)
                    
                except Exception as e:
                    logger.error(f"Error getting result for task {task_id}: {e}")
                    await self._complete_task(task_id, False, None, str(e))
                
                completed_task_ids.append(task_id)
        
        # Clean up completed tasks
        for task_id in completed_task_ids:
            if task_id in self.worker_stats:
                del self.worker_stats[task_id]
    
    async def _complete_task(self,
                           task_id: str,
                           success: bool,
                           result: Any = None,
                           error: Optional[str] = None,
                           duration: float = 0.0,
                           memory_used: float = 0.0,
                           cpu_usage: float = 0.0):
        """Mark task as completed and clean up resources"""
        try:
            # Create result
            task_result = ProcessingResult(
                task_id=task_id,
                success=success,
                result=result,
                error=error,
                duration=duration,
                memory_used=memory_used,
                cpu_usage=cpu_usage
            )
            
            # Store result
            self.completed_tasks[task_id] = task_result
            
            # Remove from running tasks
            if task_id in self.running_tasks:
                task = self.running_tasks[task_id]
                del self.running_tasks[task_id]
                
                # Call completion callback if provided
                if task.callback:
                    try:
                        if asyncio.iscoroutinefunction(task.callback):
                            await task.callback(task_result)
                        else:
                            task.callback(task_result)
                    except Exception as e:
                        logger.warning(f"Task callback failed: {e}")
            
            # Release reserved memory
            self.resource_manager.release_memory_reservation(f"task_{task_id}")
            
            # Log completion
            status = "completed" if success else "failed"
            logger.info(f"Task {task_id} {status} in {duration:.2f}s")
            
            if success:
                log_performance(
                    f"Task {task_id} completed",
                    duration=duration,
                    memory_used_mb=memory_used,
                    cpu_usage_percent=cpu_usage
                )
            
        except Exception as e:
            logger.error(f"Error completing task {task_id}: {e}")
    
    def _determine_processing_mode(self, task: ProcessingTask) -> ProcessingMode:
        """Determine best processing mode for task"""
        # Simple heuristics for mode selection
        if task.cpu_intensive and task.memory_required > 100:
            return ProcessingMode.MULTI_PROCESS
        else:
            return ProcessingMode.MULTI_THREAD
    
    async def _check_resource_availability(self, task: ProcessingTask) -> bool:
        """Check if resources are available for task"""
        try:
            # Check memory
            if not self.resource_manager.can_allocate(task.memory_required):
                return False
            
            # Check CPU (simplified check)
            cpu_usage = psutil.cpu_percent(interval=0.1)
            if cpu_usage > 90:  # System is overloaded
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error checking resource availability: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get concurrent processing statistics"""
        return {
            "max_workers": self.max_workers,
            "pending_tasks": len(self.pending_tasks),
            "running_tasks": len(self.running_tasks),
            "completed_tasks": len(self.completed_tasks),
            "processing_active": self.processing_active,
            "memory_per_worker_mb": self.max_memory_per_worker,
            "worker_stats": {
                task_id: {
                    "mode": stats["mode"],
                    "memory_reserved": stats["memory_reserved"],
                    "running_time": time.time() - stats["start_time"]
                }
                for task_id, stats in self.worker_stats.items()
            }
        }
    
    async def shutdown(self):
        """Shutdown concurrent engine and clean up resources"""
        try:
            logger.info("Shutting down concurrent engine...")
            
            # Cancel processing loop
            self.processing_active = False
            
            # Wait for running tasks to complete (with timeout)
            if self.running_tasks:
                logger.info(f"Waiting for {len(self.running_tasks)} running tasks to complete...")
                try:
                    await asyncio.wait_for(self.wait_for_all_tasks(), timeout=30.0)
                except asyncio.TimeoutError:
                    logger.warning("Some tasks did not complete within timeout")
            
            # Shutdown executors
            if self._thread_executor:
                self._thread_executor.shutdown(wait=True)
                self._thread_executor = None
            
            if self._process_executor:
                self._process_executor.shutdown(wait=True)
                self._process_executor = None
            
            logger.info("Concurrent engine shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during concurrent engine shutdown: {e}")


# Global function for process pool (must be at module level for pickling)
def _execute_task_in_process(func: Callable, args: tuple, kwargs: dict) -> Any:
    """Execute a task in a separate process"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Process task failed: {e}")
        raise


class ConcurrentEngine:
    """
    Main concurrent engine that provides high-level interface for concurrent processing
    """
    
    def __init__(self):
        self.browser_safe_concurrency = BrowserSafeConcurrency()
        self.resource_manager = get_resource_manager()
        
        # Processing statistics
        self.total_tasks_submitted = 0
        self.total_tasks_completed = 0
        self.total_processing_time = 0.0
        
        logger.info("ConcurrentEngine initialized")
    
    async def process_audio_concurrent(self,
                                     audio_files: List[str],
                                     processing_function: Callable,
                                     max_concurrent: Optional[int] = None,
                                     progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Process multiple audio files concurrently
        
        Args:
            audio_files: List of audio file paths
            processing_function: Function to process each file
            max_concurrent: Maximum concurrent tasks (None for auto)
            progress_callback: Optional progress callback
            
        Returns:
            Dict with processing results
        """
        start_time = time.time()
        
        try:
            if not audio_files:
                return {"success": False, "error": "No audio files provided"}
            
            if max_concurrent is None:
                max_concurrent = min(len(audio_files), self.browser_safe_concurrency.max_workers)
            
            logger.info(f"Processing {len(audio_files)} audio files with {max_concurrent} max concurrent")
            
            # Create processing tasks
            tasks = []
            for i, audio_file in enumerate(audio_files):
                task = ProcessingTask(
                    task_id=f"audio_process_{i}_{Path(audio_file).stem}",
                    function=processing_function,
                    args=(audio_file,),
                    kwargs={},
                    priority=TaskPriority.NORMAL,
                    estimated_duration=30.0,  # Conservative estimate
                    memory_required=200.0,  # Conservative estimate
                    cpu_intensive=True
                )
                tasks.append(task)
            
            # Submit tasks
            task_ids = []
            for task in tasks:
                task_id = await self.browser_safe_concurrency.submit_task(task)
                task_ids.append(task_id)
                self.total_tasks_submitted += 1
            
            # Wait for completion with progress updates
            results = {}
            completed_count = 0
            
            while completed_count < len(task_ids):
                await asyncio.sleep(0.5)  # Check every 500ms
                
                # Count completed tasks
                new_completed_count = sum(
                    1 for tid in task_ids 
                    if tid in self.browser_safe_concurrency.completed_tasks
                )
                
                if new_completed_count > completed_count:
                    completed_count = new_completed_count
                    
                    # Update progress
                    if progress_callback:
                        progress = (completed_count / len(task_ids)) * 100
                        await progress_callback(
                            progress,
                            f"Processados {completed_count}/{len(task_ids)} arquivos"
                        )
            
            # Collect results
            for task_id in task_ids:
                if task_id in self.browser_safe_concurrency.completed_tasks:
                    task_result = self.browser_safe_concurrency.completed_tasks[task_id]
                    results[task_id] = task_result
                    
                    if task_result.success:
                        self.total_tasks_completed += 1
                        self.total_processing_time += task_result.duration
            
            # Calculate statistics
            processing_duration = time.time() - start_time
            successful_tasks = sum(1 for r in results.values() if r.success)
            failed_tasks = len(results) - successful_tasks
            
            logger.info(f"Concurrent processing completed: {successful_tasks}/{len(task_ids)} successful")
            
            return {
                "success": True,
                "total_files": len(audio_files),
                "successful_tasks": successful_tasks,
                "failed_tasks": failed_tasks,
                "processing_duration": processing_duration,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Concurrent audio processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "total_files": len(audio_files) if audio_files else 0
            }
    
    async def process_with_pipeline(self,
                                  data_items: List[Any],
                                  pipeline_stages: List[Callable],
                                  stage_names: Optional[List[str]] = None,
                                  progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Process data through a pipeline of stages concurrently
        
        Args:
            data_items: Items to process
            pipeline_stages: List of processing functions (stages)
            stage_names: Optional names for stages
            progress_callback: Optional progress callback
            
        Returns:
            Dict with pipeline results
        """
        try:
            if not data_items or not pipeline_stages:
                return {"success": False, "error": "No data or pipeline stages provided"}
            
            if stage_names is None:
                stage_names = [f"stage_{i}" for i in range(len(pipeline_stages))]
            
            logger.info(f"Processing {len(data_items)} items through {len(pipeline_stages)} stages")
            
            current_data = data_items
            stage_results = {}
            
            for stage_idx, (stage_func, stage_name) in enumerate(zip(pipeline_stages, stage_names)):
                stage_start = time.time()
                
                if progress_callback:
                    await progress_callback(
                        (stage_idx / len(pipeline_stages)) * 100,
                        f"Executando {stage_name}..."
                    )
                
                # Create tasks for current stage
                tasks = []
                for item_idx, data_item in enumerate(current_data):
                    task = ProcessingTask(
                        task_id=f"{stage_name}_{item_idx}",
                        function=stage_func,
                        args=(data_item,),
                        kwargs={},
                        priority=TaskPriority.NORMAL,
                        estimated_duration=10.0,
                        memory_required=100.0,
                        cpu_intensive=True
                    )
                    tasks.append(task)
                
                # Submit and wait for stage completion
                task_ids = []
                for task in tasks:
                    task_id = await self.browser_safe_concurrency.submit_task(task)
                    task_ids.append(task_id)
                
                # Wait for all tasks in this stage to complete
                stage_data = []
                for task_id in task_ids:
                    result = await self.browser_safe_concurrency.wait_for_task(task_id)
                    if result.success:
                        stage_data.append(result.result)
                    else:
                        logger.error(f"Stage {stage_name} task failed: {result.error}")
                        stage_data.append(None)  # Placeholder for failed task
                
                # Update data for next stage
                current_data = [item for item in stage_data if item is not None]
                
                stage_duration = time.time() - stage_start
                stage_results[stage_name] = {
                    "duration": stage_duration,
                    "input_count": len(current_data),
                    "output_count": len(stage_data),
                    "success_rate": len(current_data) / len(stage_data) if stage_data else 0
                }
                
                logger.info(f"Stage {stage_name} completed: {len(current_data)}/{len(stage_data)} successful")
            
            if progress_callback:
                await progress_callback(100, "Pipeline concluída!")
            
            return {
                "success": True,
                "final_results": current_data,
                "stage_results": stage_results,
                "original_count": len(data_items),
                "final_count": len(current_data)
            }
            
        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """Get comprehensive engine statistics"""
        concurrency_stats = self.browser_safe_concurrency.get_stats()
        
        return {
            "concurrent_engine": {
                "total_tasks_submitted": self.total_tasks_submitted,
                "total_tasks_completed": self.total_tasks_completed,
                "total_processing_time": self.total_processing_time,
                "average_task_duration": (
                    self.total_processing_time / max(self.total_tasks_completed, 1)
                ),
                "success_rate": (
                    self.total_tasks_completed / max(self.total_tasks_submitted, 1)
                )
            },
            "concurrency_stats": concurrency_stats,
            "resource_status": self.resource_manager.get_current_metrics().__dict__
        }
    
    async def shutdown(self):
        """Shutdown the concurrent engine"""
        await self.browser_safe_concurrency.shutdown()


# Global instance for easy access
_global_concurrent_engine: Optional[ConcurrentEngine] = None


def get_concurrent_engine() -> ConcurrentEngine:
    """Get global concurrent engine instance"""
    global _global_concurrent_engine
    
    if _global_concurrent_engine is None:
        _global_concurrent_engine = ConcurrentEngine()
    
    return _global_concurrent_engine


# Convenience functions for external use
async def process_files_concurrent(files: List[str],
                                 processing_func: Callable,
                                 max_concurrent: Optional[int] = None) -> Dict[str, Any]:
    """Process files concurrently"""
    engine = get_concurrent_engine()
    return await engine.process_audio_concurrent(files, processing_func, max_concurrent)


async def run_pipeline_concurrent(data: List[Any],
                                stages: List[Callable],
                                stage_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Run data through concurrent pipeline"""
    engine = get_concurrent_engine()
    return await engine.process_with_pipeline(data, stages, stage_names)


def get_concurrent_stats() -> Dict[str, Any]:
    """Get concurrent processing statistics"""
    engine = get_concurrent_engine()
    return engine.get_engine_stats()