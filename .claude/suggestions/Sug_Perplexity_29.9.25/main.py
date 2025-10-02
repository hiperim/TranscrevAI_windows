"""
TranscrevAI Optimized - Main Application
Sistema de transcrição e diarização PT-BR com arquitetura browser-safe e performance otimizada
"""

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# Import our optimized modules
from config import CONFIG, PT_BR_CONFIG
from logging_setup import setup_app_logging, get_logger, log_performance, log_resource_usage
from resource_manager import get_resource_manager, ResourceStatus
from model_cache import get_model_cache, preload_whisper_model

# Import processing modules (will be created)
try:
    from audio_processing import AudioProcessor
    from transcription import TranscriptionEngine
    from speaker_diarization import SpeakerDiarization
    from subtitle_generator import SubtitleGenerator
    from progressive_loader import ProgressiveLoader
    from memory_optimizer import MemoryOptimizer
except ImportError as e:
    logger = logging.getLogger("transcrevai.main")
    logger.error(f"Failed to import processing modules: {e}")
    # We'll create these modules next


# Setup logging
logger = setup_app_logging("transcrevai.main")

# Global managers
resource_manager = get_resource_manager()
model_cache = get_model_cache()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager with progressive loading"""
    logger.info("=" * 60)
    logger.info("🚀 TranscrevAI Optimized Starting Up")
    logger.info("=" * 60)
    
    startup_start = time.time()
    
    try:
        # Phase 1: Initialize core systems (browser-safe)
        logger.info("📋 Phase 1: Initializing core systems...")
        
        # Start resource monitoring
        await resource_manager.start_monitoring()
        
        # Initialize progressive loader
        progressive_loader = ProgressiveLoader()
        
        # Phase 1a: Essential services (non-blocking)
        await progressive_loader.load_essential_services()
        
        # Phase 1b: Background model preloading
        logger.info("📥 Starting background model preloading...")
        preload_task = asyncio.create_task(
            preload_whisper_model("medium")
        )
        
        # Don't wait for preload - it's background
        app.state.preload_task = preload_task
        
        # Phase 2: Initialize processing engines (lightweight)
        logger.info("🔧 Phase 2: Initializing processing engines...")
        
        try:
            app.state.audio_processor = AudioProcessor()
            app.state.transcription_engine = TranscriptionEngine()
            app.state.diarization_engine = SpeakerDiarization()
            app.state.subtitle_generator = SubtitleGenerator()
            app.state.memory_optimizer = MemoryOptimizer()
        except Exception as e:
            logger.warning(f"Some processing engines failed to initialize: {e}")
            # Continue with fallback implementations
        
        # Phase 3: Setup WebSocket management
        logger.info("🌐 Phase 3: Setting up WebSocket management...")
        app.state.websocket_manager = WebSocketManager()
        app.state.session_manager = SessionManager()
        
        # Phase 4: Final optimizations
        logger.info("⚡ Phase 4: Applying final optimizations...")
        
        # Apply hardware optimizations
        await apply_hardware_optimizations()
        
        startup_duration = time.time() - startup_start
        
        # Log startup metrics
        log_performance(
            "Application startup complete",
            duration=startup_duration,
            target_time=CONFIG["performance"]["targets"]["max_startup_time"],
            success=startup_duration < CONFIG["performance"]["targets"]["max_startup_time"]
        )
        
        logger.info("=" * 60)
        logger.info(f"✅ TranscrevAI Optimized Ready! ({startup_duration:.2f}s)")
        logger.info(f"🌍 Server: http://{CONFIG['server']['host']}:{CONFIG['server']['port']}")
        logger.info(f"💾 Cache: {model_cache.get_cache_stats()['cached_models_count']} models cached")
        logger.info(f"🧠 Memory: {resource_manager.get_status_report()['memory']['usage_percent']:.1f}% used")
        logger.info("=" * 60)
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    finally:
        # Shutdown procedures
        logger.info("🔄 Shutting down TranscrevAI Optimized...")
        
        try:
            # Cancel background tasks
            if hasattr(app.state, 'preload_task'):
                app.state.preload_task.cancel()
            
            # Stop resource monitoring
            await resource_manager.stop_monitoring()
            
            # Cleanup sessions
            if hasattr(app.state, 'session_manager'):
                await app.state.session_manager.cleanup_all_sessions()
            
            # Clear model cache if needed
            if resource_manager.is_memory_pressure_high():
                await model_cache.clear_cache()
            
            logger.info("✅ Shutdown complete")
            
        except Exception as e:
            logger.error(f"Shutdown error: {e}")


# FastAPI app initialization
app = FastAPI(
    title="TranscrevAI Optimized",
    description="Sistema de transcrição e diarização PT-BR com arquitetura browser-safe",
    version="1.0.0",
    lifespan=lifespan
)

# Setup static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory=CONFIG["paths"]["templates_dir"])


class WebSocketManager:
    """Enhanced WebSocket manager with browser-safe features"""
    
    def __init__(self):
        self.connections: Dict[str, WebSocket] = {}
        self.connection_times: Dict[str, float] = {}
        
    async def connect(self, websocket: WebSocket, session_id: str) -> bool:
        """Connect WebSocket with browser-safe limits"""
        try:
            # Check connection limits
            if len(self.connections) >= CONFIG["browser_safety"]["max_concurrent_connections"]:
                await websocket.close(code=1013, reason="Too many connections")
                return False
            
            await websocket.accept()
            self.connections[session_id] = websocket
            self.connection_times[session_id] = time.time()
            
            logger.info(f"WebSocket connected: {session_id} (total: {len(self.connections)})")
            return True
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            return False
    
    async def disconnect(self, session_id: str) -> None:
        """Disconnect WebSocket and cleanup"""
        try:
            websocket = self.connections.pop(session_id, None)
            self.connection_times.pop(session_id, None)
            
            if websocket:
                try:
                    await websocket.close()
                except:
                    pass  # Connection might already be closed
            
            # Cleanup session
            if hasattr(app.state, 'session_manager'):
                await app.state.session_manager.cleanup_session(session_id)
            
            logger.info(f"WebSocket disconnected: {session_id}")
            
        except Exception as e:
            logger.error(f"WebSocket disconnect error: {e}")
    
    async def send_message(self, session_id: str, message: Dict[str, Any]) -> bool:
        """Send message with size limits and error handling"""
        websocket = self.connections.get(session_id)
        if not websocket:
            return False
        
        try:
            # Check message size
            message_str = json.dumps(message)
            if len(message_str.encode()) > CONFIG["browser_safety"]["max_message_size_mb"] * 1024 * 1024:
                logger.warning(f"Message too large for {session_id}: {len(message_str)} bytes")
                return False
            
            await websocket.send_json(message)
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message to {session_id}: {e}")
            await self.disconnect(session_id)
            return False
    
    async def broadcast_status(self, status_type: str, data: Any) -> None:
        """Broadcast status to all connections"""
        message = {
            "type": status_type,
            "data": data,
            "timestamp": time.time()
        }
        
        # Send to all connections
        disconnected = []
        for session_id in list(self.connections.keys()):
            if not await self.send_message(session_id, message):
                disconnected.append(session_id)
        
        # Cleanup disconnected sessions
        for session_id in disconnected:
            await self.disconnect(session_id)


class SessionManager:
    """Manage user sessions with resource tracking"""
    
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        
    def create_session(self, session_id: str) -> Dict[str, Any]:
        """Create new session with PT-BR defaults"""
        session = {
            "id": session_id,
            "created_at": time.time(),
            "status": "created",
            "language": "pt",  # PT-BR fixed
            "model": "medium",  # PT-BR optimized model
            "recording": False,
            "processing": False,
            "progress": {
                "transcription": 0,
                "diarization": 0,
                "total": 0
            },
            "files": {
                "audio": None,
                "srt": None
            },
            "results": {
                "transcription": [],
                "diarization": [],
                "speakers_detected": 0
            },
            "metrics": {
                "processing_time": 0,
                "accuracy": 0,
                "memory_usage": 0
            }
        }
        
        self.sessions[session_id] = session
        logger.info(f"Session created: {session_id}")
        
        return session
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID"""
        return self.sessions.get(session_id)
    
    def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """Update session data"""
        if session_id in self.sessions:
            self.sessions[session_id].update(updates)
            return True
        return False
    
    async def cleanup_session(self, session_id: str) -> None:
        """Cleanup session resources"""
        session = self.sessions.get(session_id)
        if not session:
            return
        
        try:
            # Cleanup any processing resources
            if session.get("processing"):
                # Cancel any running tasks
                if "task" in session:
                    session["task"].cancel()
            
            # Release memory reservations
            resource_manager.release_memory_reservation(f"session_{session_id}")
            
            # Remove session
            del self.sessions[session_id]
            logger.info(f"Session cleaned up: {session_id}")
            
        except Exception as e:
            logger.error(f"Session cleanup error: {e}")
    
    async def cleanup_all_sessions(self) -> None:
        """Cleanup all sessions"""
        session_ids = list(self.sessions.keys())
        for session_id in session_ids:
            await self.cleanup_session(session_id)


async def apply_hardware_optimizations() -> None:
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
        
    except Exception as e:
        logger.warning(f"Hardware optimization failed: {e}")


# ============================================================================
# API ROUTES
# ============================================================================

@app.get("/")
async def main_interface(request: Request):
    """Main application interface"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    resource_status = resource_manager.get_status_report()
    cache_stats = model_cache.get_cache_stats()
    
    return JSONResponse({
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": time.time(),
        "uptime_seconds": time.time() - app.state.startup_time if hasattr(app.state, 'startup_time') else 0,
        "features": [
            "pt-br-exclusive",
            "browser-safe-architecture", 
            "memory-pressure-management",
            "model-caching",
            "progressive-loading",
            "advanced-diarization"
        ],
        "resource_status": resource_status,
        "model_cache": {
            "hit_rate": cache_stats["hit_rate_percent"],
            "cached_models": cache_stats["cached_models_count"],
            "cache_size_mb": cache_stats["total_size_mb"]
        },
        "performance": {
            "target_processing_ratio_warm": CONFIG["performance"]["targets"]["processing_ratio_warm"],
            "target_processing_ratio_cold": CONFIG["performance"]["targets"]["processing_ratio_cold"]
        }
    })


@app.get("/api/config")
async def get_config():
    """Get public configuration"""
    return JSONResponse({
        "language": "pt",
        "model": "medium",
        "features": {
            "recording_formats": CONFIG["audio"]["recording_formats"],
            "supported_formats": CONFIG["audio"]["supported_formats"],
            "max_duration_minutes": CONFIG["audio"]["max_duration_minutes"],
            "max_file_size_mb": CONFIG["audio"]["max_file_size_mb"]
        },
        "limits": {
            "memory_warning_threshold": CONFIG["hardware"]["memory_thresholds"]["warning"] * 100,
            "max_concurrent_connections": CONFIG["browser_safety"]["max_concurrent_connections"]
        }
    })


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """Main WebSocket endpoint for real-time communication"""
    if not await app.state.websocket_manager.connect(websocket, session_id):
        return
    
    # Create session
    session = app.state.session_manager.create_session(session_id)
    
    try:
        # Send initial status
        await app.state.websocket_manager.send_message(session_id, {
            "type": "connected",
            "session_id": session_id,
            "config": {
                "language": "pt",
                "model": "medium",
                "browser_safe": True
            },
            "status": resource_manager.get_status_report()
        })
        
        # Main message loop
        while True:
            data = await websocket.receive_json()
            await handle_websocket_message(session_id, data)
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await app.state.websocket_manager.disconnect(session_id)


async def handle_websocket_message(session_id: str, data: Dict[str, Any]) -> None:
    """Handle incoming WebSocket messages"""
    message_type = data.get("type")
    message_data = data.get("data", {})
    
    session = app.state.session_manager.get_session(session_id)
    if not session:
        await app.state.websocket_manager.send_message(session_id, {
            "type": "error",
            "message": "Session not found"
        })
        return
    
    try:
        if message_type == "start_recording":
            await handle_start_recording(session_id, message_data)
        
        elif message_type == "stop_recording":
            await handle_stop_recording(session_id, message_data)
        
        elif message_type == "upload_audio":
            await handle_audio_upload(session_id, message_data)
        
        elif message_type == "get_status":
            await handle_get_status(session_id)
        
        elif message_type == "ping":
            await app.state.websocket_manager.send_message(session_id, {
                "type": "pong",
                "timestamp": time.time()
            })
        
        else:
            logger.warning(f"Unknown message type: {message_type}")
            await app.state.websocket_manager.send_message(session_id, {
                "type": "error", 
                "message": f"Unknown message type: {message_type}"
            })
    
    except Exception as e:
        logger.error(f"Message handling error: {e}")
        await app.state.websocket_manager.send_message(session_id, {
            "type": "error",
            "message": "Internal error processing request"
        })


async def handle_start_recording(session_id: str, data: Dict[str, Any]) -> None:
    """Handle start recording request"""
    session = app.state.session_manager.get_session(session_id)
    if not session:
        return
    
    # Check memory pressure
    if resource_manager.is_memory_pressure_high():
        await app.state.websocket_manager.send_message(session_id, {
            "type": "error",
            "message": "Sistema sob pressão de memória. Tente novamente em alguns segundos."
        })
        return
    
    # Reserve memory for recording
    if not resource_manager.reserve_memory(f"recording_{session_id}", 100, "audio_recording"):
        await app.state.websocket_manager.send_message(session_id, {
            "type": "error", 
            "message": "Não foi possível reservar memória para gravação"
        })
        return
    
    try:
        # Start recording using audio processor
        audio_processor = app.state.audio_processor
        recording_started = await audio_processor.start_recording(session_id)
        
        if recording_started:
            session["recording"] = True
            session["status"] = "recording"
            
            await app.state.websocket_manager.send_message(session_id, {
                "type": "recording_started",
                "message": "Gravação iniciada",
                "session_id": session_id
            })
        else:
            await app.state.websocket_manager.send_message(session_id, {
                "type": "error",
                "message": "Falha ao iniciar gravação"
            })
    
    except Exception as e:
        logger.error(f"Start recording error: {e}")
        resource_manager.release_memory_reservation(f"recording_{session_id}")
        await app.state.websocket_manager.send_message(session_id, {
            "type": "error",
            "message": "Erro interno ao iniciar gravação"
        })


async def handle_stop_recording(session_id: str, data: Dict[str, Any]) -> None:
    """Handle stop recording and start processing"""
    session = app.state.session_manager.get_session(session_id)
    if not session or not session.get("recording"):
        return
    
    try:
        # Stop recording
        audio_processor = app.state.audio_processor
        audio_file = await audio_processor.stop_recording(session_id)
        
        session["recording"] = False
        session["status"] = "processing"
        session["files"]["audio"] = audio_file
        
        await app.state.websocket_manager.send_message(session_id, {
            "type": "recording_stopped",
            "message": "Gravação finalizada. Iniciando processamento...",
            "audio_file": audio_file
        })
        
        # Start processing pipeline
        await start_processing_pipeline(session_id, audio_file)
        
    except Exception as e:
        logger.error(f"Stop recording error: {e}")
        await app.state.websocket_manager.send_message(session_id, {
            "type": "error",
            "message": "Erro ao finalizar gravação"
        })
    finally:
        resource_manager.release_memory_reservation(f"recording_{session_id}")


async def start_processing_pipeline(session_id: str, audio_file: str) -> None:
    """Start the full processing pipeline (transcription + diarization + SRT)"""
    session = app.state.session_manager.get_session(session_id)
    if not session:
        return
    
    processing_start = time.time()
    
    try:
        # Phase 1: Transcription
        await app.state.websocket_manager.send_message(session_id, {
            "type": "processing_progress",
            "phase": "transcription",
            "progress": 0,
            "message": "Iniciando transcrição..."
        })
        
        transcription_result = await app.state.transcription_engine.transcribe(
            audio_file, 
            language="pt",
            progress_callback=lambda p: asyncio.create_task(
                app.state.websocket_manager.send_message(session_id, {
                    "type": "processing_progress",
                    "phase": "transcription", 
                    "progress": p,
                    "message": f"Transcrevendo áudio... {p}%"
                })
            )
        )
        
        # Phase 2: Diarization
        await app.state.websocket_manager.send_message(session_id, {
            "type": "processing_progress",
            "phase": "diarization",
            "progress": 0,
            "message": "Identificando falantes..."
        })
        
        diarization_result = await app.state.diarization_engine.diarize(
            audio_file,
            transcription_data=transcription_result,
            progress_callback=lambda p: asyncio.create_task(
                app.state.websocket_manager.send_message(session_id, {
                    "type": "processing_progress",
                    "phase": "diarization",
                    "progress": p,
                    "message": f"Identificando falantes... {p}%"
                })
            )
        )
        
        # Phase 3: SRT Generation
        await app.state.websocket_manager.send_message(session_id, {
            "type": "processing_progress",
            "phase": "subtitle",
            "progress": 0,
            "message": "Gerando legendas..."
        })
        
        srt_file = await app.state.subtitle_generator.generate_srt(
            transcription_result,
            diarization_result,
            output_path=None  # Auto-generate path
        )
        
        # Processing complete
        processing_time = time.time() - processing_start
        audio_duration = get_audio_duration(audio_file)  # Helper function needed
        processing_ratio = processing_time / max(audio_duration, 1.0)
        
        # Update session with results
        session.update({
            "status": "completed",
            "processing": False,
            "files": {
                "audio": audio_file,
                "srt": srt_file
            },
            "results": {
                "transcription": transcription_result,
                "diarization": diarization_result,
                "speakers_detected": len(set(seg.get("speaker", 0) for seg in diarization_result))
            },
            "metrics": {
                "processing_time": processing_time,
                "processing_ratio": processing_ratio,
                "audio_duration": audio_duration
            }
        })
        
        # Send final results
        await app.state.websocket_manager.send_message(session_id, {
            "type": "processing_complete",
            "message": "Processamento concluído!",
            "results": session["results"],
            "files": session["files"],
            "metrics": session["metrics"]
        })
        
        # Log performance
        log_performance(
            "Processing pipeline complete",
            session_id=session_id,
            processing_time=processing_time,
            processing_ratio=processing_ratio,
            speakers_detected=session["results"]["speakers_detected"],
            warm_start=model_cache.get_cache_stats()["cached_models_count"] > 0
        )
        
    except Exception as e:
        logger.error(f"Processing pipeline error: {e}")
        session["status"] = "error"
        await app.state.websocket_manager.send_message(session_id, {
            "type": "processing_error",
            "message": "Erro durante o processamento",
            "error": str(e)
        })


async def handle_get_status(session_id: str) -> None:
    """Handle status request"""
    session = app.state.session_manager.get_session(session_id)
    resource_status = resource_manager.get_status_report()
    
    await app.state.websocket_manager.send_message(session_id, {
        "type": "status_update",
        "session": session,
        "system": {
            "memory_usage": resource_status["memory"]["usage_percent"],
            "status": resource_status["status"],
            "cached_models": model_cache.get_cache_stats()["cached_models_count"]
        }
    })


def get_audio_duration(audio_file: str) -> float:
    """Helper function to get audio duration"""
    try:
        import librosa
        duration = librosa.get_duration(filename=audio_file)
        return duration
    except:
        return 10.0  # Fallback


# ============================================================================
# STARTUP AND MAIN
# ============================================================================

if __name__ == "__main__":
    # Store startup time
    app.state.startup_time = time.time()
    
    logger.info("🚀 Starting TranscrevAI Optimized...")
    logger.info(f"🌍 Host: {CONFIG['server']['host']}:{CONFIG['server']['port']}")
    logger.info(f"🧠 Max Memory: {CONFIG['hardware']['max_memory_mb']}MB")
    logger.info(f"💾 Cache: {CONFIG['cache']['max_cache_size_mb']}MB")
    
    # Run with uvicorn
    uvicorn.run(
        app,
        host=CONFIG["server"]["host"],
        port=CONFIG["server"]["port"],
        log_level=CONFIG["logging"]["level"].lower(),
        access_log=False,  # Use our custom logging
        reload=CONFIG["server"]["debug"]
    )