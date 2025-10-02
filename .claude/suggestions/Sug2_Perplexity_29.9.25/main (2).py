"""
TranscrevAI Optimized - Main Application
Aplicação FastAPI otimizada com arquitetura browser-safe e PT-BR exclusivo
"""

import asyncio
import logging
import os
import signal
import sys
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional, Any

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.requests import Request
from starlette.websockets import WebSocketState

# Import our optimized modules
from config import CONFIG, PT_BR_CONFIG, get_config_summary
from logging_setup import setup_logging, get_logger, log_performance, PerformanceLogger
from resource_manager import get_resource_manager, initialize_resource_manager
from model_cache import get_model_cache, initialize_model_cache, get_cache_info
from src.progressive_loader import get_progressive_loader, load_essential, load_everything
from src.memory_optimizer import get_memory_optimizer, start_auto_optimization
from src.concurrent_engine import get_concurrent_engine


# Initialize logging first
logger = setup_logging(CONFIG["logging"])
app_logger = get_logger("transcrevai.main")

# Global application state
APP_STATE = {
    "startup_time": time.time(),
    "ready": False,
    "resource_manager": None,
    "model_cache": None,
    "progressive_loader": None,
    "memory_optimizer": None,
    "concurrent_engine": None,
    "websocket_connections": set(),
    "active_processing_tasks": {},
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    startup_start = time.time()
    
    try:
        app_logger.info("🚀 Starting TranscrevAI Optimized...")
        app_logger.info(f"Configuration: {get_config_summary()}")
        
        # Phase 1: Essential services (blocking)
        app_logger.info("📋 Phase 1: Loading essential services...")
        
        # Initialize resource manager
        APP_STATE["resource_manager"] = await initialize_resource_manager()
        
        # Initialize model cache
        APP_STATE["model_cache"] = initialize_model_cache(CONFIG["model_cache"])
        
        # Initialize memory optimizer
        APP_STATE["memory_optimizer"] = get_memory_optimizer()
        await start_auto_optimization()
        
        # Initialize concurrent engine
        APP_STATE["concurrent_engine"] = get_concurrent_engine()
        
        app_logger.info("✅ Essential services loaded")
        
        # Phase 2: Progressive loading (non-blocking)
        app_logger.info("🔄 Phase 2: Starting progressive loading...")
        
        APP_STATE["progressive_loader"] = get_progressive_loader()
        
        # Load essential components first
        essential_result = await load_essential()
        if essential_result["success"]:
            app_logger.info("✅ Essential components loaded")
            APP_STATE["ready"] = True
            
            # Start background loading of remaining components
            asyncio.create_task(load_remaining_components())
        else:
            app_logger.error("❌ Essential components failed to load")
            raise Exception("Essential components failed to load")
        
        startup_duration = time.time() - startup_start
        
        # Log startup performance
        log_performance(
            "Application startup",
            duration=startup_duration,
            essential_loaded=True,
            target_time=CONFIG["performance"]["targets"]["startup_time_target"],
            success=startup_duration < CONFIG["performance"]["targets"]["startup_time_target"]
        )
        
        app_logger.info(f"🎉 TranscrevAI Optimized ready in {startup_duration:.2f}s!")
        
        # Application is running
        yield
        
    except Exception as e:
        app_logger.error(f"❌ Startup failed: {e}")
        app_logger.error(traceback.format_exc())
        raise
    
    finally:
        # Shutdown sequence
        app_logger.info("🛑 Shutting down TranscrevAI Optimized...")
        
        try:
            # Close WebSocket connections
            for ws in list(APP_STATE["websocket_connections"]):
                try:
                    if ws.client_state == WebSocketState.CONNECTED:
                        await ws.close(code=1001, reason="Server shutdown")
                except:
                    pass
            
            # Stop services
            if APP_STATE["memory_optimizer"]:
                from src.memory_optimizer import stop_auto_optimization
                await stop_auto_optimization()
            
            if APP_STATE["concurrent_engine"]:
                await APP_STATE["concurrent_engine"].shutdown()
            
            if APP_STATE["resource_manager"]:
                await APP_STATE["resource_manager"].stop_monitoring()
            
            app_logger.info("✅ Shutdown complete")
            
        except Exception as e:
            app_logger.error(f"Error during shutdown: {e}")


async def load_remaining_components():
    """Load remaining components in background"""
    try:
        app_logger.info("🔄 Loading remaining components in background...")
        
        # Load all components progressively
        result = await load_everything()
        
        if result["success"]:
            app_logger.info("✅ All components loaded successfully")
            log_performance("Background loading completed", **result)
        else:
            app_logger.warning(f"⚠️ Background loading had issues: {result}")
            
    except Exception as e:
        app_logger.error(f"Background loading failed: {e}")


# Initialize FastAPI application
app = FastAPI(
    title=CONFIG["app"]["name"],
    version=CONFIG["app"]["version"],
    description=CONFIG["app"]["description"],
    lifespan=lifespan,
    docs_url="/docs" if CONFIG["development"]["enable_debug_endpoints"] else None,
    redoc_url="/redoc" if CONFIG["development"]["enable_debug_endpoints"] else None,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CONFIG["development"]["cors_allow_origins"],
    allow_credentials=CONFIG["development"]["cors_allow_credentials"],
    allow_methods=CONFIG["development"]["cors_allow_methods"],
    allow_headers=CONFIG["development"]["cors_allow_headers"],
)

# Static files and templates
templates_dir = Path(__file__).parent / "templates"
if templates_dir.exists():
    templates = Jinja2Templates(directory=str(templates_dir))

# Static files (if directory exists)
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ===== HEALTH AND STATUS ENDPOINTS =====

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        resource_manager = APP_STATE.get("resource_manager")
        memory_status = resource_manager.get_memory_status() if resource_manager else {}
        
        health_status = {
            "status": "healthy" if APP_STATE["ready"] else "starting",
            "timestamp": time.time(),
            "uptime_seconds": time.time() - APP_STATE["startup_time"],
            "ready": APP_STATE["ready"],
            "version": CONFIG["app"]["version"],
            "memory": memory_status,
            "components": {
                "resource_manager": APP_STATE["resource_manager"] is not None,
                "model_cache": APP_STATE["model_cache"] is not None,
                "progressive_loader": APP_STATE["progressive_loader"] is not None,
                "memory_optimizer": APP_STATE["memory_optimizer"] is not None,
                "concurrent_engine": APP_STATE["concurrent_engine"] is not None,
            }
        }
        
        return health_status
        
    except Exception as e:
        app_logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}


@app.get("/status")
async def get_status():
    """Get detailed application status"""
    if not APP_STATE["ready"]:
        raise HTTPException(status_code=503, detail="Application not ready")
    
    try:
        # Gather status from all components
        resource_manager = APP_STATE["resource_manager"]
        model_cache = APP_STATE["model_cache"]
        memory_optimizer = APP_STATE["memory_optimizer"]
        concurrent_engine = APP_STATE["concurrent_engine"]
        progressive_loader = APP_STATE["progressive_loader"]
        
        status = {
            "application": {
                "name": CONFIG["app"]["name"],
                "version": CONFIG["app"]["version"],
                "ready": APP_STATE["ready"],
                "uptime_seconds": time.time() - APP_STATE["startup_time"],
                "active_connections": len(APP_STATE["websocket_connections"]),
                "active_tasks": len(APP_STATE["active_processing_tasks"]),
            },
            "system": get_config_summary(),
            "resources": resource_manager.get_memory_status() if resource_manager else {},
            "cache": get_cache_info() if model_cache else {},
            "memory_optimizer": memory_optimizer.get_optimization_status() if memory_optimizer else {},
            "concurrent_engine": concurrent_engine.get_engine_stats() if concurrent_engine else {},
            "progressive_loader": progressive_loader.get_status() if progressive_loader else {},
        }
        
        return status
        
    except Exception as e:
        app_logger.error(f"Status endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """Get performance metrics"""
    if not CONFIG["development"]["enable_metrics"]:
        raise HTTPException(status_code=404, detail="Metrics disabled")
    
    try:
        from logging_setup import get_performance_stats, get_resource_stats
        
        metrics = {
            "timestamp": time.time(),
            "performance": get_performance_stats(),
            "resources": get_resource_stats(),
            "system": {
                "uptime": time.time() - APP_STATE["startup_time"],
                "ready": APP_STATE["ready"],
            }
        }
        
        return metrics
        
    except Exception as e:
        app_logger.error(f"Metrics endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== MAIN WEB INTERFACE =====

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """Main web interface"""
    try:
        if not templates_dir.exists():
            return HTMLResponse("""
            <html>
                <head><title>TranscrevAI Optimized</title></head>
                <body>
                    <h1>TranscrevAI Optimized</h1>
                    <p>Sistema de transcrição PT-BR otimizado funcionando!</p>
                    <p>API disponível em <a href="/docs">/docs</a></p>
                </body>
            </html>
            """)
        
        # Get application status for template
        app_status = {
            "ready": APP_STATE["ready"],
            "version": CONFIG["app"]["version"],
            "uptime": int(time.time() - APP_STATE["startup_time"]),
        }
        
        # Get system info
        resource_manager = APP_STATE.get("resource_manager")
        system_info = {
            "cpu_cores": CONFIG["hardware"]["cpu_cores"],
            "memory_gb": CONFIG["hardware"]["memory_total_gb"],
            "memory_status": resource_manager.get_memory_status() if resource_manager else {},
        }
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "app": app_status,
            "system": system_info,
            "config": CONFIG,
        })
        
    except Exception as e:
        app_logger.error(f"Index page failed: {e}")
        return HTMLResponse(f"<h1>Error</h1><p>{str(e)}</p>", status_code=500)


# ===== WEBSOCKET ENDPOINT =====

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await websocket.accept()
    APP_STATE["websocket_connections"].add(websocket)
    
    try:
        app_logger.info("WebSocket connection established")
        
        # Send initial status
        await websocket.send_json({
            "type": "status",
            "data": {
                "ready": APP_STATE["ready"],
                "version": CONFIG["app"]["version"],
                "message": "Conectado ao TranscrevAI Optimized"
            }
        })
        
        # WebSocket message loop
        while True:
            try:
                message = await websocket.receive_json()
                await handle_websocket_message(websocket, message)
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                app_logger.error(f"WebSocket message error: {e}")
                await websocket.send_json({
                    "type": "error",
                    "data": {"message": str(e)}
                })
                
    except WebSocketDisconnect:
        app_logger.info("WebSocket connection closed")
    except Exception as e:
        app_logger.error(f"WebSocket error: {e}")
    finally:
        APP_STATE["websocket_connections"].discard(websocket)


async def handle_websocket_message(websocket: WebSocket, message: Dict[str, Any]):
    """Handle WebSocket messages"""
    try:
        msg_type = message.get("type")
        
        if msg_type == "ping":
            await websocket.send_json({"type": "pong", "data": {"timestamp": time.time()}})
            
        elif msg_type == "get_status":
            status_data = await get_status()
            await websocket.send_json({"type": "status", "data": status_data})
            
        elif msg_type == "start_transcription":
            await handle_transcription_request(websocket, message.get("data", {}))
            
        else:
            app_logger.warning(f"Unknown WebSocket message type: {msg_type}")
            await websocket.send_json({
                "type": "error",
                "data": {"message": f"Unknown message type: {msg_type}"}
            })
            
    except Exception as e:
        app_logger.error(f"WebSocket message handling failed: {e}")
        await websocket.send_json({
            "type": "error",
            "data": {"message": str(e)}
        })


async def handle_transcription_request(websocket: WebSocket, data: Dict[str, Any]):
    """Handle transcription requests via WebSocket"""
    try:
        if not APP_STATE["ready"]:
            await websocket.send_json({
                "type": "error",
                "data": {"message": "Sistema ainda não está pronto"}
            })
            return
        
        # Extract request parameters
        audio_file = data.get("audio_file")
        if not audio_file:
            await websocket.send_json({
                "type": "error",
                "data": {"message": "Arquivo de áudio não fornecido"}
            })
            return
        
        # Create task ID
        task_id = f"ws_transcription_{int(time.time())}"
        
        # Update progress
        async def progress_callback(progress: float, message: str):
            await websocket.send_json({
                "type": "progress",
                "data": {
                    "task_id": task_id,
                    "progress": progress,
                    "message": message
                }
            })
        
        # Start transcription process
        await websocket.send_json({
            "type": "transcription_started",
            "data": {
                "task_id": task_id,
                "message": "Iniciando transcrição..."
            }
        })
        
        # Import transcription modules
        from src.audio_processing import AudioProcessor
        from src.transcription import TranscriptionEngine
        from src.speaker_diarization import SpeakerDiarization
        from src.subtitle_generator import SubtitleGenerator
        
        # Process with performance tracking
        with PerformanceLogger("websocket_transcription") as perf:
            # Initialize processors
            audio_processor = AudioProcessor()
            transcription_engine = TranscriptionEngine()
            diarization_engine = SpeakerDiarization()
            subtitle_generator = SubtitleGenerator()
            
            try:
                # Step 1: Process audio
                await progress_callback(10, "Processando áudio...")
                processed_audio = await audio_processor.process_file(audio_file)
                
                # Step 2: Transcription
                await progress_callback(30, "Transcrevendo áudio...")
                transcription_result = await transcription_engine.transcribe(
                    processed_audio, progress_callback=progress_callback
                )
                
                # Step 3: Speaker diarization
                await progress_callback(70, "Identificando falantes...")
                diarization_result = await diarization_engine.diarize(
                    processed_audio, transcription_result.get("segments", []),
                    progress_callback=progress_callback
                )
                
                # Step 4: Generate subtitles
                await progress_callback(90, "Gerando legendas...")
                srt_file = await subtitle_generator.generate_srt(
                    transcription_result.get("segments", []),
                    diarization_result,
                    progress_callback=progress_callback
                )
                
                # Step 5: Complete
                await progress_callback(100, "Transcrição concluída!")
                
                # Send final result
                await websocket.send_json({
                    "type": "transcription_complete",
                    "data": {
                        "task_id": task_id,
                        "transcription": transcription_result,
                        "diarization": diarization_result,
                        "srt_file": srt_file,
                        "processing_time": perf.extra_data.get("duration", 0),
                        "message": "Transcrição concluída com sucesso!"
                    }
                })
                
            except Exception as e:
                perf.add_data(error=str(e))
                raise e
                
    except Exception as e:
        app_logger.error(f"Transcription request failed: {e}")
        await websocket.send_json({
            "type": "transcription_error", 
            "data": {
                "task_id": task_id,
                "error": str(e),
                "message": "Erro na transcrição"
            }
        })


# ===== FILE UPLOAD ENDPOINT =====

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process audio file"""
    if not APP_STATE["ready"]:
        raise HTTPException(status_code=503, detail="Application not ready")
    
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in CONFIG["audio"]["supported_formats"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format: {file_ext}"
            )
        
        # Check file size
        max_size = CONFIG["audio"]["max_file_size_mb"] * 1024 * 1024
        file_content = await file.read()
        if len(file_content) > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {CONFIG['audio']['max_file_size_mb']}MB"
            )
        
        # Save file temporarily
        temp_dir = Path(CONFIG["paths"]["temp_dir"])
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        temp_file_path = temp_dir / f"{int(time.time())}_{file.filename}"
        
        with open(temp_file_path, "wb") as f:
            f.write(file_content)
        
        return {
            "success": True,
            "message": "Arquivo carregado com sucesso",
            "filename": file.filename,
            "temp_path": str(temp_file_path),
            "size_mb": len(file_content) / (1024 * 1024),
            "format": file_ext
        }
        
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== DEBUG ENDPOINTS (if enabled) =====

if CONFIG["development"]["enable_debug_endpoints"]:
    
    @app.get("/debug/memory")
    async def debug_memory():
        """Debug memory usage"""
        resource_manager = APP_STATE.get("resource_manager")
        if not resource_manager:
            raise HTTPException(status_code=503, detail="Resource manager not available")
        
        return {
            "memory_status": resource_manager.get_memory_status(),
            "current_metrics": resource_manager.get_current_metrics().__dict__ if resource_manager.get_current_metrics() else {},
            "metrics_history": [m.__dict__ for m in resource_manager.get_metrics_history()],
            "reservations": resource_manager.memory_reservations.get_reservations(),
        }
    
    @app.post("/debug/cleanup")
    async def debug_cleanup():
        """Force memory cleanup"""
        resource_manager = APP_STATE.get("resource_manager")
        if not resource_manager:
            raise HTTPException(status_code=503, detail="Resource manager not available")
        
        result = await resource_manager.perform_cleanup(aggressive=True)
        return result
    
    @app.get("/debug/cache")
    async def debug_cache():
        """Debug model cache"""
        model_cache = APP_STATE.get("model_cache")
        if not model_cache:
            raise HTTPException(status_code=503, detail="Model cache not available")
        
        return {
            "cache_info": get_cache_info(),
            "cached_models": model_cache.get_cached_models_info(),
        }


# ===== SIGNAL HANDLERS =====

def setup_signal_handlers():
    """Setup graceful shutdown signal handlers"""
    def signal_handler(signum, frame):
        app_logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        # uvicorn will handle the actual shutdown
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


# ===== MAIN ENTRY POINT =====

def main():
    """Main entry point"""
    try:
        app_logger.info("=" * 60)
        app_logger.info("🚀 TRANSCREVAI OPTIMIZED - SISTEMA PT-BR EXCLUSIVO 🇧🇷")
        app_logger.info("=" * 60)
        app_logger.info(f"Versão: {CONFIG['app']['version']}")
        app_logger.info(f"Ambiente: {'Desenvolvimento' if CONFIG['app']['debug'] else 'Produção'}")
        app_logger.info(f"Hardware: {CONFIG['hardware']['cpu_cores']} cores, {CONFIG['hardware']['memory_total_gb']:.1f}GB RAM")
        app_logger.info(f"Modelo Whisper: {CONFIG['whisper']['model_size']} (PT-BR)")
        app_logger.info("=" * 60)
        
        # Setup signal handlers
        setup_signal_handlers()
        
        # Determine if we should reload (development only)
        should_reload = CONFIG["app"]["debug"] and CONFIG["app"]["reload"]
        
        # Run the application
        uvicorn.run(
            "main:app",
            host=CONFIG["app"]["host"],
            port=CONFIG["app"]["port"],
            reload=should_reload,
            workers=CONFIG["app"]["workers"] if not should_reload else 1,
            log_level="info" if not CONFIG["app"]["debug"] else "debug",
            access_log=True,
            server_header=False,
            date_header=False,
        )
        
    except KeyboardInterrupt:
        app_logger.info("Aplicação interrompida pelo usuário")
    except Exception as e:
        app_logger.error(f"Falha crítica na aplicação: {e}")
        app_logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()