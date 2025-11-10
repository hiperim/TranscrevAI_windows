// TranscrevAI Frontend Logic

// --- Global variables and state
let currentSessionId = null;
let websocket = null;
let currentFile = null;
let liveRecorder = null;

// --- UI and tab management
function switchTab(tabName) {
    // Clear previous results when switching tabs
    clearResults();

    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.remove('active');
    });

    document.getElementById(tabName + '-tab').classList.add('active');
    // Get the clicked tab
    const clickedTab = Array.from(document.querySelectorAll('.tab')).find(t => t.getAttribute('onclick').includes(tabName));
    if(clickedTab) {
        clickedTab.classList.add('active');
    }
}

// --- File upload workflow
async function processUpload() {
    if (!currentFile) return;

    // Check if live recording is in progress
    if (liveRecorder && (liveRecorder.recordingState === 'recording' || liveRecorder.recordingState === 'paused')) {
        if (!confirm('Há uma gravação em andamento. Iniciar upload cancelará a gravação. Continuar?')) {
            return; // User cancelled
        }
        console.log('[DEBUG] Stopping active recording...');
        // Force stop recording and close WebSocket
        if (liveRecorder.mediaRecorder) {
            liveRecorder.mediaRecorder.stop();
            liveRecorder.mediaRecorder.stream.getTracks().forEach(track => track.stop());
        }
        if (liveRecorder.ws && liveRecorder.ws.readyState === WebSocket.OPEN) {
            liveRecorder.ws.close();
            liveRecorder.ws = null;
        }
        liveRecorder.recordingState = 'idle';
        liveRecorder.stopTimer();
        liveRecorder.updateButtonStates();
    }

    // Clear previous results when starting new upload
    clearResults();

    const uploadBtn = document.getElementById('upload-btn');
    uploadBtn.disabled = true;
    uploadBtn.textContent = 'Processando...';

    document.getElementById('loading-spinner').style.display = 'block';


    currentSessionId = 'upload_' + Date.now();

    const isFirstTime = await checkFirstTimeUsage();
    if (isFirstTime) {
        showFirstTimeNotice();
    }

    showStatus('Enviando arquivo...', 5);

    try {
        const connection = await setupWebSocketConnection(currentSessionId, handleUploadWebSocketMessage);
        websocket = connection.ws;

        const formData = new FormData();
        formData.append('file', currentFile);
        formData.append('session_id', currentSessionId);
        formData.append('language', 'pt');

        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.success) {
            showStatus('Arquivo enviado. Processando...', 10);
        } else {
            showStatus('Erro: ' + (result.error || 'Erro desconhecido'), 0);
            uploadBtn.disabled = false;
            uploadBtn.textContent = 'Processar Arquivo';
            document.getElementById('loading-spinner').style.display = 'none';
        }
    } catch (error) {
        showStatus('Erro no upload: ' + error.message, 0);
        uploadBtn.disabled = false;
        uploadBtn.textContent = 'Processar Arquivo';
        document.getElementById('loading-spinner').style.display = 'none';
    }
}

async function checkFirstTimeUsage() {
    try {
        const response = await fetch('/check-first-time');
        const result = await response.json();
        return result.is_first_time;
    } catch (error) {
        console.log('Could not check first-time status:', error);
        return false;
    }
}

function showFirstTimeNotice() {
    const notice = document.createElement('div');
    notice.className = 'info-box';
    notice.innerHTML = '<div class="info-box-title">Primeira utilização detectada</div><div class="info-box-content">O sistema precisará baixar o modelo de IA (1.4GB) na primeira vez.<br>Este processo pode levar 3-5 minutos dependendo da sua conexão.<br><strong>Nas próximas utilizações será muito mais rápido!</strong></div>';

    const statusElement = document.getElementById('status');
    statusElement.parentNode.insertBefore(notice, statusElement);

    setTimeout(() => {
        if (notice.parentNode) {
            notice.remove();
        }
    }, 15000);
}

// --- Live recorder workflow
class LiveRecorder {
    constructor() {
        this.mediaRecorder = null;
        this.ws = null;
        this.sessionId = 'live_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
        this.recordingState = 'idle'; // idle, recording, paused
        this.isStopping = false; // prevent chunks after stop
        this.timer = null;
        this.heartbeatInterval = null;
    }

    async init() {
        const connection = await setupWebSocketConnection(this.sessionId, (data) => this.handleServerMessage(data));
        this.ws = connection.ws;
        this.heartbeatInterval = connection.heartbeatInterval;
        this.updateStatus('Conectado', 'success');
    }

    handleServerMessage(data) {
        console.log('Live-Recorder message:', data);
        const spinner = document.getElementById('loading-spinner');

        if (data.type === 'state_change') {
            this.recordingState = data.data.status;
            this.updateButtonStates();
        } else if (data.type === 'processing') {
            this.updateStatus(data.message, 'info');
        } else if (data.type === 'error') {
            this.updateStatus(`Erro: ${data.message}`, 'error');
            if (spinner) spinner.style.display = 'none';
        } else if (data.type === 'complete') {
            if (spinner) spinner.style.display = 'none';
            this.handleTranscriptionComplete(data.result);
        }
    }

    async startRecording() {
        console.log('[DEBUG] LiveRecorder.startRecording() called');

        // Check if upload is in progress
        if (websocket && websocket.readyState === WebSocket.OPEN) {
            if (!confirm('Há um upload em andamento. Iniciar gravação cancelará o upload. Continuar?')) {
                return; // User cancelled
            }
            console.log('[DEBUG] Closing upload WebSocket...');
            websocket.close();
            websocket = null;
        }

        // Clear previous results when starting new recording
        clearResults();

        try {
            // Close previous WebSocket if exists
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                console.log('[DEBUG] Closing previous WebSocket...');
                this.ws.close();
            }

            // Generate new session ID for each recording
            this.sessionId = 'live_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
            console.log('[DEBUG] New session ID:', this.sessionId);

            // Reset state for new recording
            this.isStopping = false;
            this.recordingState = 'idle';

            // Always create fresh WebSocket connection
            console.log('[DEBUG] Creating new WebSocket connection...');
            await this.init();
            console.log('[DEBUG] WebSocket connected');

            console.log('[DEBUG] Requesting microphone access...');
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            console.log('[DEBUG] Microphone access granted');
            this.mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });

            this.mediaRecorder.ondataavailable = async (event) => {
                if (this.isStopping) return; // Don't send chunks after stop
                if (event.data.size > 0) {
                    const reader = new FileReader();
                    reader.onloadend = () => {
                        const base64Audio = reader.result.split(',')[1];
                        this.ws.send(JSON.stringify({ action: 'audio_chunk', data: base64Audio }));
                    };
                    reader.readAsDataURL(event.data);
                }
            };

            // Get selected audio format from UI
            const selectedFormat = document.querySelector('input[name="audio-format"]:checked');
            const audioFormat = selectedFormat ? selectedFormat.value : 'wav';
            console.log('[DEBUG] Selected audio format:', audioFormat);

            const chunkDurationMs = 5000; // Send chunks every 5 seconds
            this.mediaRecorder.start(chunkDurationMs); // Start recording with chunk interval
            this.ws.send(JSON.stringify({ action: 'start', format: audioFormat }));
            this.recordingState = 'recording';
            this.updateButtonStates();
            this.resetTimer();  // Reset timer for new recording
            this.startTimer();
            this.updateStatus('Gravando...', 'recording');
        } catch (error) {
            console.error('Error starting recording:', error);
            this.updateStatus('Erro ao acessar microfone', 'error');
        }
    }

    pauseRecording() {
        if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
            // Force send any accumulated audio data before pausing
            this.mediaRecorder.requestData();
            this.mediaRecorder.pause();
            this.ws.send(JSON.stringify({ action: 'pause' }));
            this.stopTimer();
            this.updateStatus('Pausado', 'paused');
        }
    }

    resumeRecording() {
        if (this.mediaRecorder && this.mediaRecorder.state === 'paused') {
            this.mediaRecorder.resume();
            this.ws.send(JSON.stringify({ action: 'resume' }));
            this.startTimer();
            this.updateStatus('Gravando...', 'recording');
        }
    }

    stopRecording() {
        if (this.mediaRecorder) {
            // Cancel heartbeat to prevent ping after stop
            if (this.heartbeatInterval) {
                clearInterval(this.heartbeatInterval);
                this.heartbeatInterval = null;
            }

            // Force send any accumulated audio data before stopping
            this.mediaRecorder.requestData();

            // Small delay to ensure data is sent before stopping
            setTimeout(() => {
                // Set flag NOW to prevent chunks from stop() event
                this.isStopping = true;

                this.mediaRecorder.stop();
                this.mediaRecorder.stream.getTracks().forEach(track => track.stop());

                // Final command to backend
                this.ws.send(JSON.stringify({ action: 'stop' }));

                //Update UI state
                this.stopTimer();
                this.updateStatus('Processando...', 'processing');
                this.recordingState = 'processing';
                this.updateButtonStates();

                document.getElementById('status').classList.add('show');
                document.getElementById('loading-spinner').style.display = 'block';
                document.getElementById('status-text').textContent = 'Processando áudio gravado...';

                // DON'T close WebSocket here - pipeline needs it for progress updates
                // WebSocket will be closed in handleTranscriptionComplete()
            }, 150);
        }
    }

    startTimer() {
        // Initialize elapsed seconds if not set (first start)
        if (this.elapsedSeconds === undefined) {
            this.elapsedSeconds = 0;
        }

        this.timer = setInterval(() => {
            this.elapsedSeconds++;
            const mins = Math.floor(this.elapsedSeconds / 60);
            const secs = this.elapsedSeconds % 60;
            document.getElementById('recording-timer').textContent = `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
        }, 1000);
    }

    stopTimer() {
        if (this.timer) {
            clearInterval(this.timer);
            this.timer = null;
        }
    }

    resetTimer() {
        this.stopTimer();
        this.elapsedSeconds = 0;
        document.getElementById('recording-timer').textContent = '00:00';
    }

    updateButtonStates() {
        const [startBtn, pauseBtn, resumeBtn, stopBtn] = ['start-live-btn', 'pause-live-btn', 'resume-live-btn', 'stop-live-btn'].map(id => document.getElementById(id));
        [startBtn, pauseBtn, resumeBtn, stopBtn].forEach(btn => { if (btn) btn.disabled = true; });

        if (this.recordingState === 'idle') {
            if (startBtn) startBtn.disabled = false;
        } else if (this.recordingState === 'recording') {
            if (pauseBtn) pauseBtn.disabled = false;
            if (stopBtn) stopBtn.disabled = false;
        } else if (this.recordingState === 'paused') {
            if (resumeBtn) resumeBtn.disabled = false;
            if (stopBtn) stopBtn.disabled = false;
        }
    }

    updateStatus(message, type) {
        const statusEl = document.getElementById('live-status');
        if (statusEl) {
            statusEl.textContent = message;
            statusEl.className = `status-indicator status-${type}`;
        }
    }

    handleTranscriptionComplete(result) {
        this.updateStatus('Concluído!', 'success');
        this.recordingState = 'idle';
        this.isStopping = false; // Reset flag for next recording
        this.updateButtonStates();
        showResults(result, this.sessionId);

        // Enable and show download audio button for live recordings
        const downloadAudioBtn = document.getElementById('download-audio-btn');
        if (downloadAudioBtn) {
            downloadAudioBtn.style.display = 'inline-block';
            downloadAudioBtn.disabled = false;
            downloadAudioBtn.onclick = () => downloadAudio(this.sessionId);
        }

        // Keep WebSocket open - session persists for downloads
        // Will be closed when starting new recording
    }
}

// --- General websocket and UI logic
const WS_CONFIG = {
    TIMEOUT: 30000,           // 30s connection timeout
    RETRY_ATTEMPTS: 3,        // max retry attempts
    RETRY_DELAY: 2000,        // 2s initial delay
    HEARTBEAT_INTERVAL: 15000 // 15s heartbeat
};

async function setupWebSocketConnection(sessionId, onMessageHandler, retryCount = 0) {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}/ws/${sessionId}`;

    try {
        const ws = new WebSocket(wsUrl);
        let heartbeatInterval = null;
        let connectionTimeout = null;

        // Setup heartbeat to keep connection alive
        ws.onopen = () => {
            console.log(`WebSocket connected for session: ${sessionId}`);
            clearTimeout(connectionTimeout);

            // Send heartbeat every 15s
            heartbeatInterval = setInterval(() => {
                if (ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ action: 'ping' }));
                }
            }, WS_CONFIG.HEARTBEAT_INTERVAL);
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type !== 'pong') {  // Ignore pong responses
                onMessageHandler(data);
            }
        };

        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            clearInterval(heartbeatInterval);
        };

        ws.onclose = (event) => {
            console.log(`WebSocket closed: ${event.code} ${event.reason}`);
            clearInterval(heartbeatInterval);

            // Auto-reconnect if not clean close and retries available
            if (!event.wasClean && retryCount < WS_CONFIG.RETRY_ATTEMPTS) {
                const delay = WS_CONFIG.RETRY_DELAY * Math.pow(2, retryCount);
                console.log(`Reconnecting in ${delay}ms (attempt ${retryCount + 1}/${WS_CONFIG.RETRY_ATTEMPTS})`);
                showStatus(`Reconectando... (tentativa ${retryCount + 1})`, 50);

                setTimeout(() => {
                    setupWebSocketConnection(sessionId, onMessageHandler, retryCount + 1)
                        .then(connection => {
                            websocket = connection.ws;  // update global reference
                        });
                }, delay);
            } else if (retryCount >= WS_CONFIG.RETRY_ATTEMPTS) {
                showStatus('Falha na conexão. Recarregue a página.', 0);
            }
        };

        // Connection timeout
        connectionTimeout = setTimeout(() => {
            if (ws.readyState !== WebSocket.OPEN) {
                ws.close();
                throw new Error('Connection timeout');
            }
        }, WS_CONFIG.TIMEOUT);

        // Wait for connection
        await new Promise((resolve, reject) => {
            const originalOnOpen = ws.onopen;
            ws.onopen = (event) => {
                originalOnOpen(event);
                resolve();
            };
            ws.onerror = reject;
        });

        return { ws, heartbeatInterval };
    } catch (error) {
        console.error(`WebSocket setup failed for session ${sessionId}:`, error);

        // Retry if attempts remaining
        if (retryCount < WS_CONFIG.RETRY_ATTEMPTS) {
            const delay = WS_CONFIG.RETRY_DELAY * Math.pow(2, retryCount);
            showStatus(`Tentando reconectar... (${retryCount + 1}/${WS_CONFIG.RETRY_ATTEMPTS})`, 25);
            await new Promise(resolve => setTimeout(resolve, delay));
            return setupWebSocketConnection(sessionId, onMessageHandler, retryCount + 1);
        } else {
            showStatus('Erro de conexão. Verifique sua internet.', 0);
            throw error;
        }
    }
}

function handleUploadWebSocketMessage(data) {
    const { type, message, progress, details, result } = data;
    const uploadBtn = document.getElementById('upload-btn');
    const spinner = document.getElementById('loading-spinner');

    switch (type) {
        case 'progress':
            const { stage, percentage, estimated_time_remaining } = data;
            let progressMessage = message || 'Processando...';
            if (estimated_time_remaining) {
                progressMessage += `<br><span style="font-size: 0.85rem; color: #A0A0A0;">Tempo restante: ~${estimated_time_remaining}s</span>`;
            }
            showStatus(progressMessage, percentage);
            break;
        case 'complete':
            showStatus('Processamento concluído!', 100);
            showResults(result, currentSessionId);
            document.getElementById('download-btn').disabled = false;
            if (uploadBtn) {
                uploadBtn.disabled = false;
                uploadBtn.textContent = 'Processar Arquivo';
            }
            if (spinner) spinner.style.display = 'none';

            // Close upload WebSocket after completion
            if (websocket && websocket.readyState === WebSocket.OPEN) {
                console.log('[DEBUG] Closing upload WebSocket after completion');
                websocket.close();
                websocket = null;
            }
            break;
        case 'error':
            showStatus('Erro: ' + (message || 'Erro desconhecido'), 0);
            if (uploadBtn) {
                uploadBtn.disabled = false;
                uploadBtn.textContent = 'Processar Arquivo';
            }
            if (spinner) spinner.style.display = 'none';
            break;
        default:
            if (message) showStatus(message, progress || 50);
    }
}

function showStatus(text, progress) {
    const status = document.getElementById('status');
    const statusText = document.getElementById('status-text');
    const progressFill = document.getElementById('progress-fill');

    status.classList.add('show');
    statusText.innerHTML = text;
    progressFill.style.width = progress + '%';
}

function clearResults() {
    const results = document.getElementById('results');
    const stats = document.getElementById('stats');
    const transcriptionResults = document.getElementById('transcription-results');
    const status = document.getElementById('status');
    const downloadBtn = document.getElementById('download-btn');
    const downloadAudioBtn = document.getElementById('download-audio-btn');

    // Hide results section
    if (results) results.classList.remove('show');

    // Clear stats
    if (stats) stats.innerHTML = '';

    // Clear transcription
    if (transcriptionResults) transcriptionResults.innerHTML = '';

    // Hide status
    if (status) status.classList.remove('show');

    // Disable download buttons
    if (downloadBtn) downloadBtn.disabled = true;
    if (downloadAudioBtn) downloadAudioBtn.style.display = 'none';
}

function showResults(data, sessionId) {
    const results = document.getElementById('results');
    const stats = document.getElementById('stats');
    const transcriptionResults = document.getElementById('transcription-results');
    const downloadBtn = document.getElementById('download-btn');

    stats.innerHTML = `
        <div class="stat-card"><div class="stat-value">${data.num_speakers || 0}</div><div class="stat-label">Falantes</div></div>
        <div class="stat-card"><div class="stat-value">${data.segments ? data.segments.length : 0}</div><div class="stat-label">Segmentos</div></div>
        <div class="stat-card"><div class="stat-value">${data.audio_duration || 'N/A'}s</div><div class="stat-label">Duração do áudio</div></div>
        <div class="stat-card"><div class="stat-value">${data.processing_ratio ? data.processing_ratio + 'x' : 'N/A'}</div><div class="stat-label">Ratio</div></div>
    `;

    transcriptionResults.innerHTML = '';
    if (data.segments && Array.isArray(data.segments)) {
        data.segments.forEach(function(segment) {
            const segmentDiv = document.createElement('div');
            segmentDiv.className = 'segment';
            segmentDiv.innerHTML = `
                <div class="speaker">${segment.speaker || 'SPEAKER_01'}</div>
                <div class="segment-text">${segment.text || ''}</div>
                <div class="timestamp">${formatTime(segment.start || 0)} - ${formatTime(segment.end || 0)}</div>
            `;
            transcriptionResults.appendChild(segmentDiv);
        });
    }

    if (downloadBtn) {
        downloadBtn.onclick = () => downloadSRT(sessionId);
        downloadBtn.disabled = false;
    }

    results.classList.add('show');
}

function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return mins + ':' + secs.toString().padStart(2, '0');
}

async function downloadSRT(sessionId) {
    if (!sessionId) return;
    try {
        const response = await fetch('/download-srt/' + sessionId);
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `transcricao_${sessionId}.srt`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        } else {
            alert('Erro ao baixar arquivo SRT');
        }
    } catch (error) {
        alert('Erro: ' + error.message);
    }
}

async function downloadAudio(sessionId) {
    if (!sessionId) return;
    try {
        const response = await fetch(`/api/download/${sessionId}/audio`);
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;

            // Determine file extension from content-type or default to .wav
            const contentType = response.headers.get('content-type');
            const extension = contentType && contentType.includes('mp4') ? '.mp4' : '.wav';
            a.download = `gravacao_${sessionId}${extension}`;

            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        } else {
            alert('Erro ao baixar arquivo de áudio');
        }
    } catch (error) {
        alert('Erro: ' + error.message);
    }
}

// --- Initialization
document.addEventListener('DOMContentLoaded', () => {
    console.log('[DEBUG] DOM Content Loaded - Initializing app...');
    // File Upload Listeners
    const fileInput = document.getElementById('file-input');
    const fileDropZone = document.querySelector('.file-drop-zone');
    const uploadBtn = document.getElementById('upload-btn');

    if (uploadBtn) uploadBtn.onclick = processUpload;

    if(fileInput) {
        fileInput.addEventListener('change', (e) => {
            currentFile = e.target.files[0];
            if (currentFile) {
                fileDropZone.querySelector('p').innerHTML = `<strong>Arquivo selecionado: ${currentFile.name}</strong><br>Clique em "Processar Arquivo" para continuar`;
                uploadBtn.disabled = false;
            }
        });
    }

    if(fileDropZone) {
        fileDropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            fileDropZone.classList.add('dragover');
        });

        fileDropZone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            fileDropZone.classList.remove('dragover');
        });

        fileDropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            fileDropZone.classList.remove('dragover');
            currentFile = e.dataTransfer.files[0];
            if (currentFile) {
                fileDropZone.querySelector('p').innerHTML = `<strong>Arquivo selecionado: ${currentFile.name}</strong><br>Clique em "Processar Arquivo" para continuar`;
                uploadBtn.disabled = false;
            }
        });
    }

    // Live Recorder Listeners
    console.log('[DEBUG] Registering live recording listeners...');
    const startBtn = document.getElementById('start-live-btn');
    console.log('[DEBUG] startBtn element:', startBtn);

    if (startBtn) {
        console.log('[DEBUG] startBtn found, adding event listener');
        startBtn.addEventListener('click', async () => {
            console.log('[DEBUG] Start button clicked!');
            try {
                if (!liveRecorder) {
                    console.log('[DEBUG] Creating new LiveRecorder instance');
                    liveRecorder = new LiveRecorder();
                }
                console.log('[DEBUG] Calling startRecording()');
                await liveRecorder.startRecording();
                console.log('[DEBUG] startRecording() completed');
            } catch (error) {
                console.error('[DEBUG] Error in startRecording:', error);
            }
        });
    } else {
        console.error('[DEBUG] startBtn NOT FOUND!');
    }

    const pauseBtn = document.getElementById('pause-live-btn');
    if (pauseBtn) pauseBtn.addEventListener('click', () => liveRecorder?.pauseRecording());

    const resumeBtn = document.getElementById('resume-live-btn');
    if (resumeBtn) resumeBtn.addEventListener('click', () => liveRecorder?.resumeRecording());

    const stopBtn = document.getElementById('stop-live-btn');
    if (stopBtn) stopBtn.addEventListener('click', () => liveRecorder?.stopRecording());
});
