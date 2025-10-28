// TranscrevAI Unified Frontend Logic

// --- GLOBAL VARIABLES AND STATE ---
let currentSessionId = null;
let websocket = null;
let currentFile = null;
let liveRecorder = null;

// --- UI AND TAB MANAGEMENT ---
function switchTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.remove('active');
    });

    document.getElementById(tabName + '-tab').classList.add('active');
    // Use a more robust way to get the clicked tab
    const clickedTab = Array.from(document.querySelectorAll('.tab')).find(t => t.getAttribute('onclick').includes(tabName));
    if(clickedTab) {
        clickedTab.classList.add('active');
    }
}

// --- FILE UPLOAD WORKFLOW ---
async function processUpload() {
    if (!currentFile) return;

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
        await setupWebSocketConnection(currentSessionId, handleUploadWebSocketMessage);

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

// --- LIVE RECORDER WORKFLOW ---
class LiveRecorder {
    constructor() {
        this.mediaRecorder = null;
        this.ws = null;
        this.sessionId = 'live_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
        this.recordingState = 'idle'; // idle, recording, paused
        this.timer = null;
    }

    async init() {
        this.ws = await setupWebSocketConnection(this.sessionId, (data) => this.handleServerMessage(data));
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
        try {
            // Connect automatically if not already connected
            if (!this.ws) {
                console.log('[DEBUG] No WebSocket connection, calling init()...');
                await this.init();
                console.log('[DEBUG] init() completed, WebSocket connected');
            }

            console.log('[DEBUG] Requesting microphone access...');
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            console.log('[DEBUG] Microphone access granted');
            this.mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });

            this.mediaRecorder.ondataavailable = async (event) => {
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

            this.mediaRecorder.start(1000); // 1-second chunks
            this.ws.send(JSON.stringify({ action: 'start', format: audioFormat }));
            this.recordingState = 'recording';
            this.updateButtonStates();
            this.startTimer();
            this.updateStatus('Gravando...', 'recording');
        } catch (error) {
            console.error('Error starting recording:', error);
            this.updateStatus('Erro ao acessar microfone', 'error');
        }
    }

    pauseRecording() {
        if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
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
            this.mediaRecorder.stop();
            this.mediaRecorder.stream.getTracks().forEach(track => track.stop());
            this.ws.send(JSON.stringify({ action: 'stop' }));
            this.stopTimer();
            this.updateStatus('Processando...', 'processing');
            this.recordingState = 'processing';
            this.updateButtonStates();

            document.getElementById('status').classList.add('show');
            document.getElementById('loading-spinner').style.display = 'block';
            document.getElementById('status-text').textContent = 'Processando áudio gravado...';
        }
    }

    startTimer() {
        let seconds = 0;
        this.timer = setInterval(() => {
            seconds++;
            const mins = Math.floor(seconds / 60);
            const secs = seconds % 60;
            document.getElementById('recording-timer').textContent = `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
        }, 1000);
    }

    stopTimer() {
        if (this.timer) clearInterval(this.timer);
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
        this.updateButtonStates();
        showResults(result, this.sessionId);

        // Enable and show download audio button for live recordings
        const downloadAudioBtn = document.getElementById('download-audio-btn');
        if (downloadAudioBtn) {
            downloadAudioBtn.style.display = 'inline-block';
            downloadAudioBtn.disabled = false;
            downloadAudioBtn.onclick = () => downloadAudio(this.sessionId);
        }
    }
}

// --- GENERAL WEBSOCKET AND UI LOGIC ---
async function setupWebSocketConnection(sessionId, onMessageHandler) {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}/ws/${sessionId}`;

    try {
        const ws = new WebSocket(wsUrl);
        ws.onmessage = (event) => onMessageHandler(JSON.parse(event.data));
        ws.onerror = (error) => console.log('WebSocket error:', error);
        ws.onclose = () => console.log('WebSocket connection closed');

        await new Promise((resolve, reject) => {
            ws.onopen = resolve;
            ws.onerror = reject; // Reject promise on connection error
        });
        console.log(`WebSocket connected for session: ${sessionId}`);
        return ws;
    } catch (error) {
        console.log(`WebSocket setup failed for session ${sessionId}:`, error);
        showStatus('Erro de conexão com o servidor.', 0);
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

function showResults(data, sessionId) {
    const results = document.getElementById('results');
    const stats = document.getElementById('stats');
    const transcriptionResults = document.getElementById('transcription-results');
    const downloadBtn = document.getElementById('download-btn');

    stats.innerHTML = `
        <div class="stat-card"><div class="stat-value">${data.num_speakers || 0}</div><div class="stat-label">Falantes</div></div>
        <div class="stat-card"><div class="stat-value">${data.segments ? data.segments.length : 0}</div><div class="stat-label">Segmentos</div></div>
        <div class="stat-card"><div class="stat-value">${data.processing_time || 'N/A'}s</div><div class="stat-label">Tempo</div></div>
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

// --- INITIALIZATION ---
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
