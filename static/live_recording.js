// Live Audio Recording Module - TranscrevAI
// Disk buffering strategy with state management

class LiveRecorder {
    constructor() {
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.ws = null;
        this.sessionId = this.generateSessionId();
        this.recordingState = 'idle'; // idle, recording, paused
    }

    generateSessionId() {
        return 'live_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
    }

    async init() {
        // Connect to live WebSocket
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        this.ws = new WebSocket(`${wsProtocol}//${window.location.host}/ws/live/${this.sessionId}`);

        this.ws.onopen = () => {
            console.log('Live WebSocket connected');
            this.updateStatus('Conectado', 'success');
        };

        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleServerMessage(data);
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateStatus('Erro de conexão', 'error');
        };

        this.ws.onclose = () => {
            console.log('Live WebSocket disconnected');
            this.updateStatus('Desconectado', 'warning');
        };
    }

    handleServerMessage(data) {
        console.log('Server message:', data);
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
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

            this.mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'audio/webm;codecs=opus'
            });

            this.mediaRecorder.ondataavailable = async (event) => {
                if (event.data.size > 0) {
                    const reader = new FileReader();
                    reader.onloadend = () => {
                        const base64Audio = reader.result.split(',')[1];
                        this.ws.send(JSON.stringify({
                            action: 'audio_chunk',
                            data: base64Audio
                        }));
                    };
                    reader.readAsDataURL(event.data);
                }
            };

            this.mediaRecorder.start(1000); // 1 second chunks

            // Send start command to server
            this.ws.send(JSON.stringify({ action: 'start' }));

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

            // Show global spinner
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
            document.getElementById('recording-timer').textContent =
                `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
        }, 1000);
    }

    stopTimer() {
        if (this.timer) {
            clearInterval(this.timer);
        }
    }

    updateButtonStates() {
        const startBtn = document.getElementById('start-live-btn');
        const pauseBtn = document.getElementById('pause-live-btn');
        const resumeBtn = document.getElementById('resume-live-btn');
        const stopBtn = document.getElementById('stop-live-btn');

        // Reset all
        [startBtn, pauseBtn, resumeBtn, stopBtn].forEach(btn => {
            if (btn) btn.disabled = true;
        });

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

        // Display results (reuse existing result display logic)
        const resultDiv = document.getElementById('results');
        if (resultDiv) {
            resultDiv.innerHTML = `
                <h3>Transcrição Completa</h3>
                <p><strong>Falantes detectados:</strong> ${result.num_speakers}</p>
                <div class="transcription-text">${result.text}</div>
                <button onclick="window.location.href='/download/srt/${this.sessionId}'">Baixar SRT</button>
            `;
            resultDiv.style.display = 'block';
        }
    }
}

// Initialize on page load
let liveRecorder = null;

document.addEventListener('DOMContentLoaded', () => {
    const initLiveBtn = document.getElementById('init-live-recording');
    if (initLiveBtn) {
        initLiveBtn.addEventListener('click', async () => {
            if (!liveRecorder) {
                liveRecorder = new LiveRecorder();
                await liveRecorder.init();
                liveRecorder.updateButtonStates();
            }
        });
    }

    // Button handlers
    const startBtn = document.getElementById('start-live-btn');
    if (startBtn) {
        startBtn.addEventListener('click', () => liveRecorder?.startRecording());
    }

    const pauseBtn = document.getElementById('pause-live-btn');
    if (pauseBtn) {
        pauseBtn.addEventListener('click', () => liveRecorder?.pauseRecording());
    }

    const resumeBtn = document.getElementById('resume-live-btn');
    if (resumeBtn) {
        resumeBtn.addEventListener('click', () => liveRecorder?.resumeRecording());
    }

    const stopBtn = document.getElementById('stop-live-btn');
    if (stopBtn) {
        stopBtn.addEventListener('click', () => liveRecorder?.stopRecording());
    }
});
