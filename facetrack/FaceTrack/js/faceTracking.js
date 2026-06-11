class FaceTracking {
    constructor() {
        this.faceMesh = null;
        this.camera = null;
        this.isTracking = false;
        this.faceData = [];
        this.isRecording = false;
        this.stats = {
            fps: 0,
            faceCount: 0,
            processingTime: 0
        };
        this.lastFrameTime = 0;
        this.frameCount = 0;
        this.currentStream = null;
        this.maxFaces = 1; // allow multi-face tracking
        this.knownFaces = this.loadKnownFaces();
        this.promptedFaces = [];
    }

    // Simple local storage for known faces
    loadKnownFaces() {
        try {
            const raw = localStorage.getItem('facetrack-known-faces');
            return raw ? JSON.parse(raw) : [];
        } catch (e) {
            console.warn('Failed to load known faces', e);
            return [];
        }
    }

    saveKnownFaces() {
        try {
            localStorage.setItem('facetrack-known-faces', JSON.stringify(this.knownFaces || []));
        } catch (e) {
            console.warn('Failed to save known faces', e);
        }
    }

    // --- PERSISTENCE PATCH START ---
    // Save faceData to localStorage
    saveDataToStorage() {
        try {
            localStorage.setItem('facetrack-face-data', JSON.stringify(this.faceData));
        } catch (e) {
            if (window.debugMode) console.error('Failed to save face data', e);
        }
    }

    // Load faceData from localStorage
    loadDataFromStorage() {
        try {
            const data = localStorage.getItem('facetrack-face-data');
            if (data) {
                this.faceData = JSON.parse(data);
                // Update data log for all loaded data
                const dataLog = document.getElementById('data-log');
                if (dataLog) dataLog.innerHTML = '';
                for (const entry of this.faceData) {
                    this.updateDataLog(entry);
                }
            }
        } catch (e) {
            if (window.debugMode) console.error('Failed to load face data', e);
        }
    }
    // --- PERSISTENCE PATCH END ---

    // Create a simple descriptor from landmarks: sample 10 landmark points and flatten relative positions
    computeDescriptor(landmarks) {
        // landmarks are normalized; pick a consistent set of indexes
        const idx = [1, 33, 263, 61, 291, 199, 10, 152, 234, 454];
        const desc = [];
        for (const i of idx) {
            const p = landmarks[i];
            desc.push(p.x, p.y, p.z || 0);
        }
        // Normalize descriptor vector length
        const norm = Math.hypot(...desc);
        if (norm > 0) return desc.map(v => v / norm);
        return desc;
    }

    enrollFace(name, landmarks) {
        if (!name || !landmarks) return false;
        const desc = this.computeDescriptor(landmarks);
        this.knownFaces = this.knownFaces || [];
        this.knownFaces.push({ name, descriptor: desc });
        this.saveKnownFaces();
        return true;
    }

    clearKnownFaces() {
        this.knownFaces = [];
        this.saveKnownFaces();
    }

    matchFace(landmarks, threshold = 0.25) {
        if (!landmarks || !this.knownFaces || this.knownFaces.length === 0) return null;
        const desc = this.computeDescriptor(landmarks);
        
        // Compare with all known faces
        let bestMatch = null;
        let minDistance = Infinity;
        
        for (const face of this.knownFaces) {
            if (!face.descriptor) continue;
            // Euclidean distance (L2 norm)
            let dist = 0;
            for (let i = 0; i < desc.length; i++) {
                const d = desc[i] - face.descriptor[i];
                dist += d * d;
            }
            dist = Math.sqrt(dist);
            
            if (dist < minDistance) {
                minDistance = dist;
                bestMatch = face.name;
            }
        }
        
        return minDistance <= threshold ? bestMatch : null;
    }

    async startCamera() {
            const videoElement = document.getElementById('input_video');
            try {
                // Check if getUserMedia is supported
                if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                    throw new Error('Camera access not supported in this browser');
                }

                console.log('Requesting camera access...');
                // Try with fallback constraints
                let constraints = {
                    video: {
                        width: { ideal: 1280 },
                        height: { ideal: 720 },
                        facingMode: 'user'
                    }
                };
                let stream;
                try {
                    stream = await navigator.mediaDevices.getUserMedia(constraints);
                } catch (error) {
                    console.warn('Failed with ideal constraints, trying basic constraints:', error);
                    // Fallback to basic constraints
                    constraints = { video: true };
                    stream = await navigator.mediaDevices.getUserMedia(constraints);
                }
            
                console.log('Camera stream obtained:', stream);
                videoElement.srcObject = stream;
                // keep a reference so we can stop it later
                this.currentStream = stream;
                // Wait for video to be ready
                await new Promise((resolve) => {
                    videoElement.onloadedmetadata = () => {
                        console.log('Video metadata loaded');
                        resolve();
                    };
                });
                this.camera = new Camera(videoElement, {
                    onFrame: async () => {
                        if (this.isTracking) {
                            const startTime = performance.now();
                            await this.faceMesh.send({ image: videoElement });
                            this.stats.processingTime = Math.round(performance.now() - startTime);
                        }
                    },
                    width: 1280,
                    height: 720
                });
                this.camera.start();
            } catch (error) {
                throw error;
            }
        }

        // Show popup near face to ask for name
        showNamePopup(x, y, landmarks, hash) {
            const popup = document.getElementById('face-name-popup');
            const input = document.getElementById('popup-name-input');
            const submit = document.getElementById('popup-name-submit');
            const cancel = document.getElementById('popup-name-cancel');
            if (!popup || !input || !submit || !cancel) return;

            // Position popup near face (x/y in video space)
            // Map to screen coordinates
            const video = document.getElementById('input_video');
            const rect = video.getBoundingClientRect();
            // x/y are in 1280x720, scale to video size
            const scaleX = rect.width / 1280;
            const scaleY = rect.height / 720;
            const px = rect.left + x * scaleX;
            const py = rect.top + y * scaleY - 40; // above face

            popup.style.left = px + 'px';
            popup.style.top = py + 'px';
            popup.style.display = 'block';
            input.value = '';
            input.focus();

            // Remove any previous listeners
            submit.onclick = () => {
                const name = input.value.trim();
                if (name) {
                    this.enrollFace(name, landmarks);
                    popup.style.display = 'none';
                }
            };
            cancel.onclick = () => {
                popup.style.display = 'none';
            };
        }

    stopCamera() {
        try {
            // Stop MediaPipe Camera if running
            if (this.camera && typeof this.camera.stop === 'function') {
                try {
                    this.camera.stop();
                } catch (e) {
                    console.warn('Error stopping MediaPipe Camera instance:', e);
                }
                this.camera = null;
            }

            const videoElement = document.getElementById('input_video');
            if (videoElement) {
                // Pause video element
                try { videoElement.pause(); } catch (e) { /* ignore */ }

                // If a MediaStream is attached, stop all tracks to release the camera
                const attachedStream = videoElement.srcObject;
                if (attachedStream && attachedStream.getTracks) {
                    attachedStream.getTracks().forEach(track => {
                        try { track.stop(); } catch (e) { /* ignore */ }
                    });
                }

                // Also stop the stored currentStream if it's different
                if (this.currentStream && this.currentStream !== attachedStream) {
                    try {
                        this.currentStream.getTracks().forEach(t => { try { t.stop(); } catch (e) {} });
                    } catch (e) { /* ignore */ }
                }

                // Clear srcObject and src attributes
                try { videoElement.srcObject = null; } catch (e) { videoElement.removeAttribute('src'); }
                try { videoElement.removeAttribute('src'); } catch (e) { /* ignore */ }
            }

            // Clear any preview canvas or frame consumers
            try { this.clearPreview(); } catch (e) { /* ignore */ }

            // Clear stored stream reference
            this.currentStream = null;
        } catch (error) {
            console.error('Error while stopping camera:', error);
        }
    }

    startTracking() {
        this.isTracking = true;
        this.lastFrameTime = performance.now();
        this.frameCount = 0;
    }

    stopTracking() {
        this.isTracking = false;
    }

    onResults(results) {
        const canvas = document.getElementById('output_canvas');
        const canvasCtx = canvas.getContext('2d');
        
        // Calculate FPS
        const currentTime = performance.now();
        this.frameCount++;
        if (currentTime - this.lastFrameTime >= 1000) {
            this.stats.fps = Math.round((this.frameCount * 1000) / (currentTime - this.lastFrameTime));
            this.frameCount = 0;
            this.lastFrameTime = currentTime;
        }

        // Clear canvas
        canvasCtx.save();
        canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
        canvasCtx.drawImage(results.image, 0, 0, canvas.width, canvas.height);

        if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
            this.stats.faceCount = results.multiFaceLandmarks.length;
            const allMetrics = [];
            const allNames = [];
            
            for (const landmarks of results.multiFaceLandmarks) {
                // Draw face mesh if enabled
                if (document.getElementById('show-mesh').checked) {
                    drawConnectors(canvasCtx, landmarks, FACEMESH_TESSELATION, 
                        { color: '#C0C0C070', lineWidth: 1 });
                }
                
                // Draw landmarks if enabled
                if (document.getElementById('show-landmarks').checked) {
                    drawLandmarks(canvasCtx, landmarks, 
                        { color: '#FF0000', lineWidth: 1, radius: 1 });
                }

                // store last landmarks for enrollment
                this.lastLandmarks = landmarks;

                // Calculate face position and metrics
                const faceMetrics = this.calculateFaceMetrics(landmarks, canvas.width, canvas.height);
                this.updateFaceInfo(faceMetrics);

                // Recognition (UI controls will set window.recognitionEnabled)
                try {
                    const enabled = !!window.recognitionEnabled;
                    if (enabled) {
                        const name = this.matchFace(landmarks, 0.28);
                        const nameEl = document.getElementById('face-name');
                        if (nameEl) nameEl.textContent = name || '--';
                    }
                } catch (e) {
                    if (window.debugMode) console.error('Recognition error', e);
                }
                
                // Record data if recording is enabled
                if (this.isRecording) {
                    this.recordFaceData(faceMetrics);
                }

                allMetrics.push(faceMetrics);
                // recognition per face
                try {
                    const enabled = !!window.recognitionEnabled;
                    if (enabled) {
                        const name = this.matchFace(landmarks, 0.28);
                        allNames.push(name || null);
                    } else {
                        allNames.push(null);
                    }
                } catch (e) {
                    allNames.push(null);
                    if (window.debugMode) console.error('Recognition error', e);
                }
            }
            // draw combined preview for all faces
            try { this.updatePreviewMulti(allMetrics, allNames); } catch (e) { if (window.debugMode) console.error(e); }
        } else {
            this.stats.faceCount = 0;
            this.updateFaceInfo(null);
            // Clear preview if no face
            this.clearPreview();
            this.lastLandmarks = null;
        }

        canvasCtx.restore();
        this.updateStats();
    }

    updatePreviewMulti(metricsArray, namesArray) {
        const preview = document.getElementById('preview_canvas');
        if (!preview) return;
        const ctx = preview.getContext('2d');

        // Clear and background
        ctx.clearRect(0, 0, preview.width, preview.height);
        ctx.fillStyle = '#111';
        ctx.fillRect(0, 0, preview.width, preview.height);

        const scaleX = preview.width / 1280;
        const scaleY = preview.height / 720;

        metricsArray.forEach((metrics, i) => {
            const x = metrics.centerX * scaleX;
            const y = metrics.centerY * scaleY;
            // color per face
            const colors = ['#00ff88', '#ffdd57', '#57a0ff', '#ff6b6b', '#c77cff', '#4ee8c4'];
            const color = colors[i % colors.length];

            // crosshair
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(x - 10, y);
            ctx.lineTo(x + 10, y);
            ctx.moveTo(x, y - 10);
            ctx.lineTo(x, y + 10);
            ctx.stroke();

            // dot
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, Math.PI * 2);
            ctx.fill();

            // name label
            const name = namesArray && namesArray[i] ? namesArray[i] : null;
            ctx.fillStyle = '#ddd';
            ctx.font = '11px monospace';
            const label = name ? `${name}` : `Face ${i+1}`;
            ctx.fillText(label, x + 8, y - 8);
        });
    }

    updatePreview(metrics, image) {
        const preview = document.getElementById('preview_canvas');
        if (!preview) return;
        const ctx = preview.getContext('2d');

        // Clear
        ctx.clearRect(0, 0, preview.width, preview.height);

        // Draw dark background
        ctx.fillStyle = '#111';
        ctx.fillRect(0, 0, preview.width, preview.height);

        // Map metrics center to preview coordinates
        const scaleX = preview.width / 1280;
        const scaleY = preview.height / 720;
        const x = metrics.centerX * scaleX;
        const y = metrics.centerY * scaleY;

        // Draw crosshair
        ctx.strokeStyle = '#00ff88';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x - 10, y);
        ctx.lineTo(x + 10, y);
        ctx.moveTo(x, y - 10);
        ctx.lineTo(x, y + 10);
        ctx.stroke();

        // Draw center dot
        ctx.fillStyle = '#00ff88';
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fill();

        // Draw bounding info
        ctx.fillStyle = '#ddd';
        ctx.font = '10px monospace';
        ctx.fillText(`X:${metrics.normalizedX}`, 6, preview.height - 28);
        ctx.fillText(`Y:${metrics.normalizedY}`, 6, preview.height - 14);
    }

    clearPreview() {
        const preview = document.getElementById('preview_canvas');
        if (!preview) return;
        const ctx = preview.getContext('2d');
        ctx.clearRect(0, 0, preview.width, preview.height);
        ctx.fillStyle = '#111';
        ctx.fillRect(0, 0, preview.width, preview.height);
    }

    calculateFaceMetrics(landmarks, canvasWidth, canvasHeight) {
        // Get key face points
        const noseTip = landmarks[1]; // Nose tip
        const leftEye = landmarks[33]; // Left eye corner
        const rightEye = landmarks[263]; // Right eye corner
        const chin = landmarks[175]; // Chin
        const forehead = landmarks[10]; // Forehead

        // Calculate center position (normalized to 0-1)
        const centerX = noseTip.x;
        const centerY = noseTip.y;

        // Calculate face size (distance between eyes as reference)
        const eyeDistance = Math.sqrt(
            Math.pow((rightEye.x - leftEye.x) * canvasWidth, 2) +
            Math.pow((rightEye.y - leftEye.y) * canvasHeight, 2)
        );

        // Calculate face rotation (based on eye line angle)
        const eyeAngle = Math.atan2(
            (rightEye.y - leftEye.y) * canvasHeight,
            (rightEye.x - leftEye.x) * canvasWidth
        ) * (180 / Math.PI);

        // Calculate face dimensions
        const faceWidth = Math.abs(rightEye.x - leftEye.x) * canvasWidth * 2.5; // Approximate
        const faceHeight = Math.abs(forehead.y - chin.y) * canvasHeight;

        return {
            centerX: Math.round(centerX * canvasWidth),
            centerY: Math.round(centerY * canvasHeight),
            normalizedX: Math.round(centerX * 100) / 100,
            normalizedY: Math.round(centerY * 100) / 100,
            size: Math.round(eyeDistance),
            rotation: Math.round(eyeAngle * 10) / 10,
            width: Math.round(faceWidth),
            height: Math.round(faceHeight),
            timestamp: Date.now()
        };
    }

    updateFaceInfo(metrics) {
        const elements = {
            'face-x': document.getElementById('face-x'),
            'face-y': document.getElementById('face-y'),
            'face-size': document.getElementById('face-size'),
            'face-rotation': document.getElementById('face-rotation')
        };

        if (metrics) {
            elements['face-x'].textContent = `${metrics.centerX}px (${metrics.normalizedX})`;
            elements['face-y'].textContent = `${metrics.centerY}px (${metrics.normalizedY})`;
            elements['face-size'].textContent = `${metrics.size}px`;
            elements['face-rotation'].textContent = `${metrics.rotation}°`;
        } else {
            Object.values(elements).forEach(el => el.textContent = '--');
        }
    }

    updateStats() {
        document.getElementById('fps').textContent = this.stats.fps;
        document.getElementById('face-count').textContent = this.stats.faceCount;
        document.getElementById('processing-time').textContent = `${this.stats.processingTime}ms`;
    }

    recordFaceData(metrics) {
        this.faceData.push({
            ...metrics,
            timestamp: new Date().toISOString()
        });

        // Update data log
        this.updateDataLog(metrics);
    }

    updateDataLog(metrics) {
        const dataLog = document.getElementById('data-log');
        const logEntry = document.createElement('div');
        logEntry.className = 'log-entry';
        
        const timestamp = new Date().toLocaleTimeString();
        logEntry.innerHTML = `
            <span class="timestamp">[${timestamp}]</span>
            <span class="data">X: ${metrics.centerX}, Y: ${metrics.centerY}, Size: ${metrics.size}, Rot: ${metrics.rotation}°</span>
        `;
        
        dataLog.appendChild(logEntry);
        dataLog.scrollTop = dataLog.scrollHeight;

        // Keep only last 50 entries
        while (dataLog.children.length > 50) {
            dataLog.removeChild(dataLog.firstChild);
        }
    }

    startRecording() {
        this.isRecording = true;
        // Do not clear faceData here; let user clear manually
        // document.getElementById('data-log').innerHTML = '';
    }

    stopRecording() {
        this.isRecording = false;
    }

    downloadData() {
        if (this.faceData.length === 0) {
            alert('No data to download. Start recording first.');
            return;
        }

        const data = {
            metadata: {
                exportDate: new Date().toISOString(),
                totalFrames: this.faceData.length,
                duration: this.faceData.length > 0 ? 
                    (new Date(this.faceData[this.faceData.length - 1].timestamp) - new Date(this.faceData[0].timestamp)) / 1000 + ' seconds' : '0 seconds'
            },
            data: this.faceData
        };

        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `facetrack-data-${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    clearData() {
        this.faceData = [];
        document.getElementById('data-log').innerHTML = '';
    }

    updateDetectionConfidence(confidence) {
        if (this.faceMesh) {
            this.faceMesh.setOptions({
                maxNumFaces: 1,
                refineLandmarks: true,
                minDetectionConfidence: confidence,
                minTrackingConfidence: confidence
            });
        }
    }

    async getCameraDevices() {
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            return devices.filter(device => device.kind === 'videoinput');
        } catch (error) {
            console.error('Error getting camera devices:', error);
            return [];
        }
    }
}
