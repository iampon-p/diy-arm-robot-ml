class UIController {
    constructor(faceTracker) {
        this.faceTracker = faceTracker;
        this.cameraActive = false;
        this.trackingActive = false;
        this.recordingActive = false;
        
        this.initializeUI();
        this.setupEventListeners();
    }

    updateKnownCount() {
        const count = (this.faceTracker.knownFaces || []).length;
        console.log('Known faces count:', count);
        // Optionally add UI indicator in the future
    }

    initializeUI() {
        // Initialize camera devices dropdown
        this.loadCameraDevices();
        
        // Set initial button states
        this.updateButtonStates();
        
        // Initialize canvas size
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());
        
        // Add visual indicator that keyboard shortcuts are active
        this.showStatus('App ready! Press C to toggle camera, T for tracking, R for recording', 'info');
        
        // Focus on body to ensure keyboard events work
        document.body.focus();
        document.body.setAttribute('tabindex', '0');
    }

    async loadCameraDevices() {
        const cameraSelect = document.getElementById('camera-select');
        const devices = await this.faceTracker.getCameraDevices();
        
        // Clear existing options except the first one
        cameraSelect.innerHTML = '<option value="">Select Camera...</option>';
        
        devices.forEach((device, index) => {
            const option = document.createElement('option');
            option.value = device.deviceId;
            option.textContent = device.label || `Camera ${index + 1}`;
            cameraSelect.appendChild(option);
        });
    }

    setupEventListeners() {
        // Camera controls
        document.getElementById('camera-toggle').addEventListener('click', () => {
            this.toggleCamera();
        });

        // Tracking controls
        document.getElementById('tracking-toggle').addEventListener('click', () => {
            this.toggleTracking();
        });

        // Recording controls
        document.getElementById('record-data').addEventListener('click', () => {
            this.toggleRecording();
        });

        document.getElementById('download-data').addEventListener('click', () => {
            this.faceTracker.downloadData();
        });

        document.getElementById('clear-data').addEventListener('click', () => {
            if (confirm('Are you sure you want to clear all recorded data?')) {
                this.faceTracker.clearData();
                this.updateDataControls();
            }
        });

        // Settings controls
        document.getElementById('detection-confidence').addEventListener('input', (e) => {
            const confidence = parseFloat(e.target.value);
            document.getElementById('confidence-value').textContent = confidence.toFixed(1);
            this.faceTracker.updateDetectionConfidence(confidence);
        });

        // Preview toggle
        const previewToggle = document.getElementById('show-preview');
        if (previewToggle) {
            previewToggle.addEventListener('change', (e) => {
                const previewPanel = document.getElementById('track-preview');
                if (previewPanel) previewPanel.style.display = e.target.checked ? 'flex' : 'none';
                if (!e.target.checked) {
                    this.faceTracker.clearPreview();
                }
            });
        }

        document.getElementById('camera-select').addEventListener('change', (e) => {
            if (this.cameraActive) {
                // Restart camera with new device
                this.toggleCamera().then(() => {
                    if (e.target.value) {
                        this.toggleCamera();
                    }
                });
            }
        });

        document.getElementById('resolution-select').addEventListener('change', () => {
            if (this.cameraActive) {
                // Restart camera with new resolution
                this.toggleCamera().then(() => {
                    this.toggleCamera();
                });
            }
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            this.handleKeyboardShortcuts(e);
        });

        // Troubleshooting controls
        document.getElementById('test-camera').addEventListener('click', () => {
            this.testCameraAccess();
        });

        document.getElementById('debug-mode').addEventListener('click', () => {
            this.toggleDebugMode();
        });

        // Check browser support
        this.checkBrowserSupport();

        // Face recognition controls
        const recogToggle = document.getElementById('recognition-toggle');
        if (recogToggle) {
            recogToggle.addEventListener('change', (e) => {
                window.recognitionEnabled = !!e.target.checked;
                this.showStatus(`Recognition ${e.target.checked ? 'enabled' : 'disabled'}`, 'info');
            });
        }

        const enrollBtn = document.getElementById('enroll-face');
        if (enrollBtn) {
            enrollBtn.addEventListener('click', () => {
                const name = (document.getElementById('enroll-name') || {}).value || '';
                if (!name.trim()) return this.showStatus('Enter a name before enrolling', 'warning');
                // get latest landmarks from faceTracker
                const landmarks = this.faceTracker.lastLandmarks;
                if (!landmarks) return this.showStatus('No face detected to enroll', 'warning');
                const ok = this.faceTracker.enrollFace(name.trim(), landmarks);
                if (ok) this.showStatus(`Enrolled ${name}`, 'success');
                this.updateKnownCount();
            });
        }

        const clearKnown = document.getElementById('clear-known');
        if (clearKnown) {
            clearKnown.addEventListener('click', () => {
                if (!confirm('Clear all known faces?')) return;
                this.faceTracker.clearKnownFaces();
                this.updateKnownCount();
                this.showStatus('Known faces cleared', 'info');
            });
        }

        this.updateKnownCount();

        // Max faces selector
        const maxFacesSelect = document.getElementById('max-faces');
        if (maxFacesSelect) {
            maxFacesSelect.addEventListener('change', (e) => {
                const n = parseInt(e.target.value, 10) || 1;
                try {
                    this.faceTracker.setMaxFaces(n);
                    this.showStatus(`Max faces set to ${n}`, 'info');
                } catch (err) {
                    console.error('Failed to set max faces', err);
                    this.showStatus('Failed to set max faces', 'error');
                }
            });
        }
    }

    async toggleCamera() {
        const button = document.getElementById('camera-toggle');
        
        if (!this.cameraActive) {
            try {
                button.innerHTML = '<span class="loading"></span> Starting...';
                button.disabled = true;
                
                console.log('Attempting to start camera...');
                await this.faceTracker.startCamera();
                this.cameraActive = true;
                this.showStatus('Camera started successfully', 'success');
                console.log('Camera started successfully');
                
            } catch (error) {
                this.showStatus('Failed to start camera: ' + error.message, 'error');
                console.error('Camera error:', error);
                this.cameraActive = false;
            }
        } else {
            try {
                console.log('Stopping camera...');
                // Ensure camera is stopped and resources released
                await this.faceTracker.stopCamera();
                this.cameraActive = false;
                // Stop tracking and clear preview
                if (this.trackingActive) {
                    this.trackingActive = false;
                    this.faceTracker.stopTracking();
                }
                const previewPanel = document.getElementById('track-preview');
                if (previewPanel) previewPanel.style.display = 'none';
                this.showStatus('Camera stopped and resources released', 'info');
                console.log('Camera stopped successfully and preview hidden');
            } catch (error) {
                console.error('Error stopping camera:', error);
                this.showStatus('Error stopping camera', 'error');
            }
        }
        
        this.updateButtonStates();
    }

    toggleTracking() {
        if (!this.trackingActive) {
            this.faceTracker.startTracking();
            this.trackingActive = true;
            this.showStatus('Face tracking started', 'success');
        } else {
            this.faceTracker.stopTracking();
            this.trackingActive = false;
            this.showStatus('Face tracking stopped', 'info');
        }
        
        this.updateButtonStates();
    }

    toggleRecording() {
        const button = document.getElementById('record-data');
        
        if (!this.recordingActive) {
            this.faceTracker.startRecording();
            this.recordingActive = true;
            button.classList.add('recording');
            this.showStatus('Recording started', 'success');
        } else {
            this.faceTracker.stopRecording();
            this.recordingActive = false;
            button.classList.remove('recording');
            this.showStatus('Recording stopped', 'info');
        }
        
        this.updateButtonStates();
        this.updateDataControls();
    }

    updateButtonStates() {
        const cameraButton = document.getElementById('camera-toggle');
        const trackingButton = document.getElementById('tracking-toggle');
        const recordButton = document.getElementById('record-data');
        
        // Camera button
        if (this.cameraActive) {
            cameraButton.textContent = 'Stop Camera';
            cameraButton.className = 'btn btn-warning';
            cameraButton.disabled = false;
        } else {
            cameraButton.textContent = 'Start Camera';
            cameraButton.className = 'btn btn-primary';
            cameraButton.disabled = false;
        }
        
        // Tracking button
        trackingButton.disabled = !this.cameraActive;
        if (this.trackingActive) {
            trackingButton.textContent = 'Stop Tracking';
            trackingButton.className = 'btn btn-warning';
        } else {
            trackingButton.textContent = 'Start Tracking';
            trackingButton.className = 'btn btn-success';
        }
        
        // Recording button
        if (this.recordingActive) {
            recordButton.textContent = 'Stop Recording';
            recordButton.className = 'btn btn-warning recording';
        } else {
            recordButton.textContent = 'Start Recording';
            recordButton.className = 'btn btn-success';
        }
    }

    updateDataControls() {
        const downloadButton = document.getElementById('download-data');
        const hasData = this.faceTracker.faceData && this.faceTracker.faceData.length > 0;
        
        downloadButton.disabled = !hasData;
        downloadButton.textContent = hasData ? 
            `Download Data (${this.faceTracker.faceData.length} frames)` : 
            'Download Data';
    }

    resizeCanvas() {
        const canvas = document.getElementById('output_canvas');
        const container = canvas.parentElement;
        const containerWidth = container.clientWidth - 40; // Account for padding
        
        // Maintain 16:9 aspect ratio
        const aspectRatio = 16 / 9;
        canvas.style.width = containerWidth + 'px';
        canvas.style.height = (containerWidth / aspectRatio) + 'px';
        
        // Set actual canvas resolution
        canvas.width = 1280;
        canvas.height = 720;
    }

    showStatus(message, type = 'info') {
        // Create or update status indicator
        let statusDiv = document.getElementById('status-message');
        if (!statusDiv) {
            statusDiv = document.createElement('div');
            statusDiv.id = 'status-message';
            statusDiv.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 12px 20px;
                border-radius: 6px;
                color: white;
                font-weight: 500;
                z-index: 1000;
                transform: translateX(100%);
                transition: transform 0.3s ease;
            `;
            document.body.appendChild(statusDiv);
        }
        
        // Set color based on type
        const colors = {
            success: '#28a745',
            error: '#dc3545',
            warning: '#ffc107',
            info: '#17a2b8'
        };
        
        statusDiv.style.backgroundColor = colors[type] || colors.info;
        statusDiv.textContent = message;
        
        // Show the message
        statusDiv.style.transform = 'translateX(0)';
        
        // Hide after 3 seconds
        setTimeout(() => {
            statusDiv.style.transform = 'translateX(100%)';
        }, 3000);
    }

    handleKeyboardShortcuts(e) {
        // Only handle shortcuts when not typing in input fields
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') {
            return;
        }
        
        console.log('Key pressed:', e.key, 'Code:', e.code, 'Target:', e.target.tagName);
        
        switch (e.key.toLowerCase()) {
            case 'c':
                if (e.ctrlKey || e.metaKey) return; // Don't interfere with copy
                console.log('Camera toggle shortcut activated');
                this.toggleCamera();
                e.preventDefault();
                break;
                
            case 't':
                if (this.cameraActive) {
                    console.log('Tracking toggle shortcut activated');
                    this.toggleTracking();
                    e.preventDefault();
                } else {
                    this.showStatus('Start camera first before tracking', 'warning');
                }
                break;
                
            case 'r':
                if (this.trackingActive) {
                    console.log('Recording toggle shortcut activated');
                    this.toggleRecording();
                    e.preventDefault();
                } else {
                    this.showStatus('Start tracking first before recording', 'warning');
                }
                break;
                
            case 'l':
                const landmarksCheckbox = document.getElementById('show-landmarks');
                landmarksCheckbox.checked = !landmarksCheckbox.checked;
                console.log('Landmarks toggled:', landmarksCheckbox.checked);
                e.preventDefault();
                break;
                
            case 'm':
                const meshCheckbox = document.getElementById('show-mesh');
                meshCheckbox.checked = !meshCheckbox.checked;
                console.log('Mesh toggled:', meshCheckbox.checked);
                e.preventDefault();
                break;
        }
    }

    // Helper method to add status indicators to buttons
    addStatusIndicator(buttonId, isActive) {
        const button = document.getElementById(buttonId);
        let indicator = button.querySelector('.status-indicator');
        
        if (!indicator) {
            indicator = document.createElement('span');
            indicator.className = 'status-indicator';
            button.insertBefore(indicator, button.firstChild);
        }
        
        indicator.className = `status-indicator ${isActive ? 'status-active' : 'status-inactive'}`;
    }

    async testCameraAccess() {
        const button = document.getElementById('test-camera');
        const originalText = button.textContent;
        
        try {
            button.textContent = 'Testing...';
            button.disabled = true;
            
            console.log('Testing camera access...');
            
            // Test basic camera access
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            
            // Get camera info
            const track = stream.getVideoTracks()[0];
            const settings = track.getSettings();
            
            console.log('Camera test successful:', settings);
            this.showStatus(`Camera test successful! Resolution: ${settings.width}x${settings.height}`, 'success');
            
            // Clean up
            stream.getTracks().forEach(track => track.stop());
            
        } catch (error) {
            console.error('Camera test failed:', error);
            this.showStatus(`Camera test failed: ${error.message}`, 'error');
        } finally {
            button.textContent = originalText;
            button.disabled = false;
        }
    }

    toggleDebugMode() {
        const debugMode = !window.debugMode;
        window.debugMode = debugMode;
        
        const button = document.getElementById('debug-mode');
        button.textContent = debugMode ? 'Disable Debug Mode' : 'Enable Debug Mode';
        button.className = debugMode ? 'btn btn-warning' : 'btn btn-secondary';
        
        this.showStatus(`Debug mode ${debugMode ? 'enabled' : 'disabled'}`, 'info');
        
        if (debugMode) {
            console.log('Debug mode enabled - check console for detailed logs');
        }
    }

    checkBrowserSupport() {
        const supportElement = document.getElementById('browser-support');
        const features = {
            'Camera Access': !!navigator.mediaDevices?.getUserMedia,
            'WebAssembly': !!window.WebAssembly,
            'Canvas': !!document.createElement('canvas').getContext,
            'WebGL': this.checkWebGLSupport()
        };
        
        const supported = Object.values(features).every(Boolean);
        const featureList = Object.entries(features)
            .map(([name, support]) => `${name}: ${support ? '✅' : '❌'}`)
            .join(', ');
        
        supportElement.textContent = supported ? '✅ All features supported' : '⚠️ Some features missing';
        supportElement.title = featureList;
        
        if (!supported) {
            console.warn('Browser compatibility issues:', features);
        }
    }

    checkWebGLSupport() {
        try {
            const canvas = document.createElement('canvas');
            return !!(canvas.getContext('webgl') || canvas.getContext('experimental-webgl'));
        } catch (e) {
            return false;
        }
    }
}
