// Main application entry point
class FaceTrackApp {
    constructor() {
        this.faceTracker = null;
        this.uiController = null;
        this.isInitialized = false;
        
        this.init();
    }

    async init() {
        try {
            // Check browser compatibility
            if (!this.checkBrowserCompatibility()) {
                this.showError('Your browser does not support the required features for face tracking.');
                return;
            }

            // Show loading state
            this.showLoading(true);

            // Initialize face tracking
            this.faceTracker = new FaceTracking();
            
            // Wait for face tracking to be ready
            await this.waitForFaceTrackerReady();
            
            // Initialize UI controller
            this.uiController = new UIController(this.faceTracker);
            
            this.isInitialized = true;
            this.showLoading(false);
            
            console.log('FaceTrack app initialized successfully');
            
            // Show welcome message
            this.showWelcomeMessage();
            
        } catch (error) {
            console.error('Failed to initialize FaceTrack app:', error);
            this.showError('Failed to initialize the application: ' + error.message);
            this.showLoading(false);
        }
    }

    checkBrowserCompatibility() {
        // Check for required APIs
        const requiredAPIs = [
            'navigator.mediaDevices',
            'navigator.mediaDevices.getUserMedia',
            'WebAssembly',
            'OffscreenCanvas'
        ];

        for (const api of requiredAPIs) {
            if (!this.hasAPI(api)) {
                console.warn(`Missing API: ${api}`);
                return false;
            }
        }

        // Check for WebGL support
        const canvas = document.createElement('canvas');
        const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
        if (!gl) {
            console.warn('WebGL not supported');
            return false;
        }

        return true;
    }

    hasAPI(apiPath) {
        const parts = apiPath.split('.');
        let current = window;
        
        for (const part of parts) {
            if (!current || typeof current[part] === 'undefined') {
                return false;
            }
            current = current[part];
        }
        
        return true;
    }

    async waitForFaceTrackerReady() {
        // Wait for MediaPipe to be ready
        return new Promise((resolve) => {
            const checkReady = () => {
                if (this.faceTracker.faceMesh) {
                    resolve();
                } else {
                    setTimeout(checkReady, 100);
                }
            };
            checkReady();
        });
    }

    showLoading(show) {
        let loader = document.getElementById('app-loader');
        
        if (show && !loader) {
            loader = document.createElement('div');
            loader.id = 'app-loader';
            loader.innerHTML = `
                <div style="
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: rgba(102, 126, 234, 0.9);
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                    z-index: 9999;
                    color: white;
                ">
                    <div class="loading" style="
                        width: 60px;
                        height: 60px;
                        border: 6px solid rgba(255,255,255,0.3);
                        border-top: 6px solid white;
                        margin-bottom: 20px;
                    "></div>
                    <h2>Initializing FaceTrack...</h2>
                    <p>Loading face detection models</p>
                </div>
            `;
            document.body.appendChild(loader);
        } else if (!show && loader) {
            loader.remove();
        }
    }

    showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.innerHTML = `
            <div style="
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
                z-index: 10000;
                max-width: 500px;
                text-align: center;
            ">
                <h3 style="color: #dc3545; margin-bottom: 15px;">❌ Error</h3>
                <p style="margin-bottom: 20px;">${message}</p>
                <button onclick="location.reload()" style="
                    background: #667eea;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    cursor: pointer;
                ">Reload Page</button>
            </div>
            <div style="
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0,0,0,0.5);
                z-index: 9999;
            "></div>
        `;
        document.body.appendChild(errorDiv);
    }

    showWelcomeMessage() {
        // Only show if it's the first visit
        if (!localStorage.getItem('facetrack-visited')) {
            const welcomeDiv = document.createElement('div');
            welcomeDiv.innerHTML = `
                <div style="
                    position: fixed;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    background: white;
                    padding: 40px;
                    border-radius: 15px;
                    box-shadow: 0 20px 50px rgba(0,0,0,0.3);
                    z-index: 10000;
                    max-width: 600px;
                    text-align: center;
                ">
                    <h2 style="color: #667eea; margin-bottom: 20px;">🎯 Welcome to FaceTrack!</h2>
                    <p style="margin-bottom: 15px;">Real-time face tracking and analysis in your browser.</p>
                    <div style="text-align: left; margin: 20px 0;">
                        <h4>Quick Start:</h4>
                        <ol style="margin: 10px 0; padding-left: 20px;">
                            <li>Click "Start Camera" to begin</li>
                            <li>Click "Start Tracking" to enable face detection</li>
                            <li>Use "Start Recording" to capture tracking data</li>
                        </ol>
                        <h4>Keyboard Shortcuts:</h4>
                        <ul style="margin: 10px 0; padding-left: 20px;">
                            <li><strong>C</strong> - Toggle Camera</li>
                            <li><strong>T</strong> - Toggle Tracking</li>
                            <li><strong>R</strong> - Toggle Recording</li>
                            <li><strong>L</strong> - Toggle Landmarks</li>
                            <li><strong>M</strong> - Toggle Face Mesh</li>
                        </ul>
                    </div>
                    <button onclick="this.parentElement.parentElement.remove()" style="
                        background: #667eea;
                        color: white;
                        border: none;
                        padding: 12px 24px;
                        border-radius: 6px;
                        cursor: pointer;
                        font-size: 16px;
                    ">Got it!</button>
                </div>
                <div style="
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: rgba(0,0,0,0.5);
                    z-index: 9999;
                " onclick="this.parentElement.remove()"></div>
            `;
            document.body.appendChild(welcomeDiv);
            localStorage.setItem('facetrack-visited', 'true');
        }
    }
}

// Performance monitoring
class PerformanceMonitor {
    constructor() {
        this.metrics = {
            memoryUsage: 0,
            cpuUsage: 0,
            frameDrops: 0
        };
        
        this.startMonitoring();
    }

    startMonitoring() {
        setInterval(() => {
            this.updateMemoryUsage();
        }, 5000);
    }

    updateMemoryUsage() {
        if (performance.memory) {
            this.metrics.memoryUsage = Math.round(performance.memory.usedJSHeapSize / 1024 / 1024);
            
            // Add memory usage to stats if element exists
            const memoryElement = document.getElementById('memory-usage');
            if (memoryElement) {
                memoryElement.textContent = `${this.metrics.memoryUsage} MB`;
            }
        }
    }
}

// Error handling
window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
    
    // Show user-friendly error message
    const errorNotification = document.createElement('div');
    errorNotification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: #dc3545;
        color: white;
        padding: 15px;
        border-radius: 6px;
        z-index: 10000;
        max-width: 300px;
    `;
    errorNotification.textContent = 'An error occurred. Check the console for details.';
    document.body.appendChild(errorNotification);
    
    setTimeout(() => {
        errorNotification.remove();
    }, 5000);
});

// Unhandled promise rejection handling
window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
    event.preventDefault();
});

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('Starting FaceTrack application...');
    
    // Initialize performance monitoring
    const performanceMonitor = new PerformanceMonitor();
    
    // Initialize main app
    const app = new FaceTrackApp();
    
    // Make app globally available for debugging
    window.faceTrackApp = app;
    window.performanceMonitor = performanceMonitor;
});

// Service Worker registration for offline functionality (optional)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        // Uncomment to enable service worker
        // navigator.serviceWorker.register('/sw.js')
        //     .then(registration => console.log('SW registered'))
        //     .catch(error => console.log('SW registration failed'));
    });
}
