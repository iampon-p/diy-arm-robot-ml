/**
 * Cookie Detection Module
 * Uses images from training folder to detect cookies in video feed
 */

class CookieDetector {
    constructor() {
        this.isDetecting = false;
        this.cookieTypes = ['oreo', 'blueberry', 'cheese'];
        this.lastDetection = null;
        this.detectionInterval = null;
        this.trainingData = {};
        this.trainingPath = './training/';
        this.trainingCanvas = document.createElement('canvas');
        this.trainingCtx = this.trainingCanvas.getContext('2d', { willReadFrequently: true });
        this.analysisCanvas = document.createElement('canvas');
        this.analysisCtx = this.analysisCanvas.getContext('2d', { willReadFrequently: true });
        
        // Set dimensions for analysis
        this.trainingCanvas.width = 100;
        this.trainingCanvas.height = 100;
        this.analysisCanvas.width = 100;
        this.analysisCanvas.height = 100;
        
        // Status of training data loading
        this.isTrainingDataReady = false;
    }

    /**
     * Initialize the cookie detector
     */
    init() {
        console.log('Initializing Cookie Detector');
        
        // Create a status message to show training progress
        this.updateCookieUI(null, 0, 'Loading training data...');
        
        // Load training data
        this.loadTrainingData();
    }

    /**
     * Load training data from images
     */
    async loadTrainingData() {
        this.isTrainingDataReady = false;
        
        try {
            // In a real application, we would scan the directory
            // For this demo, we'll set up placeholders for images
            
            this.trainingData = {
                oreo: {
                    path: this.trainingPath + 'oreo/',
                    signatures: [],
                    sampleCount: 0
                },
                blueberry: {
                    path: this.trainingPath + 'blueberry/',
                    signatures: [],
                    sampleCount: 0
                },
                cheese: {
                    path: this.trainingPath + 'cheese/',
                    signatures: [],
                    sampleCount: 0
                }
            };
            
            // Check localStorage for cached training data
            const cachedData = localStorage.getItem('cookie-detector-training');
            if (cachedData) {
                try {
                    const parsed = JSON.parse(cachedData);
                    if (parsed && parsed.signatures) {
                        console.log('Loaded training data from cache');
                        for (const type of this.cookieTypes) {
                            if (parsed.signatures[type]) {
                                this.trainingData[type].signatures = parsed.signatures[type];
                                this.trainingData[type].sampleCount = parsed.signatures[type].length;
                            }
                        }
                        this.isTrainingDataReady = true;
                        this.updateCookieUI(null, 0, 'Ready');
                        return;
                    }
                } catch (e) {
                    console.error('Error loading cached training data:', e);
                }
            }
            
            // Create demo training data (in real app, would load from images)
            await this.createDemoTrainingData();
            
            this.isTrainingDataReady = true;
            this.updateCookieUI(null, 0, 'Ready');
            console.log('Training data loaded successfully');
            
        } catch (error) {
            console.error('Error loading training data:', error);
            this.updateCookieUI(null, 0, 'Error loading training data');
        }
    }
    
    /**
     * Create demo training data
     */
    async createDemoTrainingData() {
        // Create demo color signatures for each cookie type
        
        // Oreo (dark with cream filling)
        for (let i = 0; i < 5; i++) {
            const signature = this.createDemoColorSignature(
                [20, 40],  // dark values
                [20, 40],
                [20, 40],
                [240, 255], // white cream
                [240, 255],
                [240, 255]
            );
            this.trainingData.oreo.signatures.push(signature);
        }
        this.trainingData.oreo.sampleCount = this.trainingData.oreo.signatures.length;
        
        // Blueberry (blue-purple with some brown)
        for (let i = 0; i < 5; i++) {
            const signature = this.createDemoColorSignature(
                [80, 100],  // bluish values
                [60, 85],
                [140, 200],
                [180, 210], // light brown
                [150, 180],
                [100, 140]
            );
            this.trainingData.blueberry.signatures.push(signature);
        }
        this.trainingData.blueberry.sampleCount = this.trainingData.blueberry.signatures.length;
        
        // Cheese (yellow-orange)
        for (let i = 0; i < 5; i++) {
            const signature = this.createDemoColorSignature(
                [220, 255],  // yellow-orange values
                [180, 220],
                [80, 120],
                [240, 255], // light cream areas
                [220, 240],
                [150, 190]
            );
            this.trainingData.cheese.signatures.push(signature);
        }
        this.trainingData.cheese.sampleCount = this.trainingData.cheese.signatures.length;
        
        // Save to localStorage for future use
        try {
            const toSave = {
                signatures: {
                    oreo: this.trainingData.oreo.signatures,
                    blueberry: this.trainingData.blueberry.signatures,
                    cheese: this.trainingData.cheese.signatures
                },
                timestamp: Date.now()
            };
            localStorage.setItem('cookie-detector-training', JSON.stringify(toSave));
        } catch (e) {
            console.error('Error saving training data to cache:', e);
        }
    }
    
    /**
     * Create a demo color signature for training data
     */
    createDemoColorSignature(rMin, rMax, gMin, gMax, bMin, bMax, r2Min, r2Max, g2Min, g2Max, b2Min, b2Max) {
        // Create a random color signature (histogram)
        const signature = new Array(64).fill(0);
        
        // Add primary color distribution
        for (let i = 0; i < 40; i++) {
            const r = Math.floor(rMin[0] + Math.random() * (rMax[0] - rMin[0])) >> 2;
            const g = Math.floor(gMin[0] + Math.random() * (gMax[0] - gMin[0])) >> 2;
            const b = Math.floor(bMin[0] + Math.random() * (bMax[0] - bMin[0])) >> 2;
            
            // Combine into a single index (4 bits per channel)
            const idx = (r << 4) | (g << 2) | b;
            signature[idx] += 1 + Math.random() * 5;
        }
        
        // Add secondary color distribution if provided
        if (r2Min && r2Max) {
            for (let i = 0; i < 20; i++) {
                const r = Math.floor(r2Min + Math.random() * (r2Max - r2Min)) >> 2;
                const g = Math.floor(g2Min + Math.random() * (g2Max - g2Min)) >> 2;
                const b = Math.floor(b2Min + Math.random() * (b2Max - b2Min)) >> 2;
                
                const idx = (r << 4) | (g << 2) | b;
                signature[idx] += 1 + Math.random() * 3;
            }
        }
        
        // Normalize
        const sum = signature.reduce((acc, val) => acc + val, 0);
        if (sum > 0) {
            for (let i = 0; i < signature.length; i++) {
                signature[i] = signature[i] / sum;
            }
        }
        
        return signature;
    }

    /**
     * Start cookie detection
     */
    startDetection() {
        if (this.isDetecting) return;
        
        if (!this.isTrainingDataReady) {
            console.log('Training data not ready, loading...');
            this.loadTrainingData().then(() => {
                this.startDetectionInternal();
            });
        } else {
            this.startDetectionInternal();
        }
    }
    
    /**
     * Internal method to start detection
     */
    startDetectionInternal() {
        this.isDetecting = true;
        console.log('Cookie detection started');
        
        // Run detection at intervals
        this.detectionInterval = setInterval(() => {
            this.detectCookie();
        }, 1000); // Check every second
    }

    /**
     * Stop cookie detection
     */
    stopDetection() {
        if (!this.isDetecting) return;
        
        this.isDetecting = false;
        console.log('Cookie detection stopped');
        
        if (this.detectionInterval) {
            clearInterval(this.detectionInterval);
            this.detectionInterval = null;
        }
        
        this.updateCookieUI(null, 0);
    }

    /**
     * Detect cookies using video frame and training data
     */
    detectCookie() {
        // Get video element
        const video = document.getElementById('input_video');
        if (!video || !video.srcObject || !video.videoWidth) {
            this.updateCookieUI(null, 0);
            return;
        }
        
        try {
            // Capture a frame from the video into the analysis canvas
            this.analysisCtx.drawImage(
                video,
                0, 0, video.videoWidth, video.videoHeight,
                0, 0, this.analysisCanvas.width, this.analysisCanvas.height
            );
            
            // Extract color signature from the frame
            const frameSignature = this.extractColorSignature(this.analysisCtx);
            
            // Compare with training data
            let bestMatch = null;
            let bestScore = 0;
            
            for (const type of this.cookieTypes) {
                if (!this.trainingData[type].signatures.length) continue;
                
                let totalScore = 0;
                for (const signature of this.trainingData[type].signatures) {
                    const score = this.compareSignatures(frameSignature, signature);
                    totalScore += score;
                }
                
                const avgScore = totalScore / this.trainingData[type].sampleCount;
                if (avgScore > bestScore) {
                    bestScore = avgScore;
                    bestMatch = type;
                }
            }
            
            // Use a threshold to determine if it's a valid detection
            if (bestScore > 0.6) {
                this.lastDetection = {
                    type: bestMatch,
                    confidence: bestScore
                };
                this.updateCookieUI(bestMatch, bestScore);
            } else {
                this.updateCookieUI(null, 0);
            }
        } catch (e) {
            console.error('Error in cookie detection:', e);
            this.updateCookieUI(null, 0);
        }
    }
    
    /**
     * Extract a color signature from an image
     */
    extractColorSignature(ctx) {
        // Get image data
        const imageData = ctx.getImageData(0, 0, this.analysisCanvas.width, this.analysisCanvas.height);
        const data = imageData.data;
        
        // Create color histogram (64 bins - 4 bits per channel)
        const signature = new Array(64).fill(0);
        
        // Process each pixel
        for (let i = 0; i < data.length; i += 4) {
            const r = data[i] >> 6;     // 0-3 (2 bits)
            const g = data[i + 1] >> 6;  // 0-3 (2 bits)
            const b = data[i + 2] >> 6;  // 0-3 (2 bits)
            
            // Combine into a single index (6 bits total)
            const idx = (r << 4) | (g << 2) | b;
            signature[idx]++;
        }
        
        // Normalize
        const sum = signature.reduce((acc, val) => acc + val, 0);
        if (sum > 0) {
            for (let i = 0; i < signature.length; i++) {
                signature[i] = signature[i] / sum;
            }
        }
        
        return signature;
    }
    
    /**
     * Compare two color signatures using cosine similarity
     */
    compareSignatures(sig1, sig2) {
        // Cosine similarity
        let dotProduct = 0;
        let mag1 = 0;
        let mag2 = 0;
        
        for (let i = 0; i < sig1.length; i++) {
            dotProduct += sig1[i] * sig2[i];
            mag1 += sig1[i] * sig1[i];
            mag2 += sig2[i] * sig2[i];
        }
        
        mag1 = Math.sqrt(mag1);
        mag2 = Math.sqrt(mag2);
        
        if (mag1 === 0 || mag2 === 0) return 0;
        return dotProduct / (mag1 * mag2);
    }

    /**
     * Update cookie detection UI
     * @param {string|null} cookieType - Type of cookie detected or null
     * @param {number} confidence - Confidence level (0-1)
     * @param {string|null} customStatus - Optional custom status message
     */
    updateCookieUI(cookieType, confidence, customStatus = null) {
        const statusElem = document.querySelector('.cookie-status');
        const typeElem = document.getElementById('cookie-type');
        const confidenceElem = document.getElementById('cookie-confidence');
        
        if (!statusElem || !typeElem || !confidenceElem) return;
        
        // Remove any previous cookie-type classes
        statusElem.classList.remove('cookie-oreo', 'cookie-blueberry', 'cookie-cheese');
        
        if (customStatus) {
            // Custom status message (training, loading, etc.)
            statusElem.textContent = customStatus;
            typeElem.textContent = '--';
            confidenceElem.textContent = '--';
        } else if (cookieType) {
            // Cookie detected
            statusElem.textContent = 'Cookie Detected!';
            typeElem.textContent = cookieType.charAt(0).toUpperCase() + cookieType.slice(1);
            confidenceElem.textContent = `${Math.round(confidence * 100)}%`;
            
            // Add the appropriate class for styling
            statusElem.classList.add(`cookie-${cookieType}`);
        } else {
            // No cookie detected
            statusElem.textContent = 'No cookie detected';
            typeElem.textContent = '--';
            confidenceElem.textContent = '--';
        }
    }
    
    /**
     * Add a new training sample from the current video frame
     * @param {string} cookieType - Type of cookie to train for
     */
    addTrainingSample(cookieType) {
        if (!this.cookieTypes.includes(cookieType)) {
            console.error('Invalid cookie type:', cookieType);
            return false;
        }
        
        const video = document.getElementById('input_video');
        if (!video || !video.srcObject || !video.videoWidth) {
            console.error('Video not available for training');
            return false;
        }
        
        try {
            // Capture frame to analysis canvas
            this.analysisCtx.drawImage(
                video,
                0, 0, video.videoWidth, video.videoHeight,
                0, 0, this.analysisCanvas.width, this.analysisCanvas.height
            );
            
            // Extract signature
            const signature = this.extractColorSignature(this.analysisCtx);
            
            // Add to training data
            this.trainingData[cookieType].signatures.push(signature);
            this.trainingData[cookieType].sampleCount++;
            
            // Update localStorage cache
            this.saveTrainingData();
            
            console.log(`Added training sample for ${cookieType}. Total: ${this.trainingData[cookieType].sampleCount}`);
            return true;
        } catch (e) {
            console.error('Error adding training sample:', e);
            return false;
        }
    }
    
    /**
     * Save training data to localStorage
     */
    saveTrainingData() {
        try {
            const toSave = {
                signatures: {
                    oreo: this.trainingData.oreo.signatures,
                    blueberry: this.trainingData.blueberry.signatures,
                    cheese: this.trainingData.cheese.signatures
                },
                timestamp: Date.now()
            };
            localStorage.setItem('cookie-detector-training', JSON.stringify(toSave));
        } catch (e) {
            console.error('Error saving training data to cache:', e);
        }
    }
}

// Export the class
window.CookieDetector = CookieDetector;