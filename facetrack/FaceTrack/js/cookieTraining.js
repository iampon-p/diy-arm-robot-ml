// Cookie training helper script to add to the app.js functionality
document.addEventListener('DOMContentLoaded', () => {
    // Wait for app to initialize
    setTimeout(() => {
        if (window.faceTrackApp && window.faceTrackApp.cookieDetector) {
            setupCookieTraining();
        }
    }, 1000);
    
    function setupCookieTraining() {
        // Set up training buttons
        const trainOreo = document.getElementById('train-oreo');
        const trainBlueberry = document.getElementById('train-blueberry');
        const trainCheese = document.getElementById('train-cheese');
        
        if (trainOreo) {
            trainOreo.addEventListener('click', () => {
                trainCookie('oreo');
            });
        }
        
        if (trainBlueberry) {
            trainBlueberry.addEventListener('click', () => {
                trainCookie('blueberry');
            });
        }
        
        if (trainCheese) {
            trainCheese.addEventListener('click', () => {
                trainCookie('cheese');
            });
        }
        
        console.log('Cookie training buttons initialized');
    }
    
    function trainCookie(cookieType) {
        if (!window.faceTrackApp || !window.faceTrackApp.cookieDetector) return;
        
        // Add current frame as training sample
        const detector = window.faceTrackApp.cookieDetector;
        const success = detector.addTrainingSample(cookieType);
        
        if (success) {
            // Flash feedback
            const btn = document.getElementById(`train-${cookieType}`);
            if (btn) {
                btn.textContent = 'Added!';
                setTimeout(() => {
                    btn.textContent = `Train ${cookieType.charAt(0).toUpperCase() + cookieType.slice(1)}`;
                }, 1000);
            }
            
            console.log(`Trained ${cookieType} with current frame`);
        } else {
            console.error(`Failed to train ${cookieType}`);
        }
    }
});