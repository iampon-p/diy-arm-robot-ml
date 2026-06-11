# FaceTrack

Real-time face detection and tracking using Python + OpenCV, with a browser-based version using MediaPipe (2024).

## About

Built as the perception layer for a DIY robot arm project — the goal is to connect it so the arm responds to human presence and movement. Optimised for low latency on standard hardware.

Perception is the missing layer in most DIY robotics. This is where I started building it.

## Two Implementations

### Python + OpenCV (`test_model.py`, `test-camera.py`)
- Webcam face detection using OpenCV
- Low-latency, runs locally

### Web App (`face-tracking-web-app/`)
- Browser-based face tracking using MediaPipe Face Mesh
- Static HTML/JS — no backend required
- Runs with `npm start`

## Stack

| Tool | Details |
|------|---------|
| Language | Python 3 / JavaScript |
| Vision (Python) | OpenCV |
| Vision (Web) | MediaPipe Face Mesh |
| Web Server | http-server / live-server |

## Quick Start

**Python:**
```bash
python test-camera.py
```

**Web app:**
```bash
cd face-tracking-web-app
npm install
npm start
# → http://localhost:8080
```

## Related

- [DIY Arm Robot ML](https://github.com/iampon-p/diy-arm-robot-ml) — the arm this feeds into
- [RoArm-M2-S Setup](https://github.com/iampon-p/roarm-m2s-setup) — hardware integration
