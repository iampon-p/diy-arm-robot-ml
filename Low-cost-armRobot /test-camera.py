#!/usr/bin/env python3
"""
Open one USB camera by name (macOS AVFoundation) and show it using OpenCV.

Usage:
    python test-camera.py                        # uses default for known name
    python test-camera.py --name "3D USB Camera"
    python test-camera.py --index 0              # use numeric index directly

The script attempts to use `ffmpeg -f avfoundation -list_devices true -i ""` to map device names to AVFoundation indices.
If ffmpeg is not available it will fall back to probing indices 0..7.

Press 'q' in the window to quit. If permission is required, macOS will ask for Camera access.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# On macOS mjpython sets MJPYTHON_BIN in the environment. OpenCV GUI (imshow)
# often misbehaves under the mjpython launcher. Detect this and exit with a
# clear message so the user runs the script with the venv python instead.
if os.environ.get('MJPYTHON_BIN'):
    sys.stderr.write(
        "Detected mjpython environment. OpenCV GUI windows may fail under mjpython.\n"
        "Please run this script with the venv Python instead, for example:\n"
        "  .venv_min_py312/bin/python test-camera.py --index 0\n"
    )
    sys.exit(1)


DEFAULT_NAME = "3D USB Camera"


def ffmpeg_list_avfoundation_devices(ffmpeg_bin: str = "ffmpeg") -> Dict[int, str]:
    """Run ffmpeg to list AVFoundation devices and return a map index->name.

    ffmpeg prints device list to stderr; we parse lines like:
      [AVFoundation indev @ 0x...] [0] 3D USB Camera
    """
    if not shutil_which(ffmpeg_bin):
        raise FileNotFoundError(f"ffmpeg not found: {ffmpeg_bin}")

    # ffmpeg prints to stderr; run and capture
    try:
        proc = subprocess.run([ffmpeg_bin, "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True)
    except Exception as e:
        raise RuntimeError(f"Failed to run ffmpeg: {e}")

    out = proc.stderr + "\n" + proc.stdout
    devices: Dict[int, str] = {}

    # Regex: look for lines containing [<idx>] <name>
    pattern = re.compile(r"\[.*?\]\s*\[(\d+)\]\s*(.+)")
    for line in out.splitlines():
        m = pattern.search(line)
        if m:
            try:
                idx = int(m.group(1))
                name = m.group(2).strip()
                # skip audio lines (we'll filter later if needed) but include all
                devices[idx] = name
            except ValueError:
                continue
    return devices


def shutil_which(cmd: str) -> Optional[str]:
    """Simple which fallback that uses shutil.which if available, else PATH search."""
    try:
        import shutil

        return shutil.which(cmd)
    except Exception:
        return None


def find_index_by_name(devices: Dict[int, str], target: str) -> Optional[int]:
    t = target.lower()
    for idx, name in devices.items():
        if t in name.lower():
            return idx
    return None


def probe_indices_for_working_camera(indices: List[int], backend: int = cv2.CAP_AVFOUNDATION, timeout: float = 1.0) -> Optional[int]:
    """Try indices and return the first that gives a readable frame."""
    for i in indices:
        cap = cv2.VideoCapture(i, backend)
        if not cap.isOpened():
            cap.release()
            continue
        start = time.time()
        ok_frame = False
        while time.time() - start < timeout:
            ret, frame = cap.read()
            if ret and frame is not None:
                ok_frame = True
                break
            time.sleep(0.05)
        cap.release()
        if ok_frame:
            return i
    return None


def open_and_show_camera(idx: int) -> None:
    backend = cv2.CAP_AVFOUNDATION
    cap = cv2.VideoCapture(idx, backend)

    # Threaded capture helper to always provide the latest frame
    import threading

    class ThreadedCapture:
        def __init__(self, cap: cv2.VideoCapture):
            self._cap = cap
            self._lock = threading.Lock()
            self._running = False
            self._latest = None
            self._thread = None

        def start(self):
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

        def _run(self):
            while self._running:
                try:
                    ret, frame = self._cap.read()
                    if not ret:
                        time.sleep(0.001)
                        continue
                    with self._lock:
                        self._latest = frame
                except Exception:
                    time.sleep(0.001)

        def read_latest(self):
            with self._lock:
                return None if self._latest is None else self._latest.copy()

        def stop(self):
            self._running = False
            if self._thread is not None:
                self._thread.join(timeout=0.5)

        def release(self):
            try:
                self._cap.release()
            except Exception:
                pass

    tc = ThreadedCapture(cap)
    tc.start()

    # Set safe defaults to reduce USB bandwidth / driver issues
    SAFE_W = 640
    SAFE_H = 480
    SAFE_FPS = 120
    def set_safe_props(cap):
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, SAFE_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SAFE_H)
            cap.set(cv2.CAP_PROP_FPS, SAFE_FPS)
        except Exception:
            pass

    set_safe_props(cap)

    if not cap.isOpened():
        print(f"Failed to open camera index {idx}")
        cap.release()
        return

    print(f"Opened camera: {idx}. Press 'q' to quit.")
    # retry/reopen parameters
    RETRY_READS = 5
    REOPEN_ON_FAIL = True
    while True:
        try:
            # read latest frame from threaded capture
            f = tc.read_latest()
            ret = f is not None

            # If a read failed try a few quick retries before deciding
            if not ret or f is None:
                retry_ok = False
                for _ in range(RETRY_READS):
                    time.sleep(0.05)
                    ret, f = cap.read()
                    if ret and f is not None:
                        retry_ok = True
                        break
                if not retry_ok:
                    print("Lost frame from camera (after retries)")
                    if REOPEN_ON_FAIL:
                        print(f"Attempting to reopen camera {idx}...")
                        tc.stop()
                        tc.release()
                        cap = cv2.VideoCapture(idx, backend)
                        set_safe_props(cap)
                        tc = ThreadedCapture(cap)
                        tc.start()
                        time.sleep(0.2)
                        f = tc.read_latest()
                        ret = f is not None
                        if not (ret and f is not None):
                            print("Reopen failed for camera")
                            break
                    else:
                        break

            if not ret or f is None:
                print("Lost frame from camera")
                break

            # Normalize frame: ensure numpy array, 3 channels (BGR), and uint8 dtype
            def normalize(frame):
                if not isinstance(frame, np.ndarray):
                    raise ValueError("Frame is not a numpy array")
                # Convert grayscale to BGR
                if frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                # Convert 4-channel (BGRA/RGBA) to BGR
                if frame.ndim == 3 and frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                # Ensure dtype
                if frame.dtype != np.uint8:
                    try:
                        frame = frame.astype(np.uint8)
                    except Exception:
                        raise
                return frame

            f = normalize(f)

            cv2.imshow("USB Camera", f)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        except Exception as e:
            print(f"Error showing frame: {e}")
            break

    tc.stop()
    tc.release()
    cv2.destroyAllWindows()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Open one USB camera by name on macOS (AVFoundation)")
    parser.add_argument("--name", type=str, default=DEFAULT_NAME)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--probe-range", type=int, nargs=2, default=[0, 7],
                        help="Range of indices to probe if ffmpeg is not available")
    args = parser.parse_args(argv)

    idx = args.index

    # If index supplied, use it directly
    if idx is not None:
        open_and_show_camera(idx)
        return 0

    # Get device list from ffmpeg if possible
    devices: Dict[int, str] = {}
    try:
        devices = ffmpeg_list_avfoundation_devices()
        if devices:
            print("Detected AVFoundation devices (index -> name):")
            for k, v in devices.items():
                print(f"  [{k}] {v}")
    except FileNotFoundError:
        print("ffmpeg not found; will probe numeric indices.")
    except Exception as e:
        print(f"ffmpeg listing failed: {e}; will probe numeric indices.")

    if idx is None:
        idx = find_index_by_name(devices, args.name) if devices else None

    # If still None, probe numeric indices
    probe_lo, probe_hi = args.probe_range
    probe_list = list(range(probe_lo, probe_hi + 1))

    if idx is None:
        print(f"Could not find '{args.name}' by name. Probing indices {probe_list} for a working camera...")
        found = probe_indices_for_working_camera(probe_list)
        if found is not None:
            print(f"Using probed index {found} for camera")
            idx = found
        else:
            print("Failed to find working camera")
            return 2

    open_and_show_camera(idx)
    return 0


if __name__ == "__main__":
    sys.exit(main())
