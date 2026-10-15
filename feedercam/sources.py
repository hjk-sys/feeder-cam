"""Frame sources: the Pi camera, a USB webcam, or a video file.

Every source yields (timestamp, frame) pairs, where frame is a BGR numpy array
(OpenCV's channel order). For video files the timestamp is computed from the
frame number, so a recorded clip produces the same visit times on every run.
"""

import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2


class PiCameraSource:
    """Raspberry Pi camera module via picamera2 (only importable on the Pi)."""

    def __init__(self, width, height, fps):
        from picamera2 import Picamera2

        self.cam = Picamera2()
        # picamera2's "RGB888" is stored B,G,R in memory, which is what OpenCV expects.
        config = self.cam.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            controls={"FrameRate": fps},
        )
        self.cam.configure(config)
        self.cam.start()
        time.sleep(1)  # let auto-exposure settle

    def __iter__(self):
        while True:
            yield datetime.now(), self.cam.capture_array()

    def close(self):
        self.cam.stop()


class OpenCVSource:
    """A USB webcam (pass an int index) or a video file (pass a path)."""

    def __init__(self, target, width=None, height=None, start_time=None):
        self.is_file = not isinstance(target, int)
        self.cap = cv2.VideoCapture(str(target) if self.is_file else target)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {target}")
        if not self.is_file and width and height:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        if self.is_file and start_time is None:
            # Best guess at when the clip was recorded: the file's modified time.
            start_time = datetime.fromtimestamp(Path(target).stat().st_mtime)
        self.start_time = start_time

    def __iter__(self):
        frame_idx = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                return
            if self.is_file:
                ts = self.start_time + timedelta(seconds=frame_idx / self.fps)
            else:
                ts = datetime.now()
            frame_idx += 1
            yield ts, frame

    def close(self):
        self.cap.release()


def open_source(cam_cfg: dict, override=None):
    """Open the source named in the config, or `override` (e.g. a video path from the CLI)."""
    source = override or cam_cfg["source"]
    if source == "picamera2":
        return PiCameraSource(cam_cfg["width"], cam_cfg["height"], cam_cfg["fps"])
    if source == "usb":
        return OpenCVSource(cam_cfg["usb_index"], cam_cfg["width"], cam_cfg["height"])
    return OpenCVSource(Path(source))
