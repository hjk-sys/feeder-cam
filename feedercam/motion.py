"""Cheap motion detection, so the detector only runs when something is moving.

Compares each frame to a slowly updating background average. Slow changes such
as clouds or shadows get absorbed into the background; a bird landing does not.
"""

import cv2
import numpy as np


class MotionDetector:
    def __init__(self, downscale_width=320, threshold=25, min_area_frac=0.003, learn_rate=0.05):
        self.downscale_width = downscale_width
        self.threshold = threshold
        self.min_area_frac = min_area_frac
        self.learn_rate = learn_rate
        self.background = None

    def _prep(self, frame):
        h, w = frame.shape[:2]
        scale = self.downscale_width / w
        small = cv2.resize(frame, (self.downscale_width, int(h * scale)))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)

    def update(self, frame) -> float:
        """Return the fraction of the frame that changed (0.0 to 1.0)."""
        gray = self._prep(frame)
        if self.background is None:
            self.background = gray
            return 0.0
        diff = cv2.absdiff(gray, self.background)
        cv2.accumulateWeighted(gray, self.background, self.learn_rate)
        changed = diff > self.threshold
        return float(changed.mean())

    def is_motion(self, frame) -> bool:
        return self.update(frame) >= self.min_area_frac
