"""Cut a detected bird out of a frame."""


def square_box(xyxy, frame_w, frame_h, padding=0.15):
    """Expand a box to a padded square around its center, clipped to the frame.

    Square crops resize to the classifier's 224x224 input without squashing
    the bird, and the padding keeps tail and beak tips that YOLO often clips.
    """
    x1, y1, x2, y2 = xyxy
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1 + 2 * padding)
    side = min(side, frame_w, frame_h)
    left = min(max(cx - side / 2, 0), frame_w - side)
    top = min(max(cy - side / 2, 0), frame_h - side)
    return int(left), int(top), int(left + side), int(top + side)


def crop(frame, xyxy, padding=0.15):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = square_box(xyxy, w, h, padding)
    return frame[y1:y2, x1:x2].copy()
