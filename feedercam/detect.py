"""YOLOv8 bird detection with ByteTrack tracking (via Ultralytics)."""

from .visits import Detection


class BirdDetector:
    def __init__(self, weights="yolov8n.pt", imgsz=640, conf=0.35,
                 tracker="bytetrack.yaml", device=None):
        from ultralytics import YOLO  # imported here so tests don't need it

        self.model = YOLO(weights)
        # The pretrained COCO model already knows "bird" (class 14); look it up
        # by name so a custom-trained model with different classes also works.
        self.bird_class = next(i for i, name in self.model.names.items() if name == "bird")
        self.imgsz = imgsz
        self.conf = conf
        self.tracker = tracker
        self.device = device

    def track(self, frame) -> list:
        """Detect birds in one frame. Each detection carries a persistent track ID."""
        result = self.model.track(
            frame,
            persist=True,  # keep tracker state between calls so IDs carry over
            classes=[self.bird_class],
            conf=self.conf,
            imgsz=self.imgsz,
            tracker=self.tracker,
            device=self.device,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None:  # no birds, or none confirmed as tracks yet
            return []
        return [
            Detection(int(tid), tuple(xyxy), float(conf))
            for tid, xyxy, conf in zip(boxes.id.tolist(), boxes.xyxy.tolist(), boxes.conf.tolist())
        ]
