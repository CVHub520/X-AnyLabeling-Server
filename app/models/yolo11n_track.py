import numpy as np
from typing import Any, Dict

from .base import BaseModel
from app.schemas.shape import Shape


class YOLO11nDetectionTrack(BaseModel):
    """YOLO11n object detection with tracking model."""

    def load(self):
        """Load YOLO model with tracking support."""
        from ultralytics import YOLO

        model_path = self.params.get("model_path", "yolo11n.pt")
        device = self.params.get("device", "cpu")

        self.model = YOLO(model_path)
        self.model.to(device)

        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy_img, verbose=False)

        self.tracker_type = self.params.get("tracker_type", "bytetrack")
        self.track_history = {}

    def predict(
        self, image: np.ndarray, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute object detection with tracking.

        Args:
            image: Input image in BGR format.
            params: Inference parameters.

        Returns:
            Dictionary with detection and tracking results.
        """
        if params.get("reset_tracker"):
            self.reset_tracker()

        conf_threshold = params.get(
            "conf_threshold", self.params.get("conf_threshold", 0.25)
        )
        iou_threshold = params.get(
            "iou_threshold", self.params.get("iou_threshold", 0.45)
        )

        results = self.model.track(
            image,
            conf=conf_threshold,
            iou=iou_threshold,
            persist=True,
            tracker=f"{self.tracker_type}.yaml",
            verbose=False,
        )

        shapes = []
        for result in results:
            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    track_id = int(box.id[0]) if box.id is not None else None
                    label = result.names[cls]

                    shape = Shape(
                        label=label,
                        shape_type="rectangle",
                        points=[
                            [float(xyxy[0]), float(xyxy[1])],
                            [float(xyxy[2]), float(xyxy[1])],
                            [float(xyxy[2]), float(xyxy[3])],
                            [float(xyxy[0]), float(xyxy[3])],
                        ],
                        score=conf,
                        group_id=track_id,
                    )
                    shapes.append(shape)

        return {"shapes": shapes, "description": ""}

    def reset_tracker(self):
        """Reset tracker state."""
        self.track_history = {}
        if (
            hasattr(self.model, "predictor")
            and self.model.predictor is not None
        ):
            if (
                hasattr(self.model.predictor, "trackers")
                and len(self.model.predictor.trackers) > 0
            ):
                self.model.predictor.trackers[0].reset()

    def unload(self):
        """Release model resources."""
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "track_history"):
            self.track_history.clear()
