import numpy as np
from typing import Any, Dict

from . import BaseModel
from app.schemas.shape import Shape
from app.core.registry import register_model


@register_model("yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x")
class YOLO11Detection(BaseModel):
    """YOLO11 object detection model."""

    def load(self):
        """Load YOLO model."""
        from ultralytics import YOLO

        model_path = self.params.get("model_path", "yolo11n.pt")
        device = self.params.get("device", "cpu")

        self.model = YOLO(model_path)
        self.model.to(device)

        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy_img, verbose=False)

    def predict(
        self, image: np.ndarray, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute object detection.

        Args:
            image: Input image in BGR format.
            params: Inference parameters.

        Returns:
            Dictionary with detection results.
        """
        conf_threshold = params.get(
            "conf_threshold", self.params.get("conf_threshold", 0.25)
        )
        iou_threshold = params.get(
            "iou_threshold", self.params.get("iou_threshold", 0.45)
        )

        results = self.model(
            image, conf=conf_threshold, iou=iou_threshold, verbose=False
        )

        shapes = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
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
                    )
                    shapes.append(shape)

        return {"shapes": shapes, "description": ""}

    def unload(self):
        """Release model resources."""
        if hasattr(self, "model"):
            del self.model
