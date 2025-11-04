import cv2
import numpy as np
from typing import Any, Dict

from .base import BaseModel
from app.schemas.shape import Shape


class YOLO11nSegmentation(BaseModel):
    """YOLO11n instance segmentation model."""

    @staticmethod
    def mask_to_polygon(
        mask: np.ndarray, epsilon_factor: float = 0.001
    ) -> np.ndarray:
        """Convert binary mask to polygon contour points."""
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return np.zeros((0, 2), dtype=np.float32)

        contour = max(contours, key=cv2.contourArea)

        if epsilon_factor > 0:
            epsilon = epsilon_factor * cv2.arcLength(contour, True)
            contour = cv2.approxPolyDP(contour, epsilon, True)

        points = contour.reshape(-1, 2).astype(np.float32)
        return points

    def load(self):
        """Load YOLO segmentation model."""
        from ultralytics import YOLO

        model_path = self.params.get("model_path", "yolo11n-seg.pt")
        device = self.params.get("device", "cpu")

        self.model = YOLO(model_path)
        self.model.to(device)

        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy_img, verbose=False)

    def predict(
        self, image: np.ndarray, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute instance segmentation.

        Args:
            image: Input image in BGR format.
            params: Inference parameters.

        Returns:
            Dictionary with segmentation results.
        """
        conf_threshold = params.get(
            "conf_threshold", self.params.get("conf_threshold", 0.25)
        )
        iou_threshold = params.get(
            "iou_threshold", self.params.get("iou_threshold", 0.70)
        )

        results = self.model(
            image, conf=conf_threshold, iou=iou_threshold, verbose=False
        )

        shapes = []
        show_boxes = self.params.get("show_boxes", False)
        epsilon_factor = params.get(
            "epsilon_factor", self.params.get("epsilon_factor", 0.001)
        )

        for result in results:
            boxes = result.boxes

            if hasattr(result, "masks") and result.masks is not None:
                mask_data = result.masks.data.cpu().numpy()

                for i, mask in enumerate(mask_data):
                    cls = int(boxes[i].cls[0])
                    conf = float(boxes[i].conf[0])
                    label = result.names[cls]

                    points = self.mask_to_polygon(mask, epsilon_factor)

                    if len(points) < 3:
                        continue

                    if show_boxes:
                        xyxy = boxes[i].xyxy[0].cpu().numpy()
                        bbox_shape = Shape(
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
                        shapes.append(bbox_shape)

                    mask_shape = Shape(
                        label=label,
                        shape_type="polygon",
                        points=[[float(x), float(y)] for x, y in points],
                        score=conf,
                    )
                    shapes.append(mask_shape)
            elif boxes is not None:
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
