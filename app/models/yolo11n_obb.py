import math
import numpy as np
from typing import Any, Dict

from .base import BaseModel
from app.schemas.shape import Shape


class YOLO11nOBB(BaseModel):
    """YOLO11n oriented bounding box detection model."""

    @staticmethod
    def xywhr2xyxyxyxy(box: np.ndarray) -> np.ndarray:
        """Convert rotated box to 4 corner points."""
        x, y, w, h, r = box
        cos_r = np.cos(r)
        sin_r = np.sin(r)

        w_2 = w / 2
        h_2 = h / 2

        points = np.array([[-w_2, -h_2], [w_2, -h_2], [w_2, h_2], [-w_2, h_2]])

        rotation_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])

        rotated_points = points @ rotation_matrix.T
        rotated_points[:, 0] += x
        rotated_points[:, 1] += y

        return rotated_points

    @staticmethod
    def calculate_rotation_theta(poly: np.ndarray) -> float:
        """Calculate the rotation angle of the polygon."""
        x1, y1 = poly[0]
        x2, y2 = poly[1]

        diagonal_vector_x = x2 - x1
        diagonal_vector_y = y2 - y1

        rotation_angle = math.atan2(diagonal_vector_y, diagonal_vector_x)
        rotation_angle_degrees = math.degrees(rotation_angle)

        if rotation_angle_degrees < 0:
            rotation_angle_degrees += 360

        return rotation_angle_degrees / 360 * (2 * math.pi)

    def load(self):
        """Load YOLO OBB model."""
        from ultralytics import YOLO

        model_path = self.params.get("model_path", "yolo11n-obb.pt")
        device = self.params.get("device", "cpu")

        self.model = YOLO(model_path)
        self.model.to(device)

        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy_img, verbose=False)

    def predict(
        self, image: np.ndarray, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute oriented bounding box detection.

        Args:
            image: Input image in BGR format.
            params: Inference parameters.

        Returns:
            Dictionary with OBB detection results.
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

        for result in results:
            if not hasattr(result, "obb") or result.obb is None:
                continue

            obb_boxes = result.obb
            for obb in obb_boxes:
                xywhr = obb.xywhr[0].cpu().numpy()
                conf = float(obb.conf[0])
                cls = int(obb.cls[0])
                label = result.names[cls]

                poly = self.xywhr2xyxyxyxy(xywhr)
                direction = self.calculate_rotation_theta(poly)

                shape = Shape(
                    label=label,
                    shape_type="rotation",
                    points=[
                        [float(poly[0][0]), float(poly[0][1])],
                        [float(poly[1][0]), float(poly[1][1])],
                        [float(poly[2][0]), float(poly[2][1])],
                        [float(poly[3][0]), float(poly[3][1])],
                    ],
                    score=conf,
                    direction=float(direction),
                )
                shapes.append(shape)

        return {"shapes": shapes, "description": ""}

    def unload(self):
        """Release model resources."""
        if hasattr(self, "model"):
            del self.model
