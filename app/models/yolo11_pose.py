import numpy as np
from typing import Any, Dict

from . import BaseModel
from app.schemas.shape import Shape
from app.core.registry import register_model


@register_model(
    "yolo11n_pose",
    "yolo11s_pose",
    "yolo11m_pose",
    "yolo11l_pose",
    "yolo11x_pose",
)
class YOLO11Pose(BaseModel):
    """YOLO11 pose estimation model."""

    def load(self):
        """Load YOLO pose estimation model."""
        from ultralytics import YOLO

        model_path = self.params.get("model_path", "yolo11n-pose.pt")
        device = self.params.get("device", "cpu")

        self.model = YOLO(model_path)
        self.model.to(device)

        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy_img, verbose=False)

    def predict(
        self, image: np.ndarray, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute pose estimation.

        Args:
            image: Input image in BGR format.
            params: Inference parameters.

        Returns:
            Dictionary with pose estimation results.
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
        kpt_threshold = self.params.get("kpt_threshold", 0.5)

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            keypoints = result.keypoints
            if keypoints is None:
                continue

            for i, (box, kpts) in enumerate(zip(boxes, keypoints)):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = result.names[cls]

                xyxy = box.xyxy[0].cpu().numpy()
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
                    group_id=i + 1,
                )
                shapes.append(bbox_shape)

                kpts_data = kpts.data[0].cpu().numpy()
                for j, kpt in enumerate(kpts_data):
                    if len(kpt) == 2:
                        x, y = kpt
                        s = 1.0
                    else:
                        x, y, s = kpt

                    if s < kpt_threshold or (x == 0 and y == 0):
                        continue

                    keypoint_shape = Shape(
                        label=f"{label}_kpt_{j}",
                        shape_type="point",
                        points=[[float(x), float(y)]],
                        score=float(s),
                        group_id=i + 1,
                    )
                    shapes.append(keypoint_shape)

        return {"shapes": shapes, "description": ""}

    def unload(self):
        """Release model resources."""
        if hasattr(self, "model"):
            del self.model
