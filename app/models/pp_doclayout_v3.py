import numpy as np
from loguru import logger
from PIL import Image
from typing import Any, Dict

from . import BaseModel
from app.schemas.shape import Shape
from app.core.registry import register_model


@register_model("pp_doclayout_v3")
class PPDocLayoutV3(BaseModel):
    """PP-DocLayoutV3 document layout detection model."""

    @staticmethod
    def _normalize_threshold_mapping(
        threshold_mapping: Any,
    ) -> tuple[dict[int, float], dict[str, float]]:
        by_id: dict[int, float] = {}
        by_label: dict[str, float] = {}
        if not isinstance(threshold_mapping, dict):
            return by_id, by_label

        for raw_key, raw_value in threshold_mapping.items():
            try:
                threshold = float(raw_value)
            except (TypeError, ValueError):
                continue

            if isinstance(raw_key, int):
                by_id[int(raw_key)] = threshold
                continue

            key_text = str(raw_key).strip()
            if not key_text:
                continue
            if key_text.isdigit():
                by_id[int(key_text)] = threshold
                continue
            by_label[key_text.casefold()] = threshold

        return by_id, by_label

    def _resolve_thresholds(
        self, params: Dict[str, Any]
    ) -> tuple[float, dict[int, float], dict[str, float]]:
        conf_threshold = float(
            params.get(
                "conf_threshold", self.params.get("conf_threshold", 0.5)
            )
        )
        threshold_mapping = params.get(
            "threshold", self.params.get("threshold", {})
        )
        by_id, by_label = self._normalize_threshold_mapping(threshold_mapping)
        return conf_threshold, by_id, by_label

    def load(self):
        """Load PP-DocLayoutV3 model and initialize components."""
        import torch
        from transformers import (
            AutoImageProcessor,
            AutoModelForObjectDetection,
        )

        model_path = self.params.get(
            "model_path", "PaddlePaddle/PP-DocLayoutV3_safetensors"
        )
        device = self.params.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.device = device
        self.model = AutoModelForObjectDetection.from_pretrained(model_path)
        self.model.to(device).eval()
        self.processor = AutoImageProcessor.from_pretrained(model_path)

        logger.info(f"PP-DocLayoutV3 model loaded on {device}")

    def predict(
        self, image: np.ndarray, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute layout detection on input image.

        Args:
            image: Input image in BGR format.
            params: Inference parameters including confidence threshold.

        Returns:
            Dictionary with detected layout shapes.
        """
        import torch

        conf_threshold, thresholds_by_id, thresholds_by_label = (
            self._resolve_thresholds(params)
        )
        pil_image = Image.fromarray(image[:, :, ::-1]).convert("RGB")

        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_object_detection(
            outputs, target_sizes=[pil_image.size[::-1]]
        )

        shapes = []
        for result in results:
            scores = result["scores"]
            labels = result["labels"]
            boxes = result["boxes"]
            polygon_points_list = result.get("polygon_points", [])

            has_polygons = len(polygon_points_list) == len(scores)

            for idx, (score, label_id, box) in enumerate(
                zip(scores, labels, boxes)
            ):
                score_val = score.item()

                label_index = label_id.item()
                label = self.model.config.id2label[label_index]
                threshold = thresholds_by_id.get(label_index)
                if threshold is None:
                    threshold = thresholds_by_label.get(
                        str(label).casefold(), conf_threshold
                    )
                if score_val < threshold:
                    continue

                if has_polygons and idx < len(polygon_points_list):
                    poly_pts = polygon_points_list[idx]
                    pts = (
                        poly_pts.tolist()
                        if hasattr(poly_pts, "tolist")
                        else list(poly_pts)
                    )
                    points = [[float(p[0]), float(p[1])] for p in pts]
                else:
                    box_list = box.tolist()
                    xmin, ymin, xmax, ymax = box_list
                    points = [
                        [float(xmin), float(ymin)],
                        [float(xmax), float(ymin)],
                        [float(xmax), float(ymax)],
                        [float(xmin), float(ymax)],
                    ]

                shape = Shape(
                    label=label,
                    shape_type="polygon",
                    points=points,
                    score=float(score_val),
                    group_id=idx + 1,
                )
                shapes.append(shape)

        return {"shapes": shapes}

    def unload(self):
        """Release model resources."""
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "processor"):
            del self.processor
