import numpy as np
import re
from loguru import logger
from typing import Any, Dict, List

from . import BaseModel
from .paddleocr_vl_1_5 import PaddleOCRVL15
from .pp_doclayout_v3 import PPDocLayoutV3
from app.core.registry import register_model
from app.schemas.shape import Shape

TEXT_TASK_LABELS = {
    "text",
    "doc_title",
    "vision_footnote",
    "figure_title",
    "inline_formula",
    "paragraph_title",
    "content",
    "vertical_text",
    "aside_text",
    "abstract",
    "footnote",
    "reference",
    "reference_content",
    "footer",
    "number",
    "header",
}

TASK_BY_LABEL = {
    "table": "table",
    "chart": "chart",
    "seal": "seal",
    "formula": "formula",
    "algorithm": "formula",
    "inline_formula": "formula",
    "display_formula": "formula",
    "formula_number": "formula",
}

IMAGE_ONLY_LABELS = {
    "image",
    "header_image",
    "footer_image",
}


@register_model("ppocr_layoutstructv3_vl_1_5")
class PPOCRLayoutStructV3VL15(BaseModel):
    """Server-side PPOCR document pipeline model."""

    @staticmethod
    def _normalize_label(label: Any) -> str:
        return (
            re.sub(r"[\s\-]+", "_", str(label or "").strip()).casefold()
            or "text"
        )

    @staticmethod
    def _task_for_label(label: str) -> str | None:
        normalized = PPOCRLayoutStructV3VL15._normalize_label(label)
        if normalized in IMAGE_ONLY_LABELS:
            return None
        if normalized in TASK_BY_LABEL:
            return TASK_BY_LABEL[normalized]
        if normalized in TEXT_TASK_LABELS:
            return "ocr"
        return "ocr"

    @staticmethod
    def _normalize_points(points: Any) -> List[List[float]]:
        normalized: List[List[float]] = []
        if not isinstance(points, list):
            return normalized
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            try:
                normalized.append([float(point[0]), float(point[1])])
            except (TypeError, ValueError):
                continue
        return normalized

    @staticmethod
    def _shape_sort_key(shape: Shape) -> tuple[float, float]:
        points = PPOCRLayoutStructV3VL15._normalize_points(shape.points)
        if not points:
            return (0.0, 0.0)
        min_x = min(point[0] for point in points)
        min_y = min(point[1] for point in points)
        return (min_y, min_x)

    @staticmethod
    def _points_to_bbox(
        points: List[List[float]],
        image_width: int,
        image_height: int,
    ) -> List[int]:
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        x1 = max(0, int(round(min(xs))))
        y1 = max(0, int(round(min(ys))))
        x2 = min(image_width, int(round(max(xs))))
        y2 = min(image_height, int(round(max(ys))))
        if x2 <= x1:
            x2 = min(image_width, x1 + 1)
        if y2 <= y1:
            y2 = min(image_height, y1 + 1)
        return [x1, y1, x2, y2]

    @staticmethod
    def _crop_image(image: np.ndarray, bbox: List[int]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        return image[y1:y2, x1:x2]

    def _build_submodel_config(
        self,
        model_id: str,
        display_name: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "model_id": model_id,
            "display_name": display_name,
            "params": params,
            "widgets": [],
        }

    def load(self):
        """Load pipeline dependencies."""
        layout_params = self.params.get("layout", {}) or {}
        content_params = self.params.get("content", {}) or {}

        layout_config = self._build_submodel_config(
            "pp_doclayout_v3",
            "PP-DocLayoutV3",
            layout_params,
        )
        content_config = self._build_submodel_config(
            "paddleocr_vl_1_5",
            "PaddleOCR-VL-1.5",
            content_params,
        )

        self.layout_model = PPDocLayoutV3(layout_config)
        self.layout_model.load()

        self.content_model = PaddleOCRVL15(content_config)
        self.content_model.load()

        logger.info("PPOCR pipeline model loaded")

    def predict(
        self,
        image: np.ndarray,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute server-side PPOCR pipeline for one page image."""
        image_height, image_width = image.shape[:2]
        layout_params = params.get("layout_params", {}) or {}
        layout_result = self.layout_model.predict(image, layout_params)
        layout_shapes = layout_result.get("shapes") or []

        sorted_shapes = sorted(
            (shape for shape in layout_shapes if isinstance(shape, Shape)),
            key=self._shape_sort_key,
        )

        parsing_res_list: List[Dict[str, Any]] = []
        for index, shape in enumerate(sorted_shapes, start=1):
            points = self._normalize_points(shape.points)
            if not points:
                continue

            bbox = self._points_to_bbox(points, image_width, image_height)
            label = self._normalize_label(shape.label)
            task = self._task_for_label(label)
            content = ""

            if task is not None:
                crop_image = self._crop_image(image, bbox)
                if crop_image.size > 0:
                    try:
                        content_result = self.content_model.predict(
                            crop_image,
                            {"current_task": task},
                        )
                        content = str(content_result.get("description") or "")
                    except Exception as exc:
                        logger.warning(
                            "PPOCR pipeline block recognition failed "
                            f"(block={index}, label={label}, task={task}): {exc}"
                        )

            block_data = {
                "block_label": label,
                "block_content": content,
                "block_bbox": bbox,
                "block_id": index,
                "block_order": index,
                "group_id": int(shape.group_id or index),
                "global_block_id": index,
                "global_group_id": int(shape.group_id or index),
                "block_polygon_points": points,
            }
            parsing_res_list.append(block_data)

        return {
            "shapes": layout_shapes,
            "description": "",
            "prunedResult": {
                "page_count": 1,
                "width": image_width,
                "height": image_height,
                "model_settings": {
                    "layout_analyzer": "PP-DocLayoutV3",
                    "content_recognizer": "PaddleOCR-VL-1.5",
                    "pipeline_model": self.model_id,
                },
                "parsing_res_list": parsing_res_list,
            },
            "markdown": {
                "text": "\n\n".join(
                    block.get("block_content", "")
                    for block in parsing_res_list
                    if block.get("block_content")
                ),
                "images": {},
            },
            "outputImages": {},
        }

    def unload(self):
        """Unload dependencies."""
        if hasattr(self, "layout_model"):
            self.layout_model.unload()
            del self.layout_model
        if hasattr(self, "content_model"):
            self.content_model.unload()
            del self.content_model
