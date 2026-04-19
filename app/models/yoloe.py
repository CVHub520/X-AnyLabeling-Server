import cv2
import numpy as np
from loguru import logger
from typing import Any, Dict, List, Optional

from . import BaseModel
from app.schemas.shape import Shape
from app.core.registry import register_model


@register_model(
    # YOLOE-26 text/visual-prompt variants
    "yoloe_26n_seg",
    "yoloe_26s_seg",
    "yoloe_26m_seg",
    "yoloe_26l_seg",
    "yoloe_26x_seg",
    # YOLOE-26 prompt-free variants
    "yoloe_26n_seg_pf",
    "yoloe_26s_seg_pf",
    "yoloe_26m_seg_pf",
    "yoloe_26l_seg_pf",
    "yoloe_26x_seg_pf",
    # YOLOE-11 text/visual-prompt variants
    "yoloe_11s_seg",
    "yoloe_11m_seg",
    "yoloe_11l_seg",
    # YOLOE-11 prompt-free variants
    "yoloe_11s_seg_pf",
    "yoloe_11m_seg_pf",
    "yoloe_11l_seg_pf",
    # YOLOE-v8 text/visual-prompt variants
    "yoloe_v8s_seg",
    "yoloe_v8m_seg",
    "yoloe_v8l_seg",
    # YOLOE-v8 prompt-free variants
    "yoloe_v8s_seg_pf",
    "yoloe_v8m_seg_pf",
    "yoloe_v8l_seg_pf",
)
class YOLOESegmentation(BaseModel):
    """YOLOE open-vocabulary instance segmentation model.

    Supports three inference modes determined by the request params:
      - Text prompt  : params["text_prompt"] = "person.bus.car"
      - Visual prompt: params["marks"] = [{"type": "rectangle",
                                           "data": [x1, y1, x2, y2]}]
      - Prompt-free  : no text_prompt or marks → uses internal vocabulary
                       (load a *-pf.pt checkpoint for best results)
    """

    # Static helpers

    @staticmethod
    def mask_to_polygon(
        mask: np.ndarray, epsilon_factor: float = 0.001
    ) -> np.ndarray:
        """Convert a binary float mask to an (N, 2) polygon array."""
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return np.zeros((0, 2), dtype=np.float32)

        contour = max(contours, key=cv2.contourArea)

        if epsilon_factor > 0:
            eps = epsilon_factor * cv2.arcLength(contour, True)
            contour = cv2.approxPolyDP(contour, eps, True)

        return contour.reshape(-1, 2).astype(np.float32)

    @staticmethod
    def _parse_text_prompt(text_prompt: str) -> List[str]:
        """Split a dot- or comma-separated class string into a list."""
        for sep in (".", ","):
            if sep in text_prompt:
                return [c.strip() for c in text_prompt.split(sep) if c.strip()]
        stripped = text_prompt.strip()
        return [stripped] if stripped else []

    # Lifecycle

    def load(self):
        """Load the YOLOE model and warm it up."""
        from ultralytics import YOLOE

        model_path = self.params.get("model_path", "yoloe-26s-seg.pt")
        self._device = self.params.get("device", "cpu")

        self._is_pf_model = "pf" in model_path.lower()
        self._current_classes: Optional[List[str]] = None

        self.model = YOLOE(model_path)

        # NOTE: Do NOT call model.to(device) here. Calling model.to() before
        # model.predict() corrupts the VP predictor path - ultralytics sets an
        # internal device flag that makes the VP predictor produce 0 detections.
        # Instead, pass device= per-request via **kwargs to predict(), which is
        # the supported pattern for all three YOLOE modes.

    def unload(self):
        """Release model resources."""
        if hasattr(self, "model"):
            del self.model

    # Inference

    def predict(
        self, image: np.ndarray, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute YOLOE inference.

        Args:
            image: Input image in BGR format (H x W x 3).
            params: Runtime parameters from the client request:
                - conf_threshold (float): Confidence threshold.
                - iou_threshold  (float): NMS IoU threshold.
                - epsilon_factor (float): Polygon approximation factor.
                - show_boxes     (bool):  Also emit bounding-box shapes.
                - text_prompt    (str):   Dot/comma-separated class names.
                - marks          (list):  Visual-prompt bounding boxes.

        Returns:
            Dict with "shapes" (List[Shape]) and "description" (str).
        """
        conf_threshold = params.get(
            "conf_threshold", self.params.get("conf_threshold", 0.25)
        )
        iou_threshold = params.get(
            "iou_threshold", self.params.get("iou_threshold", 0.45)
        )
        epsilon_factor = params.get(
            "epsilon_factor", self.params.get("epsilon_factor", 0.001)
        )
        show_boxes = params.get(
            "show_boxes", self.params.get("show_boxes", False)
        )

        text_prompt: Optional[str] = params.get("text_prompt")
        marks: Optional[List[Dict]] = params.get("marks")

        orig_h, orig_w = image.shape[:2]

        # Mode selection
        if text_prompt:
            results = self._predict_text(
                image, text_prompt, conf_threshold, iou_threshold
            )
        elif marks:
            results = self._predict_visual(
                image, marks, conf_threshold, iou_threshold
            )
        else:
            results = self._predict_prompt_free(
                image, conf_threshold, iou_threshold
            )

        # Post-process
        shapes = []
        logger.debug(f"[YOLOE] results count={len(results) if results else 0}")
        for result in results:
            boxes = result.boxes
            n_boxes = len(boxes) if boxes is not None else 0
            n_masks = (
                len(result.masks)
                if (hasattr(result, "masks") and result.masks is not None)
                else 0
            )
            logger.debug(
                f"[YOLOE] boxes={n_boxes} masks={n_masks} names={result.names}"
            )

            if hasattr(result, "masks") and result.masks is not None:
                mask_data = result.masks.data.cpu().numpy()

                for i, mask in enumerate(mask_data):
                    if boxes is None or i >= len(boxes):
                        continue

                    cls = int(boxes[i].cls[0])
                    conf = float(boxes[i].conf[0])
                    label = result.names.get(cls, str(cls))

                    # Resize mask to original image dimensions if needed
                    mask_h, mask_w = mask.shape
                    if mask_h != orig_h or mask_w != orig_w:
                        mask = cv2.resize(
                            mask.astype(np.float32),
                            (orig_w, orig_h),
                            interpolation=cv2.INTER_LINEAR,
                        )

                    points = self.mask_to_polygon(mask, epsilon_factor)
                    if len(points) < 3:
                        continue

                    points_list = [[float(x), float(y)] for x, y in points]
                    if points_list and points_list[0] != points_list[-1]:
                        points_list.append(points_list[0])

                    shapes.append(
                        Shape(
                            label=label,
                            shape_type="polygon",
                            points=points_list,
                            score=conf,
                        )
                    )

                    if show_boxes:
                        xyxy = boxes[i].xyxy[0].cpu().numpy()
                        shapes.append(
                            Shape(
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
                        )
            elif boxes is not None:
                # Fallback: bounding boxes only (no masks)
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = result.names.get(cls, str(cls))
                    shapes.append(
                        Shape(
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
                    )

        return {"shapes": shapes, "description": ""}

    # Mode-specific helpers

    def _predict_text(
        self,
        image: np.ndarray,
        text_prompt: str,
        conf: float,
        iou: float,
    ):
        """Run inference with text prompt, caching class embeddings."""
        classes = self._parse_text_prompt(text_prompt)
        if not classes:
            return []

        if classes != self._current_classes:
            self.model.set_classes(classes)
            self._current_classes = classes
            # Reset predictor so VP state doesn't bleed into text-prompt mode
            self.model.predictor = None

        return self.model(
            image, conf=conf, iou=iou, verbose=False, device=self._device
        )

    def _predict_visual(
        self,
        image: np.ndarray,
        marks: List[Dict],
        conf: float,
        iou: float,
    ):
        """Run inference with visual prompts from drawn bounding boxes.

        Each mark is expected to be:
            {"type": "rectangle", "data": [x1, y1, x2, y2]}
        Coordinates are in absolute pixel space of *image*.

        The official API requires:
            visual_prompts = dict(
                bboxes=np.array([[x1, y1, x2, y2], ...]),
                cls=np.array([0, 1, ...]),  # sequential IDs starting at 0
            )
        passed to model.predict(..., visual_prompts=...,
                                 predictor=YOLOEVPSegPredictor)
        """
        try:
            from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
        except ImportError:
            logger.warning(
                "[YOLOE] YOLOEVPSegPredictor not available, falling back to prompt-free"
            )
            return self._predict_prompt_free(image, conf, iou)

        boxes = []
        for mark in marks:
            if mark.get("type") == "rectangle":
                data = mark.get("data", [])
                if len(data) == 4:
                    boxes.append(
                        [
                            float(data[0]),
                            float(data[1]),
                            float(data[2]),
                            float(data[3]),
                        ]
                    )

        if not boxes:
            logger.warning(
                "[YOLOE] No valid rectangle marks, falling back to prompt-free"
            )
            return self._predict_prompt_free(image, conf, iou)

        logger.debug(f"[YOLOE] VP mode: {len(boxes)} box(es): {boxes}")
        visual_prompts = dict(
            bboxes=np.array(boxes, dtype=np.float32),
            # Sequential class IDs starting from 0, one per box
            cls=np.arange(len(boxes), dtype=np.int32),
        )

        # Reset any cached predictor so the VP predictor initialises fresh
        # with the correct visual embeddings rather than reusing a stale
        # text-prompt or standard predictor state.
        self.model.predictor = None

        logger.debug(
            f"[YOLOE] Calling predict with visual_prompts bboxes={visual_prompts['bboxes']} cls={visual_prompts['cls']}"
        )
        return self.model.predict(
            image,
            visual_prompts=visual_prompts,
            predictor=YOLOEVPSegPredictor,
            conf=conf,
            iou=iou,
            verbose=False,
            device=self._device,
        )

    def _predict_prompt_free(
        self,
        image: np.ndarray,
        conf: float,
        iou: float,
    ):
        """Run inference in prompt-free mode (internal LVIS vocabulary)."""
        return self.model(
            image, conf=conf, iou=iou, verbose=False, device=self._device
        )
