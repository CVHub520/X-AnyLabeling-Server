from loguru import logger
from PIL import Image
from typing import Any, Dict, List

import numpy as np
import torch

from . import BaseModel
from app.schemas.shape import Shape
from app.core.registry import register_model


@register_model("segment_anything_3")
class SegmentAnything3(BaseModel):
    """Segment Anything Model 3 for image segmentation with text or visual prompts."""

    def load(self):
        """Load SAM3 model and initialize components."""
        import sys
        import os

        sam3_parent_dir = os.path.join(os.path.dirname(__file__))
        if sam3_parent_dir not in sys.path:
            sys.path.insert(0, sam3_parent_dir)

        from sam3.model_builder import build_sam3_image_model

        bpe_path = self.params.get("bpe_path")
        model_path = self.params.get("model_path")
        device = self.params.get("device", "cuda:0")
        self.device = "cuda" if "cuda" in device else device

        logger.info(f"Loading SAM3 model from {model_path}")
        self.model = build_sam3_image_model(
            bpe_path=bpe_path,
            device=self.device,
            checkpoint_path=model_path,
        )

        if torch.cuda.is_available():
            # turn on tfloat32 for Ampere GPUs
            # https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # use bfloat16 for the entire env. If your card doesn't support it, try float16 instead
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

            # inference mode for the whole env. Disable if you need gradients
            torch.inference_mode().__enter__()

        logger.info("SAM3 model loaded successfully")

    def predict(
        self, image: np.ndarray, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute segmentation based on text or visual prompts.

        Args:
            image: Input image in BGR format.
            params: Inference parameters including text_prompt and marks.

        Returns:
            Dictionary with shapes and description.
        """
        text_prompt = params.get("text_prompt", "")
        marks = params.get("marks", [])

        if marks:
            return self._predict_with_boxes(image, marks, params)
        elif text_prompt:
            return self._predict_with_text(image, text_prompt, params)
        else:
            logger.warning("No prompt provided")
            return {"shapes": [], "description": ""}

    def _predict_with_text(
        self, image: np.ndarray, text_prompt: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute segmentation with text prompt.

        Args:
            image: Input image in BGR format.
            text_prompt: Text description of objects to segment.
            params: Inference parameters.

        Returns:
            Dictionary with shapes and description.
        """
        from sam3.model.sam3_image_processor import Sam3Processor

        conf_thresh = params.get(
            "conf_threshold", self.params.get("conf_threshold", 0.50)
        )
        show_boxes = params.get(
            "show_boxes", self.params.get("show_boxes", True)
        )
        show_masks = params.get(
            "show_masks", self.params.get("show_masks", False)
        )

        logger.info(
            f"Processing text prompt: '{text_prompt}' with conf_threshold={conf_thresh}"
        )

        processor = Sam3Processor(
            self.model, confidence_threshold=conf_thresh, device=self.device
        )

        pil_image = Image.fromarray(image[:, :, ::-1])
        inference_state = processor.set_image(pil_image)

        separators = [",", "."]
        separator_used = None
        for sep in separators:
            if sep in text_prompt:
                separator_used = sep
                break

        if separator_used:
            prompts = [
                p.strip()
                for p in text_prompt.split(separator_used)
                if p.strip()
            ]
        else:
            prompts = [text_prompt.strip()] if text_prompt.strip() else []

        # deduplicate prompts
        prompts = list(dict.fromkeys(prompts))

        all_masks = []
        all_boxes = []
        all_scores = []
        all_labels = []

        for prompt in prompts:
            if not prompt:
                continue
            processor.reset_all_prompts(inference_state)
            results = processor.set_text_prompt(
                state=inference_state, prompt=prompt
            )

            num_objects = len(results["scores"])
            if num_objects > 0:
                all_masks.append(results["masks"])
                all_boxes.append(results["boxes"])
                all_scores.append(results["scores"])
                all_labels.extend([prompt] * num_objects)

        epsilon_factor = params.get(
            "epsilon_factor", self.params.get("epsilon_factor", 0.001)
        )

        if all_masks:
            combined_results = {
                "masks": torch.cat(all_masks, dim=0),
                "boxes": torch.cat(all_boxes, dim=0),
                "scores": torch.cat(all_scores, dim=0),
                "labels": all_labels,
            }
            shapes = self._convert_results_to_shapes(
                combined_results, show_boxes, show_masks, epsilon_factor
            )
        else:
            shapes = []

        return {"shapes": shapes, "description": ""}

    def _predict_with_boxes(
        self,
        image: np.ndarray,
        marks: List[Dict[str, Any]],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute segmentation with visual prompts.

        Args:
            image: Input image in BGR format.
            marks: List of visual prompts with type, label, and data fields.
            params: Inference parameters.

        Returns:
            Dictionary with shapes and description.
        """
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model.box_ops import box_xywh_to_cxcywh
        from sam3.visualization_utils import normalize_bbox

        conf_thresh = params.get(
            "conf_threshold", self.params.get("conf_threshold", 0.50)
        )
        show_boxes = params.get(
            "show_boxes", self.params.get("show_boxes", True)
        )
        show_masks = params.get(
            "show_masks", self.params.get("show_masks", False)
        )

        box_input_xywh = []
        box_labels = []

        logger.info(
            f"Processing visual prompt: '{marks}' with conf_threshold={conf_thresh}"
        )

        for mark in marks:
            if mark.get("type") != "rectangle":
                continue

            label = mark.get("label", 0)
            data = mark.get("data", [])
            if len(data) != 4:
                continue

            x1, y1, x2, y2 = data
            w = x2 - x1
            h = y2 - y1
            box_input_xywh.append([x1, y1, w, h])
            box_labels.append(True if label == 1 else False)

        if not box_input_xywh:
            logger.warning("No valid box prompts provided")
            return {"shapes": [], "description": ""}

        for i, (box, label) in enumerate(zip(box_input_xywh, box_labels)):
            label_value = 1 if label else 0
            logger.debug(
                f"Box {i}: XYWH={box}, Label={label} (value={label_value})"
            )

        processor = Sam3Processor(
            self.model, confidence_threshold=conf_thresh, device=self.device
        )

        pil_image = Image.fromarray(image[:, :, ::-1])
        width, height = pil_image.size
        inference_state = processor.set_image(pil_image)

        box_input_cxcywh = box_xywh_to_cxcywh(
            torch.tensor(box_input_xywh).view(-1, 4)
        )
        norm_boxes_cxcywh = normalize_bbox(
            box_input_cxcywh, width, height
        ).tolist()
        for i, (box, label) in enumerate(zip(norm_boxes_cxcywh, box_labels)):
            logger.debug(f"Box {i}: CXCYWH={box}, Label={label}")

        processor.reset_all_prompts(inference_state)
        for box, label in zip(norm_boxes_cxcywh, box_labels):
            inference_state = processor.add_geometric_prompt(
                state=inference_state, box=box, label=label
            )

        if "scores" in inference_state and len(inference_state["scores"]) > 0:
            num_objects = len(inference_state["scores"])
            for i in range(num_objects):
                score = (
                    inference_state["scores"][i].item()
                    if hasattr(inference_state["scores"][i], 'item')
                    else inference_state["scores"][i]
                )
                box = inference_state["boxes"][i]
                if hasattr(box, 'cpu'):
                    box_np = box.cpu().numpy()
                else:
                    box_np = box
                logger.debug(f"  Object {i}:")
                logger.debug(f"    Score: {score:.4f}")
                logger.debug(f"    Box (XYXY): {box_np}")

                if "masks" in inference_state:
                    mask = inference_state["masks"][i]
                    if hasattr(mask, 'shape'):
                        logger.debug(f"    Mask shape: {mask.shape}")
        else:
            logger.debug("No objects predicted")

        epsilon_factor = params.get(
            "epsilon_factor", self.params.get("epsilon_factor", 0.001)
        )

        shapes = self._convert_results_to_shapes(
            inference_state, show_boxes, show_masks, epsilon_factor
        )

        return {"shapes": shapes, "description": ""}

    def _convert_results_to_shapes(
        self,
        results: Dict[str, Any],
        show_boxes: bool = False,
        show_masks: bool = True,
        epsilon_factor: float = 0.001,
    ) -> List[Shape]:
        """Convert SAM3 results to Shape objects.

        Args:
            results: Dictionary containing masks, boxes, scores, and labels.
            show_boxes: Whether to return bounding boxes.
            show_masks: Whether to return masks as polygons.
            epsilon_factor: Factor for polygon approximation epsilon.

        Returns:
            List of Shape objects.
        """
        shapes = []

        if "scores" not in results or len(results["scores"]) == 0:
            return shapes

        # Convert bfloat16/float16 to float32 before numpy conversion
        boxes = results["boxes"].cpu().float().numpy()
        scores = results["scores"].cpu().float().numpy()
        masks = results["masks"].cpu().float().numpy()
        labels = results.get("labels", [])

        for i in range(len(scores)):
            label = labels[i] if i < len(labels) else "AUTOLABEL_OBJECT"
            score = float(scores[i])

            if show_masks:
                mask = masks[i].squeeze()
                points = self._mask_to_polygon(mask, epsilon_factor)

                if points:
                    shape = Shape(
                        label=label,
                        shape_type="polygon",
                        points=points,
                        score=score,
                    )
                    shapes.append(shape)

            if show_boxes:
                box = boxes[i]
                x1, y1, x2, y2 = box

                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1

                shape = Shape(
                    label=label,
                    shape_type="rectangle",
                    points=[
                        [float(x1), float(y1)],
                        [float(x2), float(y1)],
                        [float(x2), float(y2)],
                        [float(x1), float(y2)],
                    ],
                    score=score,
                )
                shapes.append(shape)

        return shapes

    def _mask_to_polygon(
        self, mask: np.ndarray, epsilon_factor: float = 0.001
    ) -> List[List[float]]:
        """Convert binary mask to polygon points.

        Args:
            mask: Binary mask array.
            epsilon_factor: Factor for polygon approximation epsilon.

        Returns:
            List of polygon points with the last point equal to the first point.
        """
        import cv2

        mask_uint8 = (mask > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return []

        largest_contour = max(contours, key=cv2.contourArea)
        if epsilon_factor > 0:
            epsilon = epsilon_factor * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        else:
            approx = largest_contour

        points = []
        for point in approx:
            x, y = point[0]
            points.append([float(x), float(y)])

        if points and points[0] != points[-1]:
            points.append(points[0])

        return points

    def unload(self):
        """Release model resources."""
        if hasattr(self, "model"):
            del self.model
        logger.info("SAM3 model unloaded")
