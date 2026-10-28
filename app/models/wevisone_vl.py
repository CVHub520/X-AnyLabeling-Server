import base64
import re
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from . import BaseModel, parse_prompts
from app.core.registry import register_model
from app.schemas.shape import Shape


@register_model("wevisone_vl")
class WeVisOneVLModel(BaseModel):
    TASKS = {
        "detection": "Object Detection",
        "referring": "Referring Localization",
        "point": "Pointing",
        "count_box": "Counting with Boxes",
        "count_point": "Counting with Points",
        "ocr": "OCR",
        "ocr_grounding": "Text Grounding",
        "segmentation": "Text Segmentation",
        "matting": "Text Matting",
        "box_matting": "Box Matting",
        "visual_prompt": "Visual Prompting",
        "freeform": "Freeform",
    }

    def load(self):
        from wevisone_vl import WeVisOneVL
        from wevisone_vl.visualization import parse_predictions

        self.parse_predictions = parse_predictions
        self.model = WeVisOneVL(
            self.params.get("model_path", "tencent/WeVisOne-VL-2B"),
            backend="transformers",
            device=self.params.get("device", "cuda:0"),
            dtype=self.params.get("dtype", "bfloat16"),
            attention_implementation=self.params.get(
                "attn_implementation", "sdpa"
            ),
            min_pixels=self.params.get("min_pixels", 16384),
            max_pixels=self.params.get("max_pixels", 1003520),
            decoder_batch_size=self.params.get("decoder_batch_size", 1),
        )

    def unload(self):
        import torch

        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_metadata(self):
        metadata = super().get_metadata()
        tasks = []
        for task, name in self.TASKS.items():
            widgets = {"edit_text": {}, "button_send": {}}
            mode = "text_prompt"
            if task == "ocr":
                widgets = {"button_run": {}}
                mode = "default"
            elif task in ("visual_prompt", "box_matting"):
                widgets = {
                    "add_pos_rect": {},
                    "button_run_rect": {},
                    "button_clear": {},
                    "button_finish_object": {},
                }
                mode = None
            if task in ("segmentation", "matting", "box_matting"):
                widgets.update(
                    mask_fineness_slider={}, mask_fineness_value_label={}
                )
            widgets["toggle_preserve_existing_annotations"] = {}
            tasks.append(
                dict(
                    id=task,
                    name=name,
                    description=name,
                    batch_processing_mode=mode,
                    active_widgets=widgets,
                )
            )
        metadata["available_tasks"] = tasks
        return metadata

    @staticmethod
    def _boxes(params, width, height):
        boxes = []
        for mark in params.get("marks", []):
            if mark.get("type") != "rectangle" or mark.get("label") != 1:
                continue
            values = mark.get("data", [])
            if len(values) != 4 or not np.isfinite(values).all():
                raise ValueError(
                    "Reference boxes require four finite coordinates"
                )
            x1, x2 = sorted(np.clip(values[::2], 0, width))
            y1, y2 = sorted(np.clip(values[1::2], 0, height))
            if x2 <= x1 or y2 <= y1:
                raise ValueError("Reference box must have positive area")
            boxes.append([x1, y1, x2, y2])
        return boxes

    @staticmethod
    def _png(array):
        buffer = BytesIO()
        Image.fromarray(array).save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def predict(self, image, params):
        from wevisone_vl.engine import InferenceResult

        task = params.get("current_task", self.params.get("task", "detection"))
        if task not in self.TASKS:
            raise ValueError(f"Unsupported task: {task}")
        height, width = image.shape[:2]
        text = params.get("text_prompt", "").strip()
        visual = task in ("visual_prompt", "box_matting")
        boxes = self._boxes(params, width, height) if visual else []
        if (visual and not boxes) or (
            not visual and task != "ocr" and not text
        ):
            return {"shapes": [], "description": ""}
        rgb = Image.fromarray(image[:, :, ::-1])
        if task == "box_matting":
            prompt = (
                "\nLocate a single instance that best matches the following "
                "description and provide the alpha mattes: object"
            )
            normalized = np.asarray(boxes) / [width, height, width, height]
            response = (
                "<ref>object</ref><box>"
                + ",".join(
                    "".join(f"<{round(v * 1000)}>" for v in box) + "<OBJ>"
                    for box in normalized
                )
                + "</box>"
            )
            decoded_boxes, masks = self.model.decode_masks(
                rgb, prompt, response, boxes=normalized
            )
            result = InferenceResult(response, prompt, decoded_boxes, masks)
        else:
            query = (
                parse_prompts(text, separators=[",", ".", "\n"])
                if task == "detection"
                else text
            )
            generation = {
                key: params.get(key, self.params.get(key, default))
                for key, default in (
                    ("max_new_tokens", 4096),
                    ("temperature", 0.0),
                    ("top_p", 0.95),
                    ("top_k", 50),
                    ("repetition_penalty", 1.05),
                )
            }
            result = self.model.predict(
                rgb,
                task=task,
                query=query,
                reference_boxes=boxes,
                **generation,
            )
        predictions = self.parse_predictions(result.response, width, height)
        shapes = []
        for label, annotations in predictions.items():
            for annotation in annotations:
                coords = annotation["coords"]
                if result.masks:
                    continue
                if annotation["type"] == "point":
                    points = [coords]
                    shape_type = "point"
                elif annotation["type"] == "box":
                    x1, y1, x2, y2 = coords
                    points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    shape_type = "rectangle"
                else:
                    continue
                shapes.append(
                    Shape(
                        label="AUTOLABEL_OBJECT" if visual else label,
                        shape_type=shape_type,
                        points=points,
                    ).model_dump()
                )
        artifacts = []
        if result.masks:
            threshold = params.get(
                "mask_threshold", self.params.get("mask_threshold", 0.5)
            )
            epsilon = params.get(
                "epsilon_factor", self.params.get("epsilon_factor", 0.001)
            )
            if not 0 < threshold < 1 or not 0 <= epsilon <= 1:
                raise ValueError("Invalid mask threshold or contour fineness")
            labels = [
                label
                for label, items in predictions.items()
                for item in items
                if item["type"] == "box"
            ]
            for index, mask in enumerate(result.masks):
                alpha = np.asarray(mask, dtype=np.float32)
                if (
                    alpha.shape != (height, width)
                    or not np.isfinite(alpha).all()
                ):
                    raise ValueError("Invalid decoded alpha matte")
                binary = np.uint8(alpha >= threshold)
                contours, _ = cv2.findContours(
                    binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                label = (
                    "AUTOLABEL_OBJECT"
                    if visual
                    else (labels[index] if index < len(labels) else text)
                )
                for contour in contours:
                    points = cv2.approxPolyDP(
                        contour, epsilon * cv2.arcLength(contour, True), True
                    ).reshape(-1, 2)
                    if len(points) >= 3:
                        shapes.append(
                            Shape(
                                label=label,
                                shape_type="polygon",
                                points=points.tolist(),
                                group_id=index + 1,
                            ).model_dump()
                        )
                pixels = (
                    np.uint8(np.clip(alpha, 0, 1) * 255)
                    if task != "segmentation"
                    else binary * 255
                )
                artifacts.append(
                    {
                        "kind": "alpha" if task != "segmentation" else "mask",
                        "index": index,
                        "png_base64": self._png(pixels),
                    }
                )
                if task != "segmentation":
                    artifacts.append(
                        {
                            "kind": "cutout",
                            "index": index,
                            "png_base64": self._png(
                                np.dstack((np.asarray(rgb), pixels))
                            ),
                        }
                    )
        description = ""
        if task in ("count_box", "count_point"):
            description = f"{text}: {len(shapes)}"
        elif task == "freeform":
            description = re.sub(
                r"<(box|point|polygon)>.*?</\1>",
                "",
                result.response,
                flags=re.DOTALL | re.IGNORECASE,
            )
            description = re.sub(
                r"</?ref>|<OBJ>|<\d{1,4}>|<\|[^>]*\|>",
                "",
                description,
                flags=re.IGNORECASE,
            ).strip()
        return {
            "shapes": shapes,
            "description": description,
            "artifacts": artifacts,
        }
