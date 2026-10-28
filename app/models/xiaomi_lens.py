import re

import cv2
from PIL import Image

from app.models import BaseModel
from app.core.registry import register_model
from app.schemas.shape import Shape


@register_model("xiaomi_lens")
class XiaomiLens(BaseModel):
    TASKS = {
        "referring_comprehension": "Referring Expression Comprehension",
        "referring_segmentation": "Referring Expression Segmentation",
        "multi_instance_grounding": "Multi-Instance Grounding",
        "reasoning_segmentation": "Reasoning Segmentation",
        "embodied_referring": "Embodied Referring",
        "fine_grained_perception": "Fine-Grained Perception",
    }

    def load(self):
        import torch
        from infer import load_model_and_processor

        self.device = torch.device(self.params.get("device", "cuda"))
        self.model, self.processor, self.mask_token_id = (
            load_model_and_processor(
                self.params.get("model_path", "SeerRay-Lab/Xiaomi-Lens-0"),
                self.device,
                self.params.get("attn_implementation", "sdpa"),
            )
        )
        self.processor.image_processor.size["longest_edge"] = self.params.get(
            "max_pixels", 262144
        )

    def unload(self):
        import torch

        del self.model
        del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_metadata(self):
        metadata = super().get_metadata()
        metadata["available_tasks"] = [
            {
                "id": task,
                "name": name,
                "description": name,
                "batch_processing_mode": "text_prompt",
                "active_widgets": {
                    "edit_text": {},
                    "button_send": {},
                    "button_finish_object": {},
                    "toggle_preserve_existing_annotations": {},
                    **(
                        {
                            "mask_fineness_slider": {},
                            "mask_fineness_value_label": {},
                        }
                        if task != "referring_comprehension"
                        else {}
                    ),
                },
            }
            for task, name in self.TASKS.items()
        ]
        return metadata

    def _build_prompt(self, task, text):
        if task not in self.TASKS:
            raise ValueError(f"Unsupported task: {task}")
        if task in (
            "referring_comprehension",
            "referring_segmentation",
            "multi_instance_grounding",
        ):
            return (
                'Locate all instances in the image that match the description '
                f'"{text}". Report bbox coordinates and masks.'
            )
        if task == "fine_grained_perception":
            return text
        return f"{text}\nReport bbox coordinates and masks."

    def predict(self, image, params):
        import torch
        from infer import (
            build_inputs,
            generate_completion,
            parse_answer_bboxes,
            run_forward,
            split_answer_content,
        )

        task = params.get(
            "current_task", self.params.get("task", "referring_comprehension")
        )
        text = params.get("text_prompt", "").strip()
        prompt = self._build_prompt(task, text)
        if not text:
            return {"shapes": [], "description": ""}
        inputs, _ = build_inputs(
            self.processor,
            Image.fromarray(image[:, :, ::-1]),
            prompt,
            self.device,
        )
        max_tokens = params.get(
            "max_new_tokens", self.params.get("max_new_tokens", 4096)
        )
        completion, generated, count = generate_completion(
            self.model, self.processor, inputs, max_tokens, self.mask_token_id
        )
        eos = self.model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else eos or []
        if (
            completion.shape[1] >= max_tokens
            and completion[0, -1].item() not in eos
        ):
            raise ValueError(
                "Generation was truncated; increase max_new_tokens"
            )
        if "</think>" not in generated:
            raise ValueError("Model output has no completed reasoning block")
        answer, thought_count = split_answer_content(generated)
        description = re.sub(
            r"<bbox>.*?</bbox>|<mask>.*?</mask>|<\|[^>]+\|>",
            "",
            answer,
            flags=re.DOTALL,
        ).strip()
        height, width = image.shape[:2]
        shapes = []
        if task == "referring_comprehension":
            for group_id, box in enumerate(parse_answer_bboxes(generated), 1):
                x1, y1, x2, y2 = [
                    min(max(value, 0), 1000) / 1000 * size
                    for value, size in zip(box, (width, height, width, height))
                ]
                if x2 <= x1 or y2 <= y1:
                    continue
                shapes.append(
                    Shape(
                        label="AUTOLABEL_OBJECT",
                        shape_type="rectangle",
                        points=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                        group_id=group_id,
                        description=text,
                    ).model_dump()
                )
        elif count > thought_count:
            masks = run_forward(
                self.model, inputs, completion, self.mask_token_id, self.device
            )[0, thought_count:]
            threshold = params.get(
                "mask_threshold", self.params.get("mask_threshold", 0.5)
            )
            if not 0 < threshold < 1:
                raise ValueError("mask_threshold must be between 0 and 1")
            epsilon = params.get(
                "epsilon_factor", self.params.get("epsilon_factor", 0.001)
            )
            if not 0 <= epsilon <= 1:
                raise ValueError("epsilon_factor must be between 0 and 1")
            for group_id, mask in enumerate(masks, 1):
                binary = (
                    (mask.sigmoid() >= threshold).to(torch.uint8).cpu().numpy()
                )
                binary = cv2.resize(
                    binary, (width, height), interpolation=cv2.INTER_NEAREST
                )
                contours, _ = cv2.findContours(
                    binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                for contour in contours:
                    points = cv2.approxPolyDP(
                        contour, epsilon * cv2.arcLength(contour, True), True
                    ).reshape(-1, 2)
                    if len(points) < 3:
                        continue
                    shapes.append(
                        Shape(
                            label="AUTOLABEL_OBJECT",
                            shape_type="polygon",
                            points=points.tolist(),
                            group_id=group_id,
                            description=text,
                        ).model_dump()
                    )
        return {"shapes": shapes, "description": description}
