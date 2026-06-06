import re
import numpy as np
from loguru import logger
from PIL import Image
from typing import Any, Dict, List, Optional

from . import BaseModel, parse_prompts
from app.schemas.shape import Shape
from app.core.registry import register_model


@register_model("locateanything")
class LocateAnything(BaseModel):
    """LocateAnything vision-language grounding model."""

    def load(self):
        """Load LocateAnything model and processor."""
        import torch
        from transformers import (
            AutoModel,
            AutoProcessor,
            AutoTokenizer,
        )

        model_path = self.params.get("model_path", "nvidia/LocateAnything-3B")
        self.device = self.params.get("device", "cuda")
        dtype_name = self.params.get("dtype", "bfloat16")
        self.dtype = getattr(torch, dtype_name)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device).eval()

        logger.info(f"LocateAnything model loaded from {model_path}")

    def get_metadata(self) -> Dict[str, Any]:
        """Return model metadata including available tasks."""
        metadata = super().get_metadata()
        metadata["available_tasks"] = [
            {
                "id": "detection",
                "name": "Detection",
                "description": "Detect categories and return bounding boxes",
                "batch_processing_mode": "text_prompt",
                "active_widgets": {
                    "edit_text": {},
                    "button_send": {},
                    "toggle_preserve_existing_annotations": {},
                },
            },
            {
                "id": "grounding",
                "name": "Grounding",
                "description": "Ground a phrase and return bounding boxes",
                "batch_processing_mode": "text_prompt",
                "active_widgets": {
                    "edit_text": {},
                    "button_send": {},
                    "toggle_preserve_existing_annotations": {},
                },
            },
            {
                "id": "pointing",
                "name": "Pointing",
                "description": "Point to objects based on description",
                "batch_processing_mode": "text_prompt",
                "active_widgets": {
                    "edit_text": {},
                    "button_send": {},
                    "toggle_preserve_existing_annotations": {},
                },
            },
            {
                "id": "text_detection",
                "name": "Text Detection",
                "description": "Detect scene text boxes",
                "batch_processing_mode": "default",
                "active_widgets": {
                    "button_run": {},
                    "toggle_preserve_existing_annotations": {},
                },
            },
        ]
        return metadata

    def predict(
        self, image: np.ndarray, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute LocateAnything prediction."""
        task = params.get("current_task", self.params.get("task", "detection"))
        pil_image = Image.fromarray(image[:, :, ::-1])
        inference_image = self._resize_image(pil_image, params)

        if task == "detection":
            text_prompt = params.get("text_prompt", "")
            if not text_prompt:
                logger.warning("Please provide categories for detection task.")
                return {"shapes": [], "description": ""}
            categories = parse_prompts(text_prompt, separators=[",", ".", "\n"])
            prompt = self._build_detection_prompt(categories)
        elif task == "grounding":
            text_prompt = params.get("text_prompt", "")
            if not text_prompt:
                logger.warning("Please provide a prompt for grounding task.")
                return {"shapes": [], "description": ""}
            prompt = (
                "Locate all the instances that match the following description: "
                f"{text_prompt}."
            )
        elif task == "pointing":
            text_prompt = params.get("text_prompt", "")
            if not text_prompt:
                logger.warning("Please provide a prompt for pointing task.")
                return {"shapes": [], "description": ""}
            prompt = f"Point to: {text_prompt}."
        elif task == "text_detection":
            prompt = "Detect all the text in box format."
        else:
            raise ValueError(f"Unsupported task: {task}")

        answer = self._generate(inference_image, prompt, params)
        shapes = self._parse_answer(
            answer,
            pil_image.width,
            pil_image.height,
            text_as_description=task == "text_detection",
        )

        return {"shapes": shapes, "description": ""}

    def _resize_image(
        self, image: Image.Image, params: Dict[str, Any]
    ) -> Image.Image:
        """Resize image before inference when max_image_size is configured."""
        max_image_size = params.get(
            "max_image_size", self.params.get("max_image_size", 1024)
        )
        if not max_image_size:
            return image

        resized = image.copy()
        resized.thumbnail((max_image_size, max_image_size))
        return resized

    @staticmethod
    def _build_detection_prompt(categories: List[str]) -> str:
        """Build a LocateAnything detection prompt."""
        cats = "</c>".join(categories)
        return (
            "Locate all the instances that matches the following description: "
            f"{cats}."
        )

    def _generate(
        self,
        image: Image.Image,
        prompt: str,
        params: Dict[str, Any],
    ) -> str:
        """Run LocateAnything generation."""
        import torch

        max_new_tokens = params.get(
            "max_new_tokens", self.params.get("max_new_tokens", 2048)
        )
        generation_mode = params.get(
            "generation_mode", self.params.get("generation_mode", "hybrid")
        )
        temperature = params.get(
            "temperature", self.params.get("temperature", 0.7)
        )
        verbose = params.get("verbose", self.params.get("verbose", True))

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=images, videos=videos, return_tensors="pt"
        ).to(self.device)

        pixel_values = inputs["pixel_values"].to(self.dtype)
        input_ids = inputs["input_ids"]
        image_grid_hws = inputs.get("image_grid_hws", None)

        with torch.no_grad():
            response = self.model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=inputs["attention_mask"],
                image_grid_hws=image_grid_hws,
                tokenizer=self.tokenizer,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                generation_mode=generation_mode,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                verbose=verbose,
            )

        return response[0] if isinstance(response, tuple) else response

    def _parse_answer(
        self,
        answer: str,
        image_width: int,
        image_height: int,
        text_as_description: bool = False,
    ) -> List[Shape]:
        """Parse LocateAnything output into X-AnyLabeling shapes."""
        shapes = []
        current_label = "" if text_as_description else "object"
        pattern = re.compile(
            r"<ref>(.*?)</ref>|"
            r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>|"
            r"<box><(\d+)><(\d+)></box>|"
            r"<box>none</box>"
        )

        for match in pattern.finditer(answer):
            groups = match.groups()
            if groups[0] is not None:
                current_label = groups[0] or "object"
                continue

            if groups[1] is not None:
                shape = self._box_to_shape(
                    "object" if text_as_description else current_label,
                    [int(value) for value in groups[1:5]],
                    image_width,
                    image_height,
                    (
                        current_label
                        if text_as_description and current_label
                        else None
                    ),
                )
                shapes.append(shape)
            elif groups[5] is not None:
                x = int(groups[5]) / 1000 * image_width
                y = int(groups[6]) / 1000 * image_height
                shapes.append(
                    Shape(
                        label="object" if text_as_description else current_label,
                        shape_type="point",
                        points=[[float(x), float(y)]],
                        description=(
                            current_label
                            if text_as_description and current_label
                            else None
                        ),
                    )
                )

        return shapes

    @staticmethod
    def _box_to_shape(
        label: str,
        box: List[int],
        image_width: int,
        image_height: int,
        description: Optional[str] = None,
    ) -> Shape:
        """Convert a normalized LocateAnything box to a rectangle shape."""
        x1, y1, x2, y2 = box
        abs_x1 = x1 / 1000 * image_width
        abs_y1 = y1 / 1000 * image_height
        abs_x2 = x2 / 1000 * image_width
        abs_y2 = y2 / 1000 * image_height

        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1
        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1

        return Shape(
            label=label,
            shape_type="rectangle",
            points=[
                [float(abs_x1), float(abs_y1)],
                [float(abs_x2), float(abs_y1)],
                [float(abs_x2), float(abs_y2)],
                [float(abs_x1), float(abs_y2)],
            ],
            description=description,
        )

    def unload(self):
        """Release model resources."""
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "processor"):
            del self.processor
        if hasattr(self, "tokenizer"):
            del self.tokenizer
