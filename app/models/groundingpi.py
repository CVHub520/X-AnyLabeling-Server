import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Dict

import numpy as np
import requests
from PIL import Image

from . import BaseModel, parse_prompts
from app.core.registry import register_model
from app.schemas.shape import Shape


@register_model("groundingpi")
class GroundingPI(BaseModel):
    """GroundingPI with Transformers and OpenAI-compatible backends."""

    def load(self):
        from .groundingpi_impl import protocol
        from .groundingpi_impl.client import parse_response

        self.protocol = protocol
        self.parse_response = parse_response
        self.backend = self.params.get("backend", "transformers")
        if self.backend == "transformers":
            self._load_transformers()
        elif self.backend != "openai":
            raise ValueError(f"Unsupported backend: {self.backend}")
        self.api_base = self.params.get(
            "api_base", "http://127.0.0.1:8001/v1"
        ).rstrip("/")

    def _load_transformers(self):
        import torch
        import transformers
        from huggingface_hub import snapshot_download

        if transformers.__version__ != "5.7.0":
            raise RuntimeError(
                "GroundingPI requires the upstream bundled Transformers "
                "5.7.0 fork. Install its prepared vendor/transformers source "
                "in the environment running X-AnyLabeling-Server."
            )
        model_path = self.params.get("model_path", "GroundingPI/GroundingPI")
        if not Path(model_path).is_dir():
            model_path = snapshot_download(repo_id=model_path)
        self.device = self.params.get("device", "cuda")
        self.processor = transformers.AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = (
            transformers.AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                dtype=getattr(torch, self.params.get("dtype", "bfloat16")),
                attn_implementation=self.params.get(
                    "attn_implementation", "sdpa"
                ),
            )
            .eval()
            .to(self.device)
        )

    def unload(self):
        if self.backend == "transformers":
            import torch

            del self.model
            del self.processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_metadata(self) -> Dict[str, Any]:
        metadata = super().get_metadata()
        metadata["available_tasks"] = []
        for task, name, description in (
            ("detection", "Detection", "Detect categories as bounding boxes"),
            ("grounding", "Grounding", "Locate a described target"),
            ("pointing", "Pointing", "Point to a described target"),
            ("visual_prompting", "Visual Prompting", "Find similar objects"),
        ):
            visual = task == "visual_prompting"
            widgets = (
                {
                    "add_pos_rect": {},
                    "button_run_rect": {},
                    "button_clear": {},
                    "button_finish_object": {},
                }
                if visual
                else {"edit_text": {}, "button_send": {}}
            )
            widgets["toggle_preserve_existing_annotations"] = {}
            metadata["available_tasks"].append(
                {
                    "id": task,
                    "name": name,
                    "description": description,
                    "batch_processing_mode": None if visual else "text_prompt",
                    "active_widgets": widgets,
                }
            )
        return metadata

    def _build_prompt(self, task, params, width, height):
        if task == "visual_prompting":
            boxes = []
            for mark in params.get("marks", []):
                if mark.get("type") != "rectangle" or mark.get("label") != 1:
                    continue
                data = mark.get("data", [])
                if len(data) != 4 or not np.isfinite(data).all():
                    raise ValueError(
                        "Reference boxes need four finite coordinates"
                    )
                x1, x2 = sorted((data[0], data[2]))
                y1, y2 = sorted((data[1], data[3]))
                box = tuple(
                    round(min(max(value / max(size - 1, 1), 0), 1) * 999)
                    for value, size in zip(
                        (x1, y1, x2, y2), (width, height, width, height)
                    )
                )
                if box[0] >= box[2] or box[1] >= box[3]:
                    raise ValueError(
                        "Reference box is empty after quantization"
                    )
                if box not in boxes:
                    boxes.append(box)
            return self.protocol.build_visual_prompt(boxes) if boxes else None

        builders = {
            "detection": self.protocol.build_dense_bbox_prompt,
            "grounding": self.protocol.build_refer_bbox_prompt,
            "pointing": self.protocol.build_refer_point_prompt,
        }
        if task not in builders:
            raise ValueError(f"Unsupported task: {task}")
        text = params.get("text_prompt", "").strip()
        if not text:
            return None
        if task == "detection":
            categories = parse_prompts(text, separators=[",", ".", "\n"])
            return builders[task](categories) if categories else None
        return builders[task](text)

    def predict(
        self, image: np.ndarray, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        task = params.get("current_task", self.params.get("task", "detection"))
        height, width = image.shape[:2]
        prompt = self._build_prompt(task, params, width, height)
        if prompt is None:
            return {"shapes": [], "description": ""}

        if self.backend == "transformers":
            response = self._generate(image, prompt, params)
        else:
            response = self._request(image, prompt, params)
        geometry = "point" if task == "pointing" else "bbox"
        result = self.parse_response(response, geometry)
        if not result.valid:
            raise ValueError(
                f"Invalid GroundingPI output: {result.parse_error}"
            )

        shapes = []
        for prediction in result.predictions:
            for coordinates in prediction.coordinates or ():
                pixels = [
                    value / 999 * (width - 1 if i % 2 == 0 else height - 1)
                    for i, value in enumerate(coordinates)
                ]
                shapes.append(
                    Shape(
                        label=(
                            "AUTOLABEL_OBJECT"
                            if task == "visual_prompting"
                            else prediction.label
                        ),
                        shape_type=(
                            "point" if geometry == "point" else "rectangle"
                        ),
                        points=(
                            [
                                pixels[:2],
                                [pixels[2], pixels[1]],
                                pixels[2:],
                                [pixels[0], pixels[3]],
                            ]
                            if geometry == "bbox"
                            else [pixels]
                        ),
                    ).model_dump()
                )
        return {"shapes": shapes, "description": ""}

    def _generate(self, image, prompt, params):
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        image_inputs = self.processor.image_processor(
            images=[Image.fromarray(image[:, :, ::-1])], return_tensors="pt"
        )
        merge = self.processor.image_processor.merge_size**2
        sizes = [int(grid.prod()) for grid in image_inputs["image_grid_thw"]]
        parts = text.split("<|image_pad|>")
        if len(parts) != len(sizes) + 1 or any(
            n < merge or n % merge for n in sizes
        ):
            raise ValueError(
                "Image placeholders do not match processed images"
            )
        text = parts[0] + "".join(
            "<|image_pad|>" * (size // merge) + tail
            for size, tail in zip(sizes, parts[1:])
        )
        inputs = dict(self.processor.tokenizer(text, return_tensors="pt"))
        inputs.update(image_inputs)
        input_length = inputs["input_ids"].shape[1]
        max_tokens = params.get(
            "max_new_tokens", self.params.get("max_new_tokens", 4096)
        )
        if input_length + max_tokens > self.params.get("max_model_len", 16384):
            raise ValueError(
                "Prompt plus max_new_tokens exceeds max_model_len"
            )
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                repetition_penalty=1.0,
            )[0, input_length:].tolist()
        eos = self.model.generation_config.eos_token_id
        eos = {eos} if isinstance(eos, int) else set(eos or [])
        stopped = bool(generated and generated[-1] in eos)
        trimmed = list(generated)
        while trimmed and trimmed[-1] in eos:
            trimmed.pop()
        content = self.processor.tokenizer.decode(
            trimmed, skip_special_tokens=False
        )
        return {
            "choices": [
                {
                    "message": {"content": content},
                    "finish_reason": (
                        "stop"
                        if stopped or len(generated) < max_tokens
                        else "length"
                    ),
                }
            ]
        }

    def _request(self, image, prompt, params):

        buffered = BytesIO()
        Image.fromarray(image[:, :, ::-1]).save(buffered, format="PNG")
        encoded = base64.b64encode(buffered.getvalue()).decode("ascii")
        payload = {
            "model": self.params.get("model_name", "groundingpi"),
            "temperature": 0,
            "max_tokens": params.get(
                "max_new_tokens", self.params.get("max_new_tokens", 4096)
            ),
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{encoded}"
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        headers = {"Content-Type": "application/json"}
        api_key = self.params.get("api_key")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        response = requests.post(
            f"{self.api_base}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.params.get("timeout", 120),
        )
        response.raise_for_status()
        return response.json()
