import ast
import numpy as np
from loguru import logger
from typing import Any, Dict
from PIL import Image

from .base import BaseModel
from app.schemas.shape import Shape


class Qwen3VL(BaseModel):
    """Unified Qwen3-VL model supporting multiple tasks and backends."""

    def load(self):
        """Load Qwen3VL model based on backend configuration."""
        self.backend = self.params.get("backend", "transformers")
        self.default_task = self.params.get("task", "caption")

        if self.backend == "transformers":
            self._load_transformers()
        elif self.backend == "openai":
            self._load_openai()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _load_transformers(self):
        """Load model using Transformers library."""
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

        model_path = self.params.get("model_path", "Qwen/Qwen3-VL-2B-Instruct")
        device = self.params.get("device", "auto")

        logger.info(
            f"Loading Qwen3VL with Transformers backend from {model_path}"
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, dtype="auto", device_map=device
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        logger.info("Qwen3VL model loaded successfully")

    def _load_openai(self):
        """Initialize OpenAI API client."""
        self.api_base = self.params.get("api_base", "http://localhost:8000/v1")
        self.api_key = self.params.get("api_key", "EMPTY")
        self.model_name = self.params.get(
            "model_name", "Qwen/Qwen3-VL-2B-Instruct"
        )
        logger.info(f"Using OpenAI API backend at {self.api_base}")

    def predict(
        self, image: np.ndarray, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute prediction based on task type.

        Args:
            image: Input image in BGR format.
            params: Inference parameters including task type.

        Returns:
            Dictionary with prediction results.
        """
        task = self.default_task

        if task == "caption":
            return self._predict_caption(image, params)
        elif task == "grounding":
            return self._predict_grounding(image, params)
        else:
            raise ValueError(f"Unsupported task: {task}")

    def _predict_caption(
        self, image: np.ndarray, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute image captioning task."""
        prompt_mode = params.get("prompt_mode", "detailed")

        if prompt_mode == "detailed":
            prompt = (
                "Describe this image in detail. Include information about: "
                "the main subjects and objects, their positions and relationships, "
                "colors, lighting, background, activities or actions taking place, "
                "and the overall scene or context."
            )
        elif prompt_mode == "brief":
            prompt = "Provide a brief, concise description of this image in one or two sentences."
        else:
            prompt = params.get(
                "custom_prompt", "Describe what you see in this image."
            )

        if self.backend == "transformers":
            description = self._inference_transformers(image, prompt, params)
        else:
            description = self._inference_openai(image, prompt, params)

        logger.info(f"Generated caption: {description[:200]}...")

        return {"shapes": [], "description": description}

    def _predict_grounding(
        self, image: np.ndarray, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute 2D object grounding task."""
        text_prompt = params.get("text_prompt", "")
        if not text_prompt:
            logger.warning("Please provide a text prompt for grounding task.")
            return {"shapes": [], "description": ""}

        categories_list = [
            cat.strip() for cat in text_prompt.split(".") if cat.strip()
        ]
        categories = ", ".join(categories_list)

        prompt = (
            f'locate every instance that belongs to the following categories: "{categories}". '
            "Report bbox coordinates in JSON format."
        )

        if self.backend == "transformers":
            response = self._inference_transformers(image, prompt, params)
        else:
            response = self._inference_openai(image, prompt, params)

        shapes = self._parse_grounding_response(response, image.shape)

        return {"shapes": shapes, "description": ""}

    def _inference_transformers(
        self, image: np.ndarray, prompt: str, params: Dict[str, Any]
    ) -> str:
        """Run inference using Transformers backend."""
        max_new_tokens = params.get(
            "max_new_tokens", self.params.get("max_new_tokens", 512)
        )
        repetition_penalty = params.get(
            "repetition_penalty", self.params.get("repetition_penalty", 1.1)
        )

        pil_image = Image.fromarray(image[:, :, ::-1])

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_text[0].strip()

    def _inference_openai(
        self, image: np.ndarray, prompt: str, params: Dict[str, Any]
    ) -> str:
        """Run inference using OpenAI API backend."""
        import base64
        import requests
        from io import BytesIO

        max_tokens = params.get(
            "max_new_tokens", self.params.get("max_new_tokens", 512)
        )
        temperature = params.get(
            "temperature", self.params.get("temperature", 0.7)
        )

        pil_image = Image.fromarray(image[:, :, ::-1])
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        img_data_uri = f"data:image/png;base64,{img_base64}"

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": img_data_uri},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        timeout = params.get("timeout", self.params.get("timeout", 60))

        try:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                json=payload,
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"OpenAI API inference error: {e}")
            raise

    @staticmethod
    def _parse_json(json_output: str) -> str:
        """Parse JSON output from model, removing markdown fencing."""
        lines = json_output.splitlines()
        for i, line in enumerate(lines):
            if line == "```json":
                json_output = "\n".join(lines[i + 1 :])
                json_output = json_output.split("```")[0]
                break
        return json_output

    def _parse_grounding_response(
        self, response: str, image_shape: tuple
    ) -> list:
        """Parse grounding response and convert to shapes."""
        shapes = []
        height, width = image_shape[:2]

        try:
            json_output = self._parse_json(response)
            bounding_boxes = ast.literal_eval(json_output)
        except Exception:
            try:
                end_idx = response.rfind('"}') + len('"}')
                truncated_text = response[:end_idx] + "]"
                json_output = self._parse_json(truncated_text)
                bounding_boxes = ast.literal_eval(json_output)
            except Exception:
                return shapes

        if not isinstance(bounding_boxes, list):
            bounding_boxes = [bounding_boxes]

        for bbox_data in bounding_boxes:
            if "bbox_2d" not in bbox_data:
                continue

            bbox = bbox_data["bbox_2d"]
            label = bbox_data.get("label", "object")

            abs_x1 = int(bbox[0] / 1000 * width)
            abs_y1 = int(bbox[1] / 1000 * height)
            abs_x2 = int(bbox[2] / 1000 * width)
            abs_y2 = int(bbox[3] / 1000 * height)

            if abs_x1 > abs_x2:
                abs_x1, abs_x2 = abs_x2, abs_x1
            if abs_y1 > abs_y2:
                abs_y1, abs_y2 = abs_y2, abs_y1

            shape = Shape(
                label=label,
                shape_type="rectangle",
                points=[
                    [float(abs_x1), float(abs_y1)],
                    [float(abs_x2), float(abs_y1)],
                    [float(abs_x2), float(abs_y2)],
                    [float(abs_x1), float(abs_y2)],
                ],
            )
            shapes.append(shape)

        return shapes

    def unload(self):
        """Release model resources."""
        if self.backend == "transformers":
            if hasattr(self, "model"):
                del self.model
            if hasattr(self, "processor"):
                del self.processor
