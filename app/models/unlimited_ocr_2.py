import base64
import json
import os
from io import BytesIO
from threading import Lock

import numpy as np
import requests
from PIL import Image

from . import BaseModel
from .unlimited_ocr_2_impl.postprocess import (
    PAGE_SPLIT_RE,
    STRAY_TOKEN_RE,
    _base_type,
    _build_groups,
    _parse_blocks,
    convert,
)
from app.core.registry import register_model


@register_model("unlimited_ocr_2")
class UnlimitedOCR2(BaseModel):
    def load(self):
        self.backend = self.params.get("backend", "transformers")
        self._lock = Lock()
        if self.backend == "sglang":
            self.api_url = self.params.get(
                "api_url", "http://127.0.0.1:10000/v1"
            ).rstrip("/")
            key = self.params.get("api_key") or os.getenv("SGLANG_API_KEY", "")
            self.headers = {"Authorization": f"Bearer {key}"} if key else {}
            response = requests.get(
                f"{self.api_url}/models", headers=self.headers, timeout=10
            )
            response.raise_for_status()
            self.served_model = self.params.get(
                "served_model", "Unlimited-OCR-2"
            )
            if not any(
                model["id"] == self.served_model
                for model in response.json().get("data", [])
            ):
                raise ValueError(
                    f"SGLang model not served: {self.served_model}"
                )
            return
        if self.backend != "transformers":
            raise ValueError(f"Unsupported backend: {self.backend}")
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = torch.device(self.params.get("device", "cuda"))
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("Unlimited-OCR-2 requires a CUDA device")
        model_path = self.params.get("model_path", "baidu/Unlimited-OCR-2")
        options = {
            "trust_remote_code": True,
            "revision": self.params.get("revision"),
        }
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, **options)
        self.model = (
            AutoModel.from_pretrained(
                model_path,
                use_safetensors=True,
                torch_dtype=torch.bfloat16,
                **options,
            )
            .eval()
            .to(self.device)
        )

    def unload(self):
        if self.backend == "sglang":
            return
        import torch

        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict(self, image, params):
        return self.predict_document([image], params)

    def predict_document(self, images, params):
        if not images:
            raise ValueError("Document must contain at least one page")
        tokens_per_page = int(
            params.get(
                "max_new_tokens", self.params.get("max_new_tokens", 8192)
            )
        )
        if tokens_per_page <= 0:
            raise ValueError("max_new_tokens must be positive")
        crop_mode = params.get("crop_mode", self.params.get("crop_mode"))
        if crop_mode is None:
            crop_mode = len(images) == 1
        if not isinstance(crop_mode, bool):
            raise ValueError("crop_mode must be a boolean or null")
        pages = [
            Image.fromarray(image[:, :, ::-1]).convert("RGB")
            for image in images
        ]
        max_tokens = tokens_per_page * len(pages)
        if self.backend == "sglang":
            text = self._infer_sglang(pages, max_tokens, crop_mode)
        else:
            import torch

            with (
                self._lock,
                torch.cuda.device(self.device),
                torch.inference_mode(),
            ):
                result = self.model.infer(
                    self.tokenizer,
                    image_file=pages,
                    crop_mode=crop_mode,
                    max_new_tokens=max_tokens,
                    max_length=self.params.get("max_model_len", 131072),
                    max_page_tokens=self.params.get("max_page_tokens", 20000),
                    no_repeat_ngram_size=35,
                    ngram_window=1024 if len(pages) > 1 else 128,
                    eval_mode=True,
                    save_results=False,
                    return_dict=True,
                )
            if result["finish_reason"] != "stop":
                raise ValueError(
                    "Document parsing was truncated; increase "
                    "max_new_tokens or max_model_len"
                )
            text = result["text"]
        return self._parse_document(text, [page.size for page in pages])

    def _infer_sglang(self, pages, max_tokens, crop_mode):
        content = []
        for page in pages:
            buffer = BytesIO()
            page.save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                }
            )
        content.append(
            {
                "type": "text",
                "text": "<PAGE><image>" * len(pages) + "Multi page parsing.",
            }
        )
        processor = (
            b"csglang.srt.sampling.custom_logit_processor\n"
            b"DeepseekOCRNoRepeatNGramLogitProcessor\n."
        )
        payload = {
            "model": self.served_model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "skip_special_tokens": False,
            "images_config": {"crop_mode": crop_mode},
            "custom_logit_processor": json.dumps(
                {"callable": processor.hex()}
            ),
            "custom_params": {
                "ngram_size": 35,
                "window_size": 1024 if len(pages) > 1 else 128,
                "page_token_id": 128827,
                "eos_token_id": 1,
                "max_page_tokens": self.params.get("max_page_tokens", 20000),
            },
            "stream": True,
        }
        chunks = []
        finish_reason = None
        with requests.post(
            f"{self.api_url}/chat/completions",
            json=payload,
            headers=self.headers,
            timeout=self.params.get("request_timeout", 3600),
            stream=True,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                event = json.loads(data)
                if event.get("error"):
                    raise ValueError(
                        f"SGLang inference failed: {event['error']}"
                    )
                choices = event.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                chunks.append((choice.get("delta") or {}).get("content") or "")
                finish_reason = choice.get("finish_reason") or finish_reason
        if finish_reason != "stop":
            raise ValueError(
                f"SGLang document parsing did not finish: {finish_reason}; "
                "check token limits"
            )
        return "".join(chunks)

    def _parse_document(self, text, page_sizes):
        markers = list(PAGE_SPLIT_RE.finditer(text))
        if (
            len(markers) == len(page_sizes) + 1
            and not text[markers[-1].end() :].strip()
        ):
            text = text[: markers[-1].start()]
            markers.pop()
        if len(markers) != len(page_sizes):
            raise ValueError("Parsed page count does not match input document")
        blocks = _parse_blocks(text)
        groups = _build_groups(blocks)
        group_ids = {
            index: group + 1
            for group, members in enumerate(groups)
            for index in members
        }
        results = []
        for width, height in page_sizes:
            results.append(
                {
                    "prunedResult": {
                        "page_count": len(page_sizes),
                        "width": width,
                        "height": height,
                        "model_settings": {
                            "pipeline_model": self.model_id,
                            "backend": self.backend,
                        },
                        "parsing_res_list": [],
                    },
                    "markdown": {"text": "", "images": {}},
                    "outputImages": {},
                }
            )
        for index, block in enumerate(blocks):
            page_index = block["page"] - 1
            width, height = page_sizes[page_index]
            bbox = block["bbox"] or (0, 0, 1000, 1000)
            x1, y1, x2, y2 = (
                np.clip(bbox, 0, 1000) * [width, height, width, height] / 1000
            )
            if x2 <= x1 or y2 <= y1:
                continue
            label = _base_type(block["label"])
            label = {
                "title": "doc_title",
                "equation": "formula",
                "page_number": "number",
                "__raw__": "text",
                "figure": "image",
                "picture": "image",
            }.get(label, label)
            content = STRAY_TOKEN_RE.sub("", block["content"]).strip()
            page_blocks = results[page_index]["prunedResult"][
                "parsing_res_list"
            ]
            page_blocks.append(
                {
                    "block_label": label,
                    "block_content": content,
                    "block_bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "block_polygon_points": [
                        [x1, y1],
                        [x2, y1],
                        [x2, y2],
                        [x1, y2],
                    ],
                    "block_id": len(page_blocks) + 1,
                    "block_order": len(page_blocks) + 1,
                    "group_id": group_ids[index],
                    "global_block_id": index + 1,
                    "global_group_id": group_ids[index],
                    "source_label": block["label"],
                }
            )
        for result in results:
            result["markdown"]["text"] = "\n\n".join(
                b["block_content"]
                for b in result["prunedResult"]["parsing_res_list"]
                if b["block_content"]
            )
        return {
            "layoutParsingResults": results,
            "dataInfo": {
                "numPages": len(page_sizes),
                "pages": [{"width": w, "height": h} for w, h in page_sizes],
            },
            "markdown": {"text": convert(text), "images": {}},
        }
