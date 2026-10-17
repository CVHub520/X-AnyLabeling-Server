"""Small HTTP client for an already running grounding model service."""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
import json
import math
import mimetypes
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .protocol import (
    build_refer_bbox_prompt,
    build_refer_point_prompt,
    parse_answer,
)


@dataclass(frozen=True)
class Prediction:
    label: str
    coordinates: tuple[tuple[int, ...], ...] | None


@dataclass(frozen=True)
class Result:
    task: str
    raw_output: str
    predictions: tuple[Prediction, ...]
    finish_reason: str | None
    usage: dict | None
    parse_error: str | None

    @property
    def valid(self) -> bool:
        return self.parse_error is None and self.finish_reason == "stop"

    def to_dict(self) -> dict:
        return {**asdict(self), "valid": self.valid}


def parse_response(response: dict, task: str) -> Result:
    if task not in {"bbox", "point"}:
        raise ValueError("task must be bbox or point")
    choice = response["choices"][0]
    raw = choice["message"]["content"]
    if not isinstance(raw, str):
        raise ValueError("service returned non-text content")
    finish = choice.get("finish_reason")
    predictions, error = (), None
    try:
        entries = parse_answer(raw, task=task, require_canonical=True)
        predictions = tuple(
            Prediction(e.phrase, e.coordinates) for e in entries
        )
    except ValueError as exc:
        error = str(exc)
    # Partial output must never become a successful prediction, even if its
    # final complete prefix happens to satisfy the grammar.
    if finish != "stop":
        error = error or f"incomplete generation: finish_reason={finish!r}"
    if error:
        predictions = ()
    return Result(task, raw, predictions, finish, response.get("usage"), error)


class GroundingPi:
    """Connect to the native service; construction does not load model weights."""

    def __init__(
        self,
        base_url="http://127.0.0.1:8000/v1",
        model="groundingpi",
        *,
        api_key=None,
        timeout=120,
    ):
        parsed = urlsplit(base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "base_url must be an HTTP(S) service URL without credentials, query or fragment"
            )
        if (
            not isinstance(timeout, (float, int))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("timeout must be finite and positive")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a nonempty service model ID")
        self.base_url, self.model = base_url.rstrip("/"), model
        self.api_key, self.timeout = api_key, timeout

    def predict(
        self, image: str | Path, phrase: str, *, task="bbox", max_tokens=4096
    ) -> Result:
        if task not in {"bbox", "point"}:
            raise ValueError("task must be bbox or point")
        prompt = (
            build_refer_bbox_prompt
            if task == "bbox"
            else build_refer_point_prompt
        )(phrase)
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        path = Path(image)
        mime = mimetypes.guess_type(path.name)[0]
        if mime not in {"image/jpeg", "image/png", "image/webp"}:
            raise ValueError("image must have a JPEG, PNG or WebP extension")
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": max_tokens,
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{encoded}"
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        request = Request(
            self.base_url + "/chat/completions",
            data=json.dumps(body).encode(),
            headers=headers,
        )
        with urlopen(request, timeout=self.timeout) as response:
            return parse_response(json.load(response), task)


def visualize(image: str | Path, result: Result):
    """Return a Pillow RGB image. Coordinates use the canonical 0..999 grid."""
    if not result.valid:
        raise ValueError(
            "cannot visualize an invalid or incomplete result: "
            + str(result.parse_error)
        )
    from PIL import Image, ImageDraw

    with Image.open(image) as source:
        canvas = source.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for prediction in result.predictions:
        for coords in prediction.coordinates or ():
            pixels = [
                v
                / 999
                * (canvas.width - 1 if i % 2 == 0 else canvas.height - 1)
                for i, v in enumerate(coords)
            ]
            if result.task == "bbox":
                draw.rectangle(pixels, outline="red", width=3)
            else:
                x, y = pixels
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="red")
    return canvas
