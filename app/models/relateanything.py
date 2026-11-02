"""Object detection and open-vocabulary relation prediction."""

from threading import Lock
from typing import Any, Dict
from uuid import uuid4

import numpy as np

from app.core.registry import register_model
from app.models import BaseModel, parse_prompts
from app.schemas.shape import Shape


@register_model("relateanything")
class RelateAnythingModel(BaseModel):
    def load(self):
        from relsgg import RelateAnything
        from ultralytics import YOLO

        device = self.params.get("device", "cpu")
        self.detector = YOLO(self.params.get("detector_path", "yolo11n.pt"))
        self.detector.to(device)
        model_path = self.params.get("model_path", "maelic/relsgg-vits16plus")
        self.model = RelateAnything.from_pretrained(model_path, device=device)
        self.default_vocabulary = list(self.model.predicates)
        self.lock = Lock()

    def predict(
        self, image: np.ndarray, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Both the detector and relation vocabulary are mutable across requests.
        with self.lock:
            result = self.detector(
                image,
                conf=float(
                    params.get(
                        "conf_threshold",
                        self.params.get("conf_threshold", 0.25),
                    )
                ),
                iou=float(
                    params.get(
                        "iou_threshold", self.params.get("iou_threshold", 0.7)
                    )
                ),
                verbose=False,
            )[0]
            max_boxes = int(self.params.get("max_boxes", 60))
            detections = result.boxes[:max_boxes]
            boxes = detections.xyxy.cpu().numpy()
            scores = detections.conf.cpu().numpy()
            labels = [result.names[int(c)] for c in detections.cls]
            shapes = []
            shape_ids = [uuid4().hex for _ in boxes]
            for box, score, label, shape_id in zip(
                boxes, scores, labels, shape_ids
            ):
                x1, y1, x2, y2 = map(float, box)
                shapes.append(
                    Shape(
                        label=label,
                        shape_type="rectangle",
                        points=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                        score=float(score),
                        shape_id=shape_id,
                    ).model_dump()
                )

            vocabulary = (
                parse_prompts(params.get("text_prompt") or "")
                or self.default_vocabulary
            )
            if vocabulary != self.model.predicates:
                self.model.set_vocabulary(vocabulary)
            relations = self.model.predict(
                image,
                boxes,
                box_labels=labels,
                topk=int(self.params.get("topk", 5)),
                max_boxes=max_boxes,
            )
            relation_threshold = float(
                params.get(
                    "relation_threshold",
                    self.params.get("relation_threshold", 0.25),
                )
            )
            relations = [
                {
                    "subject_id": shape_ids[t.subject_idx],
                    "object_id": shape_ids[t.object_idx],
                    "label": t.predicate,
                    "score": float(t.score),
                }
                for t in relations
                if t.score >= relation_threshold
            ]
            return {
                "shapes": shapes,
                "description": "",
                "relations": relations,
            }

    def unload(self):
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "detector"):
            del self.detector
