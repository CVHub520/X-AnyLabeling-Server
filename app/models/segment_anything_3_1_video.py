import threading
from pathlib import Path

import torch
from loguru import logger

from app.core.registry import register_model
from .segment_anything_3_video import SegmentAnything3Video


class _MultiplexPredictor:
    def __init__(self, predictor, device):
        self.predictor = predictor
        self.device = device
        self.lock = threading.RLock()

    def handle_request(self, request):
        request = dict(request)
        if request["type"] == "start_session":
            request["offload_video_to_cpu"] = True
        with (
            self.lock,
            torch.cuda.device(self.device),
            torch.autocast(device_type="cuda", dtype=torch.bfloat16),
        ):
            return self._normalize_outputs(
                self.predictor.handle_request(request)
            )

    def handle_stream_request(self, request):
        request = dict(request)
        if "max_frame_num_to_track" in request:
            start = request.get("start_frame_index")
            if start is None:
                state = self.predictor._get_session(request["session_id"])[
                    "state"
                ]
                start = min(state["cached_frame_outputs"], default=0)
            request["max_frame_num_to_track"] -= start
            if request["max_frame_num_to_track"] < 0:
                raise ValueError("end_frame must not precede start_frame")
        with (
            self.lock,
            torch.cuda.device(self.device),
            torch.autocast(device_type="cuda", dtype=torch.bfloat16),
        ):
            for response in self.predictor.handle_stream_request(request):
                yield self._normalize_outputs(response)

    @staticmethod
    def _normalize_outputs(response):
        outputs = response.get("outputs", {})
        for key in (
            "out_obj_ids",
            "out_probs",
            "out_boxes_xywh",
            "out_binary_masks",
        ):
            value = outputs.get(key)
            if isinstance(value, torch.Tensor):
                outputs[key] = value.detach().cpu().numpy().copy()
        return response

    def shutdown(self):
        with self.lock, torch.cuda.device(self.device):
            self.predictor.shutdown()


@register_model("segment_anything_3_1_video")
class SegmentAnything31Video(SegmentAnything3Video):
    """SAM 3.1 multiplex video segmentation with text and point prompts."""

    def load(self):
        from .segment_anything_3_1_impl.model_builder import (
            build_sam3_multiplex_video_predictor,
        )

        device = torch.device(self.params.get("device", "cuda:0"))
        if device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("SAM 3.1 Video requires a CUDA device")
        model_path = self.params.get("model_path", "sam3.1_multiplex.pt")
        if not Path(model_path).is_file():
            raise FileNotFoundError(model_path)
        max_num_objects = self.params.get("max_num_objects", 16)
        if not isinstance(max_num_objects, int) or max_num_objects < 1:
            raise ValueError("max_num_objects must be a positive integer")
        bpe_path = self.params.get("bpe_path") or str(
            Path(__file__).parent / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
        )
        with torch.cuda.device(device):
            predictor = build_sam3_multiplex_video_predictor(
                checkpoint_path=model_path,
                bpe_path=bpe_path,
                max_num_objects=max_num_objects,
                use_fa3=False,
                use_rope_real=False,
                compile=False,
                async_loading_frames=False,
                default_output_prob_thresh=0.0,
            )
            predictor.bf16_context.__exit__(None, None, None)
        predictor.model.use_batched_grounding = False
        predictor.model.batched_grounding_batch_size = 1
        predictor.model.postprocess_batch_size = 1
        self.predictor = _MultiplexPredictor(predictor, device)
        logger.info(f"SAM 3.1 video model loaded on {device}")

    def add_prompt(self, session_id, text_prompt, frame_index, params):
        result = super().add_prompt(
            session_id, text_prompt, frame_index, params
        )
        if "error" not in result:
            session = self._get_session(session_id)
            session.is_point_prompt = False
            session.text_prompt = text_prompt
            if not result["masks"]:
                session.last_prompt_frame = frame_index
                session.prompt_frame_outputs = None
                session.prompt_frame_params = params.copy()
        return result

    def add_point_prompt(
        self, session_id, points, point_labels, obj_id, frame_index, params
    ):
        session = self._get_session(session_id)
        if session is None:
            return {"error": f"Session {session_id} not found"}
        relative_frame_index = frame_index - session.start_frame_index
        if not 0 <= relative_frame_index < len(session.frames):
            return {"error": f"Frame index {frame_index} out of range"}
        if not points or len(points) != len(point_labels):
            return {"error": "Points and point_labels must have same length"}
        outputs = session.predictor.handle_request(
            dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=relative_frame_index,
                points=points,
                point_labels=point_labels,
                obj_id=99999 if obj_id is None else obj_id,
            )
        )["outputs"]
        session.is_point_prompt = True
        session.last_prompt_frame = frame_index
        session.prompt_frame_outputs = outputs
        session.prompt_frame_params = params.copy()
        inference_params = self._get_inference_params(params)
        height, width = self._get_frame_dimensions(session.frames)
        shapes = self._convert_outputs_to_shapes(
            outputs["out_binary_masks"],
            outputs["out_probs"],
            outputs["out_obj_ids"],
            outputs["out_boxes_xywh"],
            session.text_prompt or "AUTOLABEL_OBJECT",
            **inference_params,
            orig_width=width,
            orig_height=height,
        )
        return {
            "frame_index": frame_index,
            "masks": shapes,
            "num_objects": len(shapes),
        }

    def unload(self):
        with self._tasks_lock:
            for task in self._tasks.values():
                task.cancel()
        self._executor.shutdown(wait=True)
        self._tasks.clear()
        with self._sessions_lock:
            for session in self._sessions.values():
                session.cleanup()
            self._sessions.clear()
        if self.predictor is not None:
            self.predictor.shutdown()
            self.predictor = None
        logger.info("SAM 3.1 video model unloaded")
