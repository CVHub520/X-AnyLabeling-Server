import base64
from io import BytesIO
from pathlib import Path
import unittest
from unittest.mock import patch, Mock

import numpy as np
from PIL import Image
import requests
import yaml

from app.models.groundingpi import GroundingPI


class TestGroundingPI(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1]
        config = yaml.safe_load(
            (path / "configs/auto_labeling/groundingpi.yaml").read_text()
        )
        self.model = GroundingPI(config)
        with patch.object(self.model, "_load_transformers"):
            self.model.load()
        self.model.backend = "openai"
        self.image = np.zeros((101, 201, 3), dtype=np.uint8)
        self.image[0, 0] = [1, 2, 3]

    def predict(self, output, params, finish="stop"):
        response = Mock()
        response.json.return_value = {
            "choices": [
                {"message": {"content": output}, "finish_reason": finish}
            ]
        }
        with patch(
            "app.models.groundingpi.requests.post", return_value=response
        ) as post:
            result = self.model.predict(self.image, params)
        return result, post

    def test_detection_transport_and_coordinates(self):
        result, post = self.predict(
            "<|object_ref_start|>car<|object_ref_end|>"
            "<|box_start|><0><0><999><999>,<100><200><500><600><|box_end|>",
            {"current_task": "detection", "text_prompt": "car, person\ncar"},
        )
        self.assertEqual(len(result["shapes"]), 2)
        self.assertEqual(
            result["shapes"][0]["points"],
            [[0, 0], [200, 0], [200, 100], [0, 100]],
        )
        self.assertIsNone(result["shapes"][0]["score"])
        self.assertEqual(
            post.call_args.args[0], "http://127.0.0.1:8001/v1/chat/completions"
        )
        body = post.call_args.kwargs["json"]
        self.assertFalse(body["skip_special_tokens"])
        self.assertFalse(body["spaces_between_special_tokens"])
        content = body["messages"][0]["content"]
        self.assertIn("car</c>person", content[1]["text"])
        decoded = base64.b64decode(
            content[0]["image_url"]["url"].split(",", 1)[1]
        )
        self.assertEqual(
            Image.open(BytesIO(decoded)).getpixel((0, 0)), (3, 2, 1)
        )

    def test_detection_category_separators(self):
        expected = self.model.protocol.build_dense_bbox_prompt(
            ["umbrella", "person"]
        )
        for text in (
            "umbrella.person",
            "umbrella,person",
            "umbrella\nperson",
            "umbrella.person,umbrella\nperson.",
        ):
            with self.subTest(text=text):
                self.assertEqual(
                    self.model._build_prompt(
                        "detection", {"text_prompt": text}, 201, 101
                    ),
                    expected,
                )

    def test_referring_point_preserves_description(self):
        result, post = self.predict(
            "<|object_ref_start|>target<|object_ref_end|>"
            "<|box_start|><999><0><|box_end|>",
            {
                "current_task": "pointing",
                "text_prompt": "red car, on the left",
            },
        )
        self.assertEqual(result["shapes"][0]["shape_type"], "point")
        self.assertEqual(result["shapes"][0]["points"], [[200, 0]])
        self.assertEqual(
            post.call_args.kwargs["json"]["messages"][0]["content"][1]["text"],
            "Point to the target referred to by the following description: red car, on the left.",
        )

    def test_grounding_absent(self):
        result, post = self.predict(
            "<|object_ref_start|>red car<|object_ref_end|>"
            "<|box_start|>None<|box_end|>",
            {"current_task": "grounding", "text_prompt": "red car"},
        )
        self.assertEqual(result["shapes"], [])
        self.assertIn(
            "target referred to",
            post.call_args.kwargs["json"]["messages"][0]["content"][1]["text"],
        )

    def test_additional_tasks(self):
        for task, text, prompt, coordinates, shape_type, label in (
            (
                "object_points",
                "car.person,car\nperson",
                "Point to: car</c>person.",
                "<999><0>",
                "point",
                "car",
            ),
            (
                "ocr",
                "",
                "OCR task detect all the text in box format.",
                "<0><0><999><999>",
                "rectangle",
                "Hello, world!",
            ),
            (
                "document_layout",
                "title.text,title\ntext",
                "Detect all document layout elements that match the following categories: title</c>text.",
                "<0><0><999><999>",
                "rectangle",
                "title",
            ),
            (
                "gui_grounding",
                "open settings, then select display",
                "Point to the UI element to click for the following instruction: open settings, then select display.",
                "<999><0>",
                "point",
                "target",
            ),
        ):
            with self.subTest(task=task):
                result, post = self.predict(
                    f"<|object_ref_start|>{label}<|object_ref_end|>"
                    f"<|box_start|>{coordinates}<|box_end|>",
                    {"current_task": task, "text_prompt": text},
                )
                self.assertEqual(
                    post.call_args.kwargs["json"]["messages"][0]["content"][1][
                        "text"
                    ],
                    prompt,
                )
                shape = result["shapes"][0]
                self.assertEqual(shape["label"], label)
                self.assertEqual(shape["shape_type"], shape_type)
                self.assertEqual(
                    shape["points"],
                    (
                        [[200, 0]]
                        if shape_type == "point"
                        else [[0, 0], [200, 0], [200, 100], [0, 100]]
                    ),
                )

    def test_visual_prompt_quantization(self):
        mark = {"type": "rectangle", "label": 1, "data": [200, 100, 0, 0]}
        prompt = self.model._build_prompt(
            "visual_prompting",
            {"marks": [mark, mark, {**mark, "label": 0}]},
            201,
            101,
        )
        self.assertEqual(prompt.count("<0><0><999><999>"), 1)
        self.assertIn("find all similar objects", prompt)
        with self.assertRaises(ValueError):
            self.model._build_prompt(
                "visual_prompting",
                {"marks": [{**mark, "data": [1, 1, 1, 2]}]},
                201,
                101,
            )

    def test_invalid_outputs_are_rejected(self):
        valid = "<|object_ref_start|>car<|object_ref_end|><|box_start|><0><0><999><999><|box_end|>"
        for output, finish in (
            (valid, "length"),
            (valid + "garbage", "stop"),
            (valid.replace("<999>", "<1000>"), "stop"),
            (valid.replace("<999><999>", "<0><0>"), "stop"),
        ):
            with (
                self.subTest(output=output, finish=finish),
                self.assertRaises(ValueError),
            ):
                self.predict(output, {"text_prompt": "car"}, finish)

    def test_visual_prompt_prediction_and_settings(self):
        self.model.params.update(
            api_key="test-key", model_name="custom-worker", timeout=45
        )
        result, post = self.predict(
            "<|object_ref_start|>object<|object_ref_end|>"
            "<|box_start|><0><0><999><999><|box_end|>",
            {
                "current_task": "visual_prompting",
                "max_new_tokens": 1024,
                "marks": [
                    {"type": "rectangle", "label": 1, "data": [0, 0, 200, 100]}
                ],
            },
        )
        self.assertEqual(result["shapes"][0]["label"], "AUTOLABEL_OBJECT")
        self.assertEqual(
            result["shapes"][0]["points"],
            [[0, 0], [200, 0], [200, 100], [0, 100]],
        )
        kwargs = post.call_args.kwargs
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(kwargs["timeout"], 45)
        self.assertEqual(kwargs["json"]["max_tokens"], 1024)
        self.assertEqual(kwargs["json"]["model"], "custom-worker")

    def test_registry_and_config(self):
        from app.core.registry import ModelRegistry, _MODEL_REGISTRY

        self.assertIs(_MODEL_REGISTRY["groundingpi"], GroundingPI)
        registry = ModelRegistry.__new__(ModelRegistry)
        registry._validate_configs([self.model.config])
        self.assertEqual(
            self.model.params["model_path"], "GroundingPI/GroundingPI"
        )
        self.assertEqual(self.model.params["backend"], "transformers")
        self.assertNotIn("api_base", self.model.params)

    def test_transformers_resolves_model_directory(self):
        import transformers

        for local in (False, True):
            with (
                self.subTest(local=local),
                patch.object(transformers, "__version__", "5.7.0"),
                patch(
                    "app.models.groundingpi.Path.is_dir", return_value=local
                ),
                patch(
                    "huggingface_hub.snapshot_download",
                    return_value="/cache/groundingpi",
                ) as download,
                patch.object(
                    transformers.AutoProcessor, "from_pretrained"
                ) as processor,
                patch.object(
                    transformers.AutoModelForCausalLM, "from_pretrained"
                ) as model,
            ):
                self.model._load_transformers()
                expected = (
                    "GroundingPI/GroundingPI"
                    if local
                    else "/cache/groundingpi"
                )
                self.assertEqual(processor.call_args.args[0], expected)
                self.assertEqual(model.call_args.args[0], expected)
                if local:
                    download.assert_not_called()
                else:
                    download.assert_called_once_with(
                        repo_id="GroundingPI/GroundingPI"
                    )

    def test_transformers_generation(self):
        import torch

        self.model.backend = "transformers"
        self.model.device = "cpu"
        processor = Mock()
        processor.apply_chat_template.return_value = (
            "image:<|image_pad|> answer:"
        )
        processor.image_processor.merge_size = 2
        processor.image_processor.return_value = {
            "image_grid_thw": torch.tensor([[1, 4, 4]]),
            "pixel_values": torch.zeros(1, 3),
        }
        processor.tokenizer.return_value = {
            "input_ids": torch.tensor([[10, 11]])
        }
        processor.tokenizer.decode.return_value = (
            "<|object_ref_start|>car<|object_ref_end|>"
            "<|box_start|><0><0><999><999><|box_end|>"
        )
        self.model.processor = processor
        self.model.model = Mock()
        self.model.model.generation_config.eos_token_id = 99
        self.model.model.generate.return_value = torch.tensor(
            [[10, 11, 20, 99]]
        )
        result = self.model.predict(
            self.image, {"text_prompt": "car", "max_new_tokens": 2}
        )
        self.assertEqual(
            result["shapes"][0]["points"],
            [[0, 0], [200, 0], [200, 100], [0, 100]],
        )
        self.assertEqual(
            processor.tokenizer.call_args.args[0].count("<|image_pad|>"), 4
        )
        processor.tokenizer.decode.assert_called_once_with(
            [20], skip_special_tokens=False
        )
        self.assertFalse(
            processor.apply_chat_template.call_args.kwargs["enable_thinking"]
        )
        self.assertFalse(
            self.model.model.generate.call_args.kwargs["do_sample"]
        )
        self.model.model.generate.return_value = torch.tensor(
            [[10, 11, 20, 21]]
        )
        with self.assertRaisesRegex(ValueError, "incomplete generation"):
            self.model.predict(
                self.image, {"text_prompt": "car", "max_new_tokens": 2}
            )
        self.model.params["max_model_len"] = 2
        with self.assertRaisesRegex(ValueError, "exceeds max_model_len"):
            self.model.predict(self.image, {"text_prompt": "car"})

    def test_missing_prompts_skip_request(self):
        with patch("app.models.groundingpi.requests.post") as post:
            for task in (
                "detection",
                "grounding",
                "pointing",
                "object_points",
                "document_layout",
                "gui_grounding",
                "visual_prompting",
            ):
                self.assertEqual(
                    self.model.predict(self.image, {"current_task": task})[
                        "shapes"
                    ],
                    [],
                )
            post.assert_not_called()
        with self.assertRaises(ValueError):
            self.model.predict(self.image, {"current_task": "unknown"})

    def test_http_errors_propagate(self):
        with patch(
            "app.models.groundingpi.requests.post",
            side_effect=requests.Timeout,
        ):
            with self.assertRaises(requests.Timeout):
                self.model.predict(self.image, {"text_prompt": "car"})

    def test_task_metadata(self):
        tasks = self.model.get_metadata()["available_tasks"]
        self.assertEqual(
            [t["id"] for t in tasks],
            [
                "detection",
                "grounding",
                "pointing",
                "object_points",
                "ocr",
                "document_layout",
                "gui_grounding",
                "visual_prompting",
            ],
        )
        self.assertIsNone(tasks[-1]["batch_processing_mode"])
        self.assertIn("add_pos_rect", tasks[-1]["active_widgets"])
        self.assertTrue(
            all(
                t["batch_processing_mode"] == "text_prompt"
                for t in tasks[:-1]
                if t["id"] != "ocr"
            )
        )
        ocr = next(t for t in tasks if t["id"] == "ocr")
        self.assertEqual(ocr["batch_processing_mode"], "default")
        self.assertIn("button_run", ocr["active_widgets"])
        self.assertNotIn("edit_text", ocr["active_widgets"])


if __name__ == "__main__":
    unittest.main()
