from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
import yaml

from app.models.groundanything import GroundAnything, GroundAnythingVLM


class TestGroundAnything(unittest.TestCase):
    def test_registration_and_local_configs(self):
        from app.core.registry import ModelRegistry, _MODEL_REGISTRY

        root = Path(__file__).resolve().parents[1]
        for model_id, cls, repo in (
            ("groundanything", GroundAnything, "GroundAnything"),
            ("groundanything_vlm", GroundAnythingVLM, "GroundAnything-VLM"),
        ):
            with self.subTest(model_id=model_id):
                config = yaml.safe_load(
                    (
                        root / "configs/auto_labeling" / f"{model_id}.yaml"
                    ).read_text()
                )
                self.assertIs(_MODEL_REGISTRY[model_id], cls)
                ModelRegistry.__new__(ModelRegistry)._validate_configs(
                    [config]
                )
                self.assertEqual(config["params"]["backend"], "transformers")
                self.assertEqual(
                    config["params"]["model_path"], f"GroundingPI/{repo}"
                )
                self.assertEqual(
                    len(cls(config).get_metadata()["available_tasks"]), 8
                )

    def test_native_diffusion_output_and_truncation(self):
        model = GroundAnything(
            {"model_id": "groundanything", "display_name": "GroundAnything"}
        )
        with patch.object(model, "_load_transformers"):
            model.load()
        model.device = "cpu"
        model.processor = Mock()
        model.processor.tokenizer.decode.return_value = (
            "<|object_ref_start|>dog<|object_ref_end|>"
            "<|box_start|><0><0><999><999><|box_end|>"
        )
        model.model = Mock()
        model.model.im_end_token_id = 99
        image = np.zeros((101, 201, 3), dtype=np.uint8)
        image[0, 0] = [1, 2, 3]
        inputs = {
            "input_ids": torch.tensor([[10, 11]]),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        }
        with patch(
            "app.models.groundanything_impl.inputs.prepare_qwen3_inputs",
            return_value=inputs,
        ) as prepare:
            model.model.speculative_generate.return_value = torch.tensor(
                [[20, 99]]
            )
            result = model.predict(
                image, {"text_prompt": "dog", "max_new_tokens": 2}
            )
            self.assertEqual(
                result["shapes"][0]["points"],
                [[0, 0], [200, 0], [200, 100], [0, 100]],
            )
            supplied_image = prepare.call_args.args[1][0]["content"][0][
                "image"
            ]
            self.assertEqual(supplied_image.getpixel((0, 0)), (3, 2, 1))
            self.assertEqual(
                model.model.speculative_generate.call_args.kwargs[
                    "max_new_tokens"
                ],
                2,
            )
            model.model.generate.assert_not_called()
            model.model.speculative_generate.return_value = torch.tensor(
                [[20, 21]]
            )
            with self.assertRaisesRegex(ValueError, "incomplete generation"):
                model.predict(
                    image, {"text_prompt": "dog", "max_new_tokens": 2}
                )
            model.params["max_model_len"] = 2
            with self.assertRaisesRegex(ValueError, "exceeds max_model_len"):
                model.predict(image, {"text_prompt": "dog"})


if __name__ == "__main__":
    unittest.main()
