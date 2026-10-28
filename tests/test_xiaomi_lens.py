import importlib.util
import unittest
from unittest.mock import Mock, patch

import numpy as np

from app.models.xiaomi_lens import XiaomiLens


class TestXiaomiLens(unittest.TestCase):
    def setUp(self):
        self.model = XiaomiLens(
            {
                "model_id": "xiaomi_lens",
                "display_name": "Xiaomi-Lens-0",
            }
        )

    def test_natural_language_is_preserved(self):
        text = "the cup, beside the pot. Not the lamp"
        self.assertIn(
            text, self.model._build_prompt("referring_segmentation", text)
        )
        self.assertEqual(
            self.model._build_prompt("fine_grained_perception", text), text
        )
        with self.assertRaises(ValueError):
            self.model._build_prompt("unsupported", text)

    def test_tasks_expose_text_batch_and_finish(self):
        tasks = self.model.get_metadata()["available_tasks"]
        self.assertEqual(len(tasks), 6)
        for task in tasks:
            self.assertEqual(task["batch_processing_mode"], "text_prompt")
            self.assertIn("button_finish_object", task["active_widgets"])

    @unittest.skipUnless(
        importlib.util.find_spec("infer"), "Requires the xiaomi_lens extra"
    )
    def test_answer_geometry_and_truncation(self):
        import torch

        self.model.device = torch.device("cpu")
        self.model.processor = Mock()
        self.model.model = Mock()
        self.model.model.generation_config.eos_token_id = 99
        self.model.mask_token_id = 3
        output = "<bbox>[0,0,1,1]</bbox><mask><|mask|></mask></think><bbox>[100,200,900,800]</bbox><mask><|mask|></mask><|im_end|>"
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        image[0, 0] = [1, 2, 3]
        with (
            patch("infer.build_inputs", return_value=({}, "")) as build,
            patch(
                "infer.generate_completion",
                return_value=(torch.tensor([[3, 99]]), output, 2),
            ) as generate,
            patch("infer.run_forward") as forward,
        ):
            result = self.model.predict(
                image,
                {
                    "current_task": "referring_comprehension",
                    "text_prompt": "dog",
                },
            )
            self.assertEqual(
                build.call_args.args[1].getpixel((0, 0)), (3, 2, 1)
            )
            self.assertEqual(
                result["shapes"][0]["points"],
                [[20, 20], [180, 20], [180, 80], [20, 80]],
            )
            self.assertEqual(result["shapes"][0]["label"], "AUTOLABEL_OBJECT")
            self.assertEqual(result["description"], "")
            forward.assert_not_called()
            masks = torch.full((1, 2, 10, 20), -10.0)
            masks[0, 0] = 10
            masks[0, 1, 2:8, 4:16] = 10
            forward.return_value = masks
            result = self.model.predict(
                image,
                {
                    "current_task": "referring_segmentation",
                    "text_prompt": "dog",
                },
            )
            self.assertEqual(len(result["shapes"]), 1)
            points = result["shapes"][0]["points"]
            self.assertTrue(
                all(40 <= x <= 159 and 20 <= y <= 79 for x, y in points)
            )
            generate.return_value = (torch.tensor([[1, 2]]), output, 2)
            with self.assertRaisesRegex(ValueError, "truncated"):
                self.model.predict(
                    image, {"text_prompt": "dog", "max_new_tokens": 2}
                )
            generate.return_value = (torch.tensor([[99]]), "thinking", 0)
            with self.assertRaisesRegex(ValueError, "reasoning block"):
                self.model.predict(image, {"text_prompt": "dog"})
            generate.return_value = (
                torch.tensor([[99]]),
                "</think>No target.<|im_end|>",
                0,
            )
            result = self.model.predict(image, {"text_prompt": "dog"})
            self.assertEqual(
                result, {"shapes": [], "description": "No target."}
            )


if __name__ == "__main__":
    unittest.main()
