import base64
import importlib.util
import unittest
from io import BytesIO
from unittest.mock import Mock

import numpy as np
from PIL import Image

from app.models.wevisone_vl import WeVisOneVLModel


@unittest.skipUnless(
    importlib.util.find_spec("wevisone_vl"), "Requires wevisone_vl extra"
)
class TestWeVisOneVL(unittest.TestCase):
    def test_descriptions_show_only_parsed_results(self):
        from wevisone_vl.engine import InferenceResult

        image = np.zeros((100, 200, 3), dtype=np.uint8)
        response = (
            "<ref>dog</ref><box><0><0><500><500></box>\nTotal number: 9."
        )
        self.model.model.predict.return_value = InferenceResult(response, "")
        for task in ("detection", "referring", "ocr", "ocr_grounding"):
            result = self.model.predict(
                image, {"current_task": task, "text_prompt": "dog"}
            )
            self.assertEqual(result["description"], "")
        result = self.model.predict(
            image, {"current_task": "count_box", "text_prompt": "dogs"}
        )
        self.assertEqual(result["description"], "dogs: 1")
        self.model.model.predict.return_value = InferenceResult(
            "A <ref>dog</ref><box><0><0><500><500></box> is beside "
            "a <ref>plant</ref><point><750><500></point>.<|im_end|>",
            "",
        )
        result = self.model.predict(
            image, {"current_task": "freeform", "text_prompt": "Describe."}
        )
        self.assertEqual(result["description"], "A dog is beside a plant.")

    def setUp(self):
        from wevisone_vl.visualization import parse_predictions

        self.model = WeVisOneVLModel(
            {"model_id": "wevisone_vl", "display_name": "WeVisOne-VL"}
        )
        self.model.model = Mock()
        self.model.parse_predictions = parse_predictions

    def test_mask_export_preserves_alpha_and_rgb(self):
        from wevisone_vl.engine import InferenceResult

        alpha = np.array([[0, 0.25], [0.75, 1]], dtype=np.float32)
        self.model.model.predict.return_value = InferenceResult(
            "<ref>dog</ref><box><0><0><1000><1000><OBJ></box>",
            "",
            [[0, 0, 1, 1]],
            [alpha],
        )
        image = np.full((2, 2, 3), [10, 20, 30], dtype=np.uint8)
        result = self.model.predict(
            image, {"current_task": "matting", "text_prompt": "dog"}
        )
        self.assertEqual(result["description"], "")
        cutout = np.asarray(
            Image.open(
                BytesIO(base64.b64decode(result['artifacts'][1]['png_base64']))
            )
        )
        np.testing.assert_array_equal(cutout[:, :, 3], [[0, 63], [191, 255]])
        np.testing.assert_array_equal(cutout[0, 0, :3], [30, 20, 10])
        result = self.model.predict(
            image, {"current_task": "segmentation", "text_prompt": "dog"}
        )
        mask = np.asarray(
            Image.open(
                BytesIO(base64.b64decode(result['artifacts'][0]['png_base64']))
            )
        )
        np.testing.assert_array_equal(mask, [[0, 0], [255, 255]])

    def test_reference_boxes_are_normalized_and_validated(self):
        boxes = self.model._boxes(
            {
                "marks": [
                    {
                        "type": "rectangle",
                        "label": 1,
                        "data": [40, 80, -10, 10],
                    }
                ]
            },
            100,
            60,
        )
        self.assertEqual(boxes, [[0, 10, 40, 60]])
        for data in ([1, 2, 1, 3], [0, 0, float('nan'), 3]):
            with self.assertRaises(ValueError):
                self.model._boxes(
                    {
                        "marks": [
                            {"type": "rectangle", "label": 1, "data": data}
                        ]
                    },
                    100,
                    100,
                )

    def test_queries_and_original_coordinates(self):
        from wevisone_vl.engine import InferenceResult

        self.model.model.predict.return_value = InferenceResult(
            "<ref>dog</ref><point><250><750></point>", ""
        )
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        for text in ('person.car', 'person,car', 'person\ncar'):
            self.model.predict(image, {"text_prompt": text})
            self.assertEqual(
                self.model.model.predict.call_args.kwargs['query'],
                ['person', 'car'],
            )
        text = 'the dog, beside the sofa.'
        result = self.model.predict(
            image, {"current_task": "point", "text_prompt": text}
        )
        self.assertEqual(
            self.model.model.predict.call_args.kwargs['query'], text
        )
        self.assertEqual(result['shapes'][0]['points'], [[50, 75]])
        self.model.model.predict.return_value = InferenceResult(
            '<ref>dog</ref><box>None</box>', ''
        )
        self.assertEqual(
            self.model.predict(image, {"text_prompt": "dog"})['shapes'], []
        )
