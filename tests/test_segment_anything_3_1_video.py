import unittest

import numpy as np
import torch

from app.models.segment_anything_3_1_impl.model.sam3_multiplex_tracking import (
    Sam3MultiplexTracking,
)
from app.models.segment_anything_3_1_video import _MultiplexPredictor


class MultiplexOutputTests(unittest.TestCase):
    def test_first_point_prompt_without_detector_cache(self):
        mask = torch.ones((1, 8, 8), dtype=torch.bool)
        state = {"cached_frame_outputs": {}}
        outputs = Sam3MultiplexTracking._build_sam2_output(
            None, state, 3, {7: mask}
        )
        self.assertIs(outputs[7], mask)
        self.assertEqual(state["cached_frame_outputs"], {})

    def test_refinement_preserves_other_objects_and_cached_frame(self):
        original = torch.zeros((1, 8, 8), dtype=torch.bool)
        refined = torch.ones_like(original)
        state = {"cached_frame_outputs": {3: {1: original, 2: original}}}
        outputs = Sam3MultiplexTracking._build_sam2_output(
            None, state, 3, {1: refined}
        )
        self.assertIs(outputs[1], refined)
        self.assertIs(outputs[2], original)
        self.assertIs(state["cached_frame_outputs"][3][1], original)

    def test_tensor_results_do_not_alias_reusable_output_buffer(self):
        buffer = torch.ones((1, 8, 8), dtype=torch.bool)
        response = {
            "frame_index": 0,
            "outputs": {
                "out_obj_ids": torch.tensor([7]),
                "out_binary_masks": buffer,
            },
        }
        normalized = _MultiplexPredictor._normalize_outputs(response)
        buffer.zero_()
        self.assertIsInstance(normalized["outputs"]["out_obj_ids"], np.ndarray)
        self.assertTrue(normalized["outputs"]["out_binary_masks"].all())


if __name__ == "__main__":
    unittest.main()
