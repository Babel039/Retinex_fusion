import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'Enhancement'))

from inference_utils import self_ensemble, tiled_forward


class RecordingDouble(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shapes = []

    def forward(self, x):
        self.shapes.append(tuple(x.shape[-2:]))
        return x * 2


class InferenceUtilsTest(unittest.TestCase):
    def test_tiled_forward_limits_patch_size_and_reconstructs_output(self):
        model = RecordingDouble()
        image = torch.arange(3 * 12 * 16, dtype=torch.float32).reshape(
            1, 3, 12, 16)

        result = tiled_forward(image, model, tile_size=8, overlap=4)

        torch.testing.assert_close(result, image * 2)
        self.assertGreater(len(model.shapes), 1)
        self.assertTrue(all(h <= 8 and w <= 8 for h, w in model.shapes))

    def test_self_ensemble_streams_through_tiled_forward(self):
        model = RecordingDouble()
        image = torch.rand(1, 3, 12, 16)

        result = self_ensemble(
            image,
            model,
            forward_fn=lambda value, network: tiled_forward(
                value, network, tile_size=8, overlap=4))

        torch.testing.assert_close(result, image * 2)
        self.assertGreater(len(model.shapes), 8)


if __name__ == '__main__':
    unittest.main()
