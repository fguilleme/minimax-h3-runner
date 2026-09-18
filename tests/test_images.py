import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from h3runner.images import aspect_preserving_canvas, load_image_tensor


class ImageTests(unittest.TestCase):
    def test_load_image_tensor_is_rgb_float_bhwc(self):
        pixels = np.zeros((2, 3, 3), dtype=np.uint8)
        pixels[0, 0] = [255, 128, 0]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.png"
            Image.fromarray(pixels, "RGB").save(path)
            tensor = load_image_tensor(path)
        self.assertEqual(tuple(tensor.shape), (1, 2, 3, 3))
        self.assertEqual(str(tensor.dtype), "torch.float32")
        self.assertAlmostEqual(float(tensor[0, 0, 0, 0]), 1.0)
        self.assertAlmostEqual(float(tensor[0, 0, 0, 1]), 128 / 255)

    def test_aspect_preserving_canvas_uses_multiples_of_32(self):
        self.assertEqual(aspect_preserving_canvas(1920, 1080), (1344, 768))
        self.assertEqual(aspect_preserving_canvas(1080, 1920), (768, 1344))

    def test_aspect_preserving_canvas_rejects_invalid_dimensions(self):
        with self.assertRaises(ValueError):
            aspect_preserving_canvas(0, 1080)


if __name__ == "__main__":
    unittest.main()
