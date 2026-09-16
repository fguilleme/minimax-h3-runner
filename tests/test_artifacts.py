import tempfile
import unittest
from pathlib import Path

import torch

from h3runner.artifacts import load_tree, save_tree


class ArtifactTests(unittest.TestCase):
    def test_round_trip_nested_tensor_tree(self):
        value = [
            [torch.arange(6, dtype=torch.float32).reshape(2, 3), {"tags": torch.tensor([0, 1]), "name": "prompt", "enabled": True, "missing": None}],
            (torch.ones(2, dtype=torch.bfloat16), 7, 1.25),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "conditioning"
            save_tree(prefix, value, artifact_type="conditioning")
            restored, metadata = load_tree(prefix, expected_type="conditioning")

        self.assertIsInstance(restored, list)
        self.assertIsInstance(restored[1], tuple)
        torch.testing.assert_close(restored[0][0], value[0][0])
        torch.testing.assert_close(restored[0][1]["tags"], value[0][1]["tags"])
        torch.testing.assert_close(restored[1][0], value[1][0])
        self.assertEqual(restored[0][1]["name"], "prompt")
        self.assertTrue(restored[0][1]["enabled"])
        self.assertIsNone(restored[0][1]["missing"])
        self.assertEqual(metadata["artifact_type"], "conditioning")

    def test_rejects_unsupported_objects(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(TypeError, "unsupported"):
                save_tree(Path(tmp) / "bad", object(), artifact_type="bad")

    def test_expected_type_is_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "latent"
            save_tree(prefix, {"video": torch.zeros(1)}, artifact_type="latent")
            with self.assertRaisesRegex(ValueError, "artifact type"):
                load_tree(prefix, expected_type="conditioning")


if __name__ == "__main__":
    unittest.main()
