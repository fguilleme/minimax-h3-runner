import json
import tempfile
import unittest
from pathlib import Path

from h3runner.config import H3Config, align_frame_count, load_config, temporal_shape


class ConfigTests(unittest.TestCase):
    def test_frame_count_snaps_to_h3_grid(self):
        self.assertEqual(align_frame_count(5), 5)
        self.assertEqual(align_frame_count(50), 56)
        self.assertEqual(align_frame_count(56), 56)

    def test_temporal_shape_matches_working_render(self):
        self.assertEqual(temporal_shape(50), (56, 17, 93))

    def test_config_keeps_generation_fps_separate_from_output_fps(self):
        config = H3Config(length=50, output_fps=10.0)
        self.assertEqual(config.aligned_frames, 56)
        self.assertEqual(config.output_fps, 10.0)
        self.assertEqual(config.output_duration, 5.6)

    def test_load_config_rejects_unknown_keys_and_converts_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"length": 22, "comfy_root": "/tmp/comfy"}))
            config = load_config(path)
            self.assertEqual(config.length, 22)
            self.assertEqual(config.comfy_root, Path("/tmp/comfy"))
            path.write_text(json.dumps({"unknown": 1}))
            with self.assertRaisesRegex(ValueError, "unknown config keys"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
