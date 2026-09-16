import tempfile
import unittest
from pathlib import Path

from h3runner.run import input_signature, phase_plan


class RunTests(unittest.TestCase):
    def test_phase_plan_skips_completed_artifacts(self):
        existing = {"conditioning", "empty-latent"}
        self.assertEqual(phase_plan(existing, force=False), ["denoise", "decode"])

    def test_phase_plan_force_runs_everything(self):
        self.assertEqual(phase_plan(set(), force=True), ["encode", "denoise", "decode"])

    def test_phase_plan_requires_both_encode_outputs(self):
        self.assertEqual(phase_plan({"conditioning"}, force=False), ["encode", "denoise", "decode"])

    def test_input_signature_changes_with_keyframe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "start.png"
            config = root / "config.json"
            image.write_bytes(b"first")
            config.write_text('{"first_frame":"start.png"}')
            before = input_signature(config)
            image.write_bytes(b"second")
            after = input_signature(config)
            self.assertNotEqual(before, after)


if __name__ == "__main__":
    unittest.main()
