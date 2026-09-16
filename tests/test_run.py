import unittest

from h3runner.run import phase_plan


class RunTests(unittest.TestCase):
    def test_phase_plan_skips_completed_artifacts(self):
        existing = {"conditioning", "empty-latent"}
        self.assertEqual(phase_plan(existing, force=False), ["denoise", "decode"])

    def test_phase_plan_force_runs_everything(self):
        self.assertEqual(phase_plan(set(), force=True), ["encode", "denoise", "decode"])

    def test_phase_plan_requires_both_encode_outputs(self):
        self.assertEqual(phase_plan({"conditioning"}, force=False), ["encode", "denoise", "decode"])


if __name__ == "__main__":
    unittest.main()
