import unittest

import torch

from h3runner.latents import flatten_av_latent, restore_av_latent


class FakeNested:
    is_nested = True

    def __init__(self, tensors):
        self.tensors = tuple(tensors)

    def unbind(self):
        return self.tensors


class LatentTests(unittest.TestCase):
    def test_flatten_and_restore_av_latent(self):
        video = torch.zeros(1, 24, 17, 22, 38)
        audio = torch.ones(1, 32, 2, 93)
        latent = {"samples": FakeNested((video, audio)), "batch_index": torch.tensor([0])}
        flat = flatten_av_latent(latent)
        self.assertEqual(set(flat), {"video", "audio", "extras"})
        torch.testing.assert_close(flat["video"], video)
        torch.testing.assert_close(flat["audio"], audio)
        restored = restore_av_latent(flat, FakeNested)
        self.assertTrue(restored["samples"].is_nested)
        torch.testing.assert_close(restored["samples"].unbind()[0], video)
        torch.testing.assert_close(restored["batch_index"], torch.tensor([0]))

    def test_rejects_non_av_latent(self):
        with self.assertRaisesRegex(ValueError, "two streams"):
            flatten_av_latent({"samples": FakeNested((torch.zeros(1),))})


if __name__ == "__main__":
    unittest.main()
