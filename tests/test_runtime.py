import unittest
from types import SimpleNamespace

from h3runner.runtime import configure_memory_mode


class RuntimeTests(unittest.TestCase):
    def test_headless_runner_uses_lowvram_for_full_resolution_keyframes(self):
        args = SimpleNamespace()

        configure_memory_mode(args)

        self.assertTrue(args.lowvram)
        self.assertTrue(args.disable_async_offload)
        self.assertTrue(args.disable_pinned_memory)
        self.assertFalse(args.novram)
        self.assertFalse(args.highvram)


if __name__ == "__main__":
    unittest.main()