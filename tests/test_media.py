import unittest

from h3runner.decode import atempo_factors, audio_ffmpeg_options


class MediaTests(unittest.TestCase):
    def test_atempo_factors_stay_in_ffmpeg_range(self):
        factors = atempo_factors(10 / 24)
        self.assertTrue(all(0.5 <= value <= 2.0 for value in factors))
        product = 1.0
        for value in factors:
            product *= value
        self.assertAlmostEqual(product, 10 / 24, places=8)

    def test_atempo_identity_has_no_filter(self):
        self.assertEqual(atempo_factors(1.0), [])

    def test_loop_audio_preserves_native_tempo(self):
        input_options, audio_filter = audio_ffmpeg_options("loop", 10.0, 24.0)
        self.assertEqual(input_options, ["-stream_loop", "-1"])
        self.assertIsNone(audio_filter)

    def test_stretch_audio_retains_previous_atempo_behavior(self):
        input_options, audio_filter = audio_ffmpeg_options("stretch", 10.0, 24.0)
        self.assertEqual(input_options, [])
        self.assertIn("atempo=0.5", audio_filter)
        self.assertTrue(audio_filter.endswith(",apad"))


if __name__ == "__main__":
    unittest.main()
