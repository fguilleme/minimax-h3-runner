import unittest

from h3runner.decode import atempo_factors


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


if __name__ == "__main__":
    unittest.main()
