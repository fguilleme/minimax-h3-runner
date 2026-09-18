import subprocess
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from h3runner.longrun import (
    build_chunk_config,
    build_concat_filter,
    build_video_concat_filter,
    build_scheduled_chunk_config,
    extract_last_frame,
    segment_count,
    parse_prompt_schedule,
)


class LongRunTests(unittest.TestCase):
    def test_parse_prompt_schedule(self):
        self.assertEqual(
            parse_prompt_schedule("2:the woman walks|3:the woman looks left and right slowly"),
            [(2.0, "the woman walks"), (3.0, "the woman looks left and right slowly")],
        )

    def test_parse_prompt_schedule_rejects_invalid_entries(self):
        with self.assertRaisesRegex(ValueError, "duration"):
            parse_prompt_schedule("2:walks|foo:turns")
        with self.assertRaisesRegex(ValueError, "positive"):
            parse_prompt_schedule("0:walks")

    def test_parse_prompt_schedule_returns_none_for_plain_prompt(self):
        self.assertIsNone(parse_prompt_schedule("the woman walks"))
        self.assertIsNone(parse_prompt_schedule("A fox moves. Audio: forest ambience."))

    def test_build_scheduled_chunk_config_sets_prompt_and_requested_duration(self):
        config = build_scheduled_chunk_config(
            {"seed": 100, "prompt": "old", "length": 50, "output_fps": 10.0},
            index=1,
            continuation_frame=None,
            duration=3.0,
            prompt="the woman looks left and right slowly",
        )
        self.assertEqual(config["prompt"], "the woman looks left and right slowly")
        self.assertEqual(config["length"], 31)

    def test_first_scheduled_chunk_uses_requested_output_fps_without_boundary_frame(self):
        config = build_scheduled_chunk_config(
            {"seed": 100, "prompt": "old", "length": 50, "output_fps": 10.0},
            index=0,
            continuation_frame=None,
            duration=2.0,
            prompt="the woman walks",
        )
        self.assertEqual(config["length"], 20)

    def test_segment_count_accounts_for_deduplicated_boundary_frame(self):
        self.assertEqual(segment_count(5.6, segment_duration=5.6, fps=10.0), 1)
        self.assertEqual(segment_count(5.7, segment_duration=5.6, fps=10.0), 2)
        self.assertEqual(segment_count(11.1, segment_duration=5.6, fps=10.0), 2)
        self.assertEqual(segment_count(11.11, segment_duration=5.6, fps=10.0), 3)

    def test_segment_count_rejects_non_positive_duration(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            segment_count(0, segment_duration=5.6, fps=10.0)

    def test_build_chunk_config_advances_seed_and_chains_frame(self):
        base = {"seed": 100, "first_frame": "/source.png", "last_frame": None, "prompt": "dance"}
        first = build_chunk_config(base, index=0, continuation_frame=None)
        self.assertEqual(first["seed"], 100)
        self.assertEqual(first["first_frame"], "/source.png")

        with tempfile.TemporaryDirectory() as tmp:
            frame = Path(tmp) / "last.png"
            frame.write_bytes(b"image")
            second = build_chunk_config(base, index=1, continuation_frame=frame)
        self.assertEqual(second["seed"], 101)
        self.assertEqual(second["first_frame"], str(frame.resolve()))
        self.assertIsNone(second["last_frame"])

    def test_concat_filter_drops_one_boundary_frame_and_trims_target(self):
        graph = build_concat_filter(3, fps=10.0, target_duration=12.0)
        self.assertEqual(graph.count("trim=start=0.100000000"), 4)
        self.assertIn("concat=n=3:v=1:a=1", graph)
        self.assertIn("trim=duration=12.000000000", graph)
        self.assertIn("atrim=duration=12.000000000", graph)

    def test_scheduled_concat_trims_each_segment_to_its_timeline_duration(self):
        graph = build_concat_filter(
            2, fps=10.0, target_duration=5.0, segment_durations=[2.0, 3.0]
        )
        self.assertIn("[0:v]trim=duration=2.000000000", graph)
        self.assertIn("[0:a]atrim=duration=2.000000000", graph)
        self.assertIn("[1:v]trim=start=0.100000000:duration=3.000000000", graph)
        self.assertIn("[1:a]atrim=start=0.100000000:duration=3.000000000", graph)

    def test_video_only_concat_ignores_segment_audio(self):
        graph = build_video_concat_filter(3, fps=10.0, target_duration=12.0)
        self.assertEqual(graph.count("trim=start=0.100000000"), 2)
        self.assertIn("concat=n=3:v=1:a=0", graph)
        self.assertNotIn(":a]", graph)
        self.assertIn("trim=duration=12.000000000", graph)

    def test_scheduled_video_concat_trims_each_segment(self):
        graph = build_video_concat_filter(
            2, fps=10.0, target_duration=5.0, segment_durations=[2.0, 3.0]
        )
        self.assertIn("[0:v]trim=duration=2.000000000", graph)
        self.assertIn("[1:v]trim=start=0.100000000:duration=3.000000000", graph)

    def test_extract_last_frame_uses_final_decoded_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255))
            for index, color in enumerate(colors):
                Image.new("RGB", (32, 32), color).save(root / f"frame-{index}.png")
            video = root / "three.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-y", "-v", "error", "-framerate", "10",
                    "-i", str(root / "frame-%d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(video),
                ],
                check=True,
            )
            output = root / "last.png"
            extract_last_frame(video, output)
            red, green, blue = Image.open(output).convert("RGB").getpixel((16, 16))
            self.assertGreater(blue, red)
            self.assertGreater(blue, green)


if __name__ == "__main__":
    unittest.main()
