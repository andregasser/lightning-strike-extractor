from __future__ import annotations

import unittest

import cv2
import numpy as np

from lightning_extractor.config import Config
from lightning_extractor.detection import frame_geometry_score, percentile


class DetectionTests(unittest.TestCase):
    def test_percentile_interpolates(self) -> None:
        self.assertEqual(percentile([0.0, 10.0], 0.5), 5.0)
        self.assertEqual(percentile([], 0.5), 0.0)

    def test_geometry_rewards_new_bright_line(self) -> None:
        config = Config()
        previous = np.zeros((540, 960, 3), dtype=np.uint8)
        current = previous.copy()
        cv2.line(current, (150, 450), (700, 80), (255, 255, 255), 3)
        score, segments, area = frame_geometry_score(previous, current, config)
        self.assertGreater(score, 0)
        self.assertGreater(segments, 0)
        self.assertGreater(area, 0)

    def test_geometry_rewards_longer_channel(self) -> None:
        config = Config()
        background = np.zeros((540, 960, 3), dtype=np.uint8)
        short = background.copy()
        long = background.copy()
        cv2.line(short, (200, 450), (400, 250), (255, 255, 255), 2)
        cv2.line(long, (100, 500), (800, 60), (255, 255, 255), 2)

        short_score, _, _ = frame_geometry_score(background, short, config)
        long_score, _, _ = frame_geometry_score(background, long, config)

        self.assertGreater(long_score, short_score)

    def test_geometry_rewards_stronger_channel(self) -> None:
        config = Config()
        background = np.zeros((540, 960, 3), dtype=np.uint8)
        faint = background.copy()
        strong = background.copy()
        cv2.line(faint, (100, 500), (800, 60), (80, 80, 80), 2)
        cv2.line(strong, (100, 500), (800, 60), (255, 255, 255), 2)

        faint_score, _, _ = frame_geometry_score(background, faint, config)
        strong_score, _, _ = frame_geometry_score(background, strong, config)

        self.assertGreater(strong_score, faint_score)

    def test_geometry_rewards_branching_channel(self) -> None:
        config = Config()
        background = np.zeros((540, 960, 3), dtype=np.uint8)
        straight = background.copy()
        branched = background.copy()
        trunk = ((480, 500), (480, 80))
        cv2.line(straight, *trunk, (255, 255, 255), 2)
        cv2.line(branched, *trunk, (255, 255, 255), 2)
        cv2.line(branched, (480, 260), (300, 130), (255, 255, 255), 2)
        cv2.line(branched, (480, 320), (650, 190), (255, 255, 255), 2)

        straight_score, _, _ = frame_geometry_score(background, straight, config)
        branched_score, _, _ = frame_geometry_score(background, branched, config)

        self.assertGreater(branched_score, straight_score)

    def test_geometry_has_no_vertical_orientation_bias(self) -> None:
        config = Config()
        background = np.zeros((540, 960, 3), dtype=np.uint8)
        vertical = background.copy()
        horizontal = background.copy()
        cv2.line(vertical, (480, 480), (480, 60), (255, 255, 255), 2)
        cv2.line(horizontal, (270, 270), (690, 270), (255, 255, 255), 2)

        vertical_score, _, _ = frame_geometry_score(background, vertical, config)
        horizontal_score, _, _ = frame_geometry_score(background, horizontal, config)

        ratio = horizontal_score / vertical_score
        self.assertGreater(ratio, 0.8)
        self.assertLess(ratio, 1.25)

    def test_geometry_rewards_sideways_branching(self) -> None:
        config = Config()
        background = np.zeros((540, 960, 3), dtype=np.uint8)
        straight = background.copy()
        branched = background.copy()
        trunk = ((100, 270), (850, 270))
        cv2.line(straight, *trunk, (255, 255, 255), 2)
        cv2.line(branched, *trunk, (255, 255, 255), 2)
        cv2.line(branched, (340, 270), (470, 130), (255, 255, 255), 2)
        cv2.line(branched, (600, 270), (730, 410), (255, 255, 255), 2)

        straight_score, _, _ = frame_geometry_score(background, straight, config)
        branched_score, _, _ = frame_geometry_score(background, branched, config)

        self.assertGreater(branched_score, straight_score)

    def test_geometry_rejects_uniform_sky_brightening(self) -> None:
        config = Config()
        background = np.zeros((540, 960, 3), dtype=np.uint8)
        brightened = np.full_like(background, 100)

        score, segments, area = frame_geometry_score(background, brightened, config)

        self.assertEqual(score, 0)
        self.assertEqual(segments, 0)
        self.assertGreater(area, 0)


if __name__ == "__main__":
    unittest.main()
