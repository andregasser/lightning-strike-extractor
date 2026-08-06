from __future__ import annotations

import unittest

import cv2
import numpy as np

from lightning_extractor.config import Config
from lightning_extractor.detection import (
    _apply_multiframe_support,
    frame_channel_metrics,
    frame_geometry_score,
    percentile,
)
from lightning_extractor.models import CandidateFrame


def textured_scene() -> np.ndarray:
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    for x in range(20, 640, 40):
        cv2.line(image, (x, 40), (x, 340), (80 + x % 120,) * 3, 2)
    for y in range(30, 360, 45):
        cv2.circle(image, (100 + y, y), 12, (180, 180, 180), 2)
    cv2.putText(image, "STATIC", (220, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (220,) * 3, 3)
    return image


class DetectionTests(unittest.TestCase):
    def test_multiframe_support_rewards_repeated_channel_geometry(self) -> None:
        config = Config()
        candidates = [
            CandidateFrame(0, "event", 1, 0.00, 100.0, 1, 10.0, frame_quality=100.0),
            CandidateFrame(0, "event", 2, 0.02, 90.0, 1, 10.0, frame_quality=90.0),
            CandidateFrame(0, "event", 3, 0.20, 150.0, 1, 10.0, frame_quality=150.0),
        ]
        repeated = np.zeros((100, 100), dtype=np.uint8)
        cv2.line(repeated, (10, 90), (80, 10), 255, 2)
        shifted = np.zeros_like(repeated)
        cv2.line(shifted, (12, 90), (82, 10), 255, 2)
        isolated = np.zeros_like(repeated)
        cv2.line(isolated, (90, 90), (90, 10), 255, 2)

        _apply_multiframe_support(candidates, [repeated, shifted, isolated], config)

        self.assertGreater(candidates[0].multiframe_support, 0.9)
        self.assertGreater(candidates[0].multiframe_quality, candidates[0].frame_quality)
        self.assertEqual(candidates[2].multiframe_support, 0.0)
        self.assertEqual(candidates[2].multiframe_quality, candidates[2].frame_quality)

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

    def test_affine_stabilization_suppresses_camera_motion(self) -> None:
        config = Config()
        config.channel.analysis_width = 960
        config.channel.stabilization_width = 480
        config.channel.stabilization_min_matches = 8
        previous = textured_scene()
        transform = cv2.getRotationMatrix2D((320, 180), 1.2, 1.0)
        transform[:, 2] += (5, -3)
        current = cv2.warpAffine(previous, transform, (640, 360), borderMode=cv2.BORDER_REFLECT)

        config.channel.stabilization_enabled = False
        unstable = frame_channel_metrics(previous, current, config)
        config.channel.stabilization_enabled = True
        stabilized = frame_channel_metrics(previous, current, config)

        self.assertLess(stabilized.bright_area, unstable.bright_area * 0.35)

    def test_affine_stabilization_preserves_new_lightning_channel(self) -> None:
        config = Config()
        config.channel.analysis_width = 960
        config.channel.stabilization_width = 480
        config.channel.stabilization_min_matches = 8
        previous = textured_scene()
        transform = cv2.getRotationMatrix2D((320, 180), -1.0, 1.0)
        transform[:, 2] += (-4, 3)
        moved = cv2.warpAffine(previous, transform, (640, 360), borderMode=cv2.BORDER_REFLECT)
        lightning = moved.copy()
        cv2.line(lightning, (80, 300), (500, 40), (255, 255, 255), 3)
        cv2.line(lightning, (300, 165), (500, 250), (255, 255, 255), 2)

        motion_only = frame_channel_metrics(previous, moved, config)
        with_lightning = frame_channel_metrics(previous, lightning, config)

        self.assertGreater(with_lightning.channel_length, motion_only.channel_length)
        self.assertGreater(with_lightning.frame_quality, motion_only.frame_quality)


if __name__ == "__main__":
    unittest.main()
