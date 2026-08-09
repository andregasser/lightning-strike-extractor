from __future__ import annotations

import unittest

import cv2
import numpy as np

from lse.config import Config
from lse.detection import (
    _apply_multiframe_peak_quality,
    _apply_multiframe_support,
    frame_channel_metrics,
    frame_geometry_score,
    percentile,
)
from lse.models import CandidateFrame


def textured_scene() -> np.ndarray:
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    for x in range(20, 640, 40):
        cv2.line(image, (x, 40), (x, 340), (80 + x % 120,) * 3, 2)
    for y in range(30, 360, 45):
        cv2.circle(image, (100 + y, y), 12, (180, 180, 180), 2)
    cv2.putText(image, "STATIC", (220, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (220,) * 3, 3)
    return image


class DetectionTests(unittest.TestCase):
    def test_peak_template_recovers_saturated_adjacent_frame(self) -> None:
        config = Config()
        target = CandidateFrame(
            0,
            "event",
            10,
            0.10,
            1.0,
            0,
            1000.0,
            channel_length=10.0,
            channel_thickness=1.0,
            frame_quality=1.0,
        )
        template = CandidateFrame(
            0,
            "event",
            11,
            0.11,
            100.0,
            1,
            10.0,
            channel_length=80.0,
            branch_points=3,
            channel_thickness=1.0,
            frame_quality=100.0,
            multiframe_support=1.0,
        )
        empty = np.zeros((40, 40), dtype=np.uint8)
        channel = empty.copy()
        cv2.line(channel, (20, 35), (20, 5), 255, 1)
        bright = np.full((40, 40), 240, dtype=np.uint8)
        normal = np.full((40, 40), 120, dtype=np.uint8)

        _apply_multiframe_peak_quality(
            [target, template],
            [empty, channel],
            [bright, normal],
            config,
        )

        self.assertEqual(target.channel_template_frame_number, template.frame_number)
        self.assertEqual(target.peak_multiframe_support, 1.0)
        self.assertGreater(target.frame_quality, template.frame_quality)

    def test_peak_template_does_not_cross_frame_radius(self) -> None:
        config = Config()
        candidates = [
            CandidateFrame(0, "event", index, index / 100.0, 1.0, 0, 1.0)
            for index in range(4)
        ]
        candidates[-1].multiframe_support = 1.0
        candidates[-1].channel_length = 100.0
        candidates[-1].channel_thickness = 1.0
        masks = [np.zeros((20, 20), dtype=np.uint8) for _ in candidates]
        cv2.line(masks[-1], (10, 18), (10, 2), 255, 1)
        grays = [np.full((20, 20), 250, dtype=np.uint8) for _ in candidates]

        _apply_multiframe_peak_quality(candidates, masks, grays, config)

        self.assertNotEqual(
            candidates[0].channel_template_frame_number,
            candidates[-1].frame_number,
        )

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

    def test_frame_quality_weights_channel_luminance_superlinearly(self) -> None:
        config = Config()
        background = np.zeros((540, 960, 3), dtype=np.uint8)
        medium = background.copy()
        strong = background.copy()
        cv2.line(medium, (100, 500), (800, 60), (100, 100, 100), 2)
        cv2.line(strong, (100, 500), (800, 60), (200, 200, 200), 2)

        medium_metrics = frame_channel_metrics(background, medium, config)
        strong_metrics = frame_channel_metrics(background, strong, config)

        self.assertGreater(strong_metrics.frame_quality, medium_metrics.frame_quality * 3)

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
