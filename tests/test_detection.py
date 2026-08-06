from __future__ import annotations

import cv2
import numpy as np
import unittest

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


if __name__ == "__main__":
    unittest.main()
