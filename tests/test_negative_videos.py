from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from lse.config import Config
from lse.detection import rank_event_frames
from lse.models import FlashEvent
from lse.pipeline import select_export_candidates


def textured_frame() -> np.ndarray:
    frame = np.full((180, 320, 3), 45, dtype=np.uint8)
    for x in range(15, 320, 30):
        cv2.line(frame, (x, 20), (x, 165), (110 + x % 90,) * 3, 2)
    for y in range(25, 180, 35):
        cv2.circle(frame, (80 + y, y), 9, (190, 190, 190), 2)
    cv2.putText(frame, "STATIC", (105, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220,) * 3, 2)
    return frame


def write_video(path: Path, frames: list[np.ndarray], fps: float = 30.0) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("MJPEG test video writer is unavailable")
    for frame in frames:
        writer.write(frame)
    writer.release()


class NegativeVideoTests(unittest.TestCase):
    def config(self) -> Config:
        config = Config()
        config.analysis.event_window_before = 0.4
        config.analysis.event_window_after = 0.4
        config.channel.analysis_width = 640
        config.channel.stabilization_width = 320
        config.channel.stabilization_min_matches = 8
        return config

    def event(self) -> FlashEvent:
        return FlashEvent("evt_negative", 1, 0.5, 0.45, 0.55, 100.0, 20.0, 20.0, 20.0, 3)

    def test_uniform_cloud_brightening_has_no_exportable_channel(self) -> None:
        base = np.full((180, 320, 3), 50, dtype=np.uint8)
        frames = [base.copy() for _ in range(30)]
        for index in range(13, 18):
            frames[index][:] = 170

        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "cloud-brightening.avi"
            write_video(video, frames)
            candidates = rank_event_frames(video, 30.0, [self.event()], self.config())

        self.assertEqual(select_export_candidates(candidates, self.config()), [])

    def test_affine_camera_motion_has_no_exportable_channel(self) -> None:
        base = textured_frame()
        frames = []
        for index in range(30):
            transform = cv2.getRotationMatrix2D((160, 90), (index - 15) * 0.05, 1.0)
            transform[:, 2] += ((index - 15) * 0.25, (15 - index) * 0.12)
            frames.append(
                cv2.warpAffine(base, transform, (320, 180), borderMode=cv2.BORDER_REFLECT)
            )

        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "camera-motion.avi"
            write_video(video, frames)
            config = self.config()
            candidates = rank_event_frames(video, 30.0, [self.event()], config)

        self.assertEqual(select_export_candidates(candidates, config), [])


if __name__ == "__main__":
    unittest.main()
