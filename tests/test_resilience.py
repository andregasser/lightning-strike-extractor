from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from lse.config import Config
from lse.detection import detect_flashes
from lse.pipeline import analyze, run_identity


def _write_test_video(path: Path, fps: float = 10.0, frames: int = 40) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (160, 90))
    if not writer.isOpened():
        raise RuntimeError("Test video writer is unavailable")
    for index in range(frames):
        frame = np.zeros((90, 160, 3), dtype=np.uint8)
        if index in {20, 21}:
            frame[:] = 180
            cv2.line(frame, (30, 80), (110, 10), (255, 255, 255), 2)
        writer.write(frame)
    writer.release()


class RunIdentityTests(unittest.TestCase):
    def test_identity_changes_with_range_and_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "source.mp4"
            video.write_bytes(b"video")
            source: dict[str, object] = {"size_bytes": video.stat().st_size}
            default = Config()
            first = run_identity(video, source, default, 0.0, None)
            self.assertNotEqual(first, run_identity(video, source, default, 1.0, None))
            changed = Config()
            changed.export.top = 10
            self.assertNotEqual(first, run_identity(video, source, changed, 0.0, None))


class FlashCheckpointTests(unittest.TestCase):
    def test_interrupted_scan_resumes_to_same_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "fixture.avi"
            _write_test_video(video)
            config = Config()
            config.analysis.width = 160
            config.analysis.checkpoint_seconds = 0.5
            uninterrupted = detect_flashes(video, 10.0, config)

            def interrupt(completed: int, total: int) -> None:
                del total
                if completed >= 15:
                    raise KeyboardInterrupt

            checkpoint = root / "checkpoint"
            with self.assertRaises(KeyboardInterrupt):
                detect_flashes(
                    video,
                    10.0,
                    config,
                    progress=interrupt,
                    checkpoint_dir=checkpoint,
                )
            self.assertTrue(list(checkpoint.glob("chunk-*.npz")))
            resumed = detect_flashes(
                video,
                10.0,
                config,
                checkpoint_dir=checkpoint,
                resume=True,
            )
            self.assertGreaterEqual(len(resumed), 1)
            self.assertEqual(
                [event.as_dict() for event in uninterrupted],
                [event.as_dict() for event in resumed],
            )


class RunStateTests(unittest.TestCase):
    def test_interruption_is_recorded_in_run_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "source.mp4"
            video.write_bytes(b"video")
            source: dict[str, object] = {
                "size_bytes": video.stat().st_size,
                "duration_seconds": 10.0,
                "video": {"fps": 25.0},
            }
            with (
                patch("lse.pipeline.probe_video", return_value=source),
                patch(
                    "lse.pipeline.detect_flashes",
                    side_effect=KeyboardInterrupt,
                ),
                self.assertRaises(KeyboardInterrupt),
            ):
                analyze(video, root / "runs", Config())
            run_state = next((root / "runs").glob("*/run.json"))
            state = json.loads(run_state.read_text())
            self.assertEqual(state["status"], "interrupted")
            self.assertEqual(state["phase"], "flash-scan")


if __name__ == "__main__":
    unittest.main()
