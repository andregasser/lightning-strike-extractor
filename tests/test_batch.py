from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lightning_extractor.batch import VideoJob, load_manifest, run_batch
from lightning_extractor.config import Config


class ManifestTests(unittest.TestCase):
    def test_manifest_resolves_relative_files_and_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = root / "media"
            media.mkdir()
            (media / "one.mp4").write_bytes(b"one")
            (media / "two-preview.mp4").write_bytes(b"two")
            manifest_path = root / "batch.toml"
            manifest_path.write_text(
                """
[batch]
jobs = 2
output = "result"

[[input]]
path = "media"
recursive = true
include = "*.mp4"
exclude = ["*preview*"]
start = 2.5
""".strip()
            )
            manifest = load_manifest(manifest_path)
            self.assertEqual(manifest.defaults.jobs, 2)
            self.assertEqual(manifest.defaults.output, (root / "result").resolve())
            self.assertEqual(len(manifest.jobs), 1)
            self.assertEqual(manifest.jobs[0].path.name, "one.mp4")
            self.assertEqual(manifest.jobs[0].start_seconds, 2.5)


class BatchExecutionTests(unittest.TestCase):
    def test_completed_batch_is_skipped_on_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            videos = []
            for name in ("one.mp4", "two.mp4"):
                video = root / name
                video.write_bytes(name.encode())
                videos.append(video)
            jobs = [VideoJob(video, Config()) for video in videos]

            def fake_resolve(video: Path, runs: Path, *args: object, **kwargs: object) -> Path:
                del args, kwargs
                return runs / video.stem

            def fake_analyze(video: Path, runs: Path, *args: object, **kwargs: object) -> Path:
                del args, kwargs
                run = runs / video.stem
                (run / "results").mkdir(parents=True, exist_ok=True)
                (run / "run.json").write_text('{"status": "complete"}')
                (run / "results" / "summary.json").write_text(
                    '{"events": 2, "candidate_frames": 3, "exported_stills": 1}'
                )
                return run

            with (
                patch("lightning_extractor.batch.resolve_run_path", side_effect=fake_resolve),
                patch(
                    "lightning_extractor.batch.analyze", side_effect=fake_analyze
                ) as analyze_mock,
            ):
                first = run_batch(jobs, root / "runs", progress_mode="quiet", worker_count=2)
                created_at = json.loads((first.path / "batch.json").read_text())["created_at"]
                second = run_batch(
                    jobs, root / "runs", progress_mode="quiet", worker_count=2, resume=True
                )

            self.assertEqual(analyze_mock.call_count, 2)
            self.assertEqual([item.status for item in first.items], ["complete", "complete"])
            self.assertEqual([item.status for item in second.items], ["skipped", "skipped"])
            summary = json.loads((second.path / "summary.json").read_text())
            self.assertEqual(summary["counts"]["skipped"], 2)
            self.assertEqual(
                json.loads((second.path / "batch.json").read_text())["created_at"], created_at
            )
            self.assertTrue((second.path / "summary.csv").exists())

    def test_one_invalid_video_does_not_stop_the_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = root / "good.mp4"
            bad = root / "bad.mp4"
            good.write_bytes(b"good")
            bad.write_bytes(b"bad")

            def fake_resolve(video: Path, runs: Path, *args: object, **kwargs: object) -> Path:
                del args, kwargs
                if video.name == "bad.mp4":
                    raise RuntimeError("cannot probe video")
                return runs / video.stem

            def fake_analyze(video: Path, runs: Path, *args: object, **kwargs: object) -> Path:
                del args, kwargs
                run = runs / video.stem
                (run / "results").mkdir(parents=True)
                (run / "run.json").write_text('{"status": "complete"}')
                (run / "results" / "summary.json").write_text('{"events": 1}')
                return run

            with (
                patch("lightning_extractor.batch.resolve_run_path", side_effect=fake_resolve),
                patch("lightning_extractor.batch.analyze", side_effect=fake_analyze),
            ):
                result = run_batch(
                    [VideoJob(good, Config()), VideoJob(bad, Config())],
                    root / "runs",
                    progress_mode="quiet",
                )

            self.assertEqual(result.exit_code, 1)
            self.assertEqual({item.status for item in result.items}, {"complete", "failed"})


if __name__ == "__main__":
    unittest.main()
