from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from lightning_extractor.models import CandidateFrame
from tools.model_development.export_training_frames import export_training_frames


class TrainingDatasetExportTests(unittest.TestCase):
    def create_run(self, root: Path) -> Path:
        video = root / "storm.avi"
        writer = cv2.VideoWriter(
            str(video),
            cv2.VideoWriter_fourcc(*"MJPG"),
            10.0,
            (64, 48),
        )
        for index in range(12):
            writer.write(np.full((48, 64, 3), index * 10, dtype=np.uint8))
        writer.release()
        run = root / "videos" / "storm-source-analysis"
        results = run / "results"
        results.mkdir(parents=True)
        (run / "run.json").write_text(json.dumps({"source_id": "source123"}))
        (run / "source.json").write_text(
            json.dumps(
                {
                    "path": str(video),
                    "name": video.name,
                    "video": {"fps": 10.0},
                }
            )
        )
        candidates = [
            CandidateFrame(1, "evt_a", 4, 0.4, 100.0, 3, 20.0, frame_quality=100.0),
            CandidateFrame(2, "evt_a", 5, 0.5, 90.0, 3, 20.0, frame_quality=90.0),
            CandidateFrame(3, "evt_b", 8, 0.8, 80.0, 2, 30.0, frame_quality=80.0),
        ]
        (results / "candidates.json").write_text(
            json.dumps([candidate.as_dict() for candidate in candidates])
        )
        labels = root / "labels"
        labels.mkdir()
        (labels / "review.json").write_text(
            json.dumps(
                {
                    "items": {
                        "storm-source-analysis:evt_a": {
                            "frame_number": 4,
                            "label": "lightning",
                        }
                    }
                }
            )
        )
        return run

    def test_exports_unique_context_frames_with_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.create_run(root)
            output = root / "dataset"

            manifest = export_training_frames(
                root,
                output,
                max_events_per_video=2,
                context_frames=1,
            )

            self.assertEqual(len(manifest["sources"]), 1)
            self.assertEqual(len(manifest["frames"]), 6)
            peak = next(row for row in manifest["frames"] if row["frame_number"] == 4)
            self.assertEqual(peak["source_id"], "source123")
            self.assertEqual(len(peak["sha256"]), 64)
            self.assertEqual(peak["roles"][0]["review_label"], "lightning")
            self.assertTrue((output / peak["file_name"]).is_file())
            self.assertTrue((output / "manifest.json").is_file())

    def test_refuses_to_overwrite_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.create_run(root)
            output = root / "dataset"
            output.mkdir()

            with self.assertRaisesRegex(ValueError, "overwrite"):
                export_training_frames(root, output)

    def test_deduplicates_the_same_source_frame_across_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = self.create_run(root)
            shutil.copytree(run, run.with_name("storm-source-second-analysis"))

            manifest = export_training_frames(
                root,
                root / "dataset",
                max_events_per_video=2,
                context_frames=1,
            )

            self.assertEqual(len(manifest["frames"]), 6)
            self.assertEqual(manifest["sources"][1]["duplicate_frames_skipped"], 6)

    def test_rejects_empty_candidate_collection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = self.create_run(root)
            (run / "results" / "candidates.json").write_text("[]")

            with self.assertRaisesRegex(ValueError, "No candidate frames"):
                export_training_frames(root, root / "dataset")

    def test_skips_context_beyond_end_of_video(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = self.create_run(root)
            candidate = CandidateFrame(
                1,
                "evt_end",
                11,
                1.1,
                100.0,
                3,
                20.0,
                frame_quality=100.0,
            )
            (run / "results" / "candidates.json").write_text(
                json.dumps([candidate.as_dict()])
            )

            manifest = export_training_frames(
                root,
                root / "dataset",
                context_frames=1,
            )

            self.assertEqual(len(manifest["frames"]), 2)
            self.assertEqual(manifest["sources"][0]["out_of_range_frames_skipped"], 1)


if __name__ == "__main__":
    unittest.main()
