from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from lightning_extractor.config import Config
from lightning_extractor.models import CandidateFrame
from lightning_extractor.review import review_candidates


class ReviewTests(unittest.TestCase):
    def create_run(self, root: Path) -> Path:
        video = root / "storm.avi"
        writer = cv2.VideoWriter(
            str(video),
            cv2.VideoWriter_fourcc(*"MJPG"),
            10.0,
            (320, 180),
        )
        for index in range(12):
            frame = np.full((180, 320, 3), index * 10, dtype=np.uint8)
            cv2.line(frame, (30, 150), (280, 20), (255, 255, 255), 2)
            writer.write(frame)
        writer.release()
        run = root / "videos" / "storm-source-analysis"
        results = run / "results"
        results.mkdir(parents=True)
        (run / "source.json").write_text(
            json.dumps({"path": str(video), "name": video.name})
        )
        candidates = []
        for rank, event, frame_number in ((1, "evt_000001", 4), (2, "evt_000002", 7)):
            candidates.append(
                CandidateFrame(
                    rank,
                    event,
                    frame_number,
                    frame_number / 10.0,
                    600.0,
                    8,
                    1000.0,
                    channel_length=350.0,
                    channel_strength=30.0,
                    channel_thickness=2.0,
                    frame_quality=1000.0 - rank,
                ).as_dict()
            )
        (results / "candidates.json").write_text(json.dumps(candidates))
        return run

    def test_review_saves_labels_and_resumes_pending_item(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.create_run(root)
            answers = iter(("y", "q"))

            labels_path, counts = review_candidates(
                root,
                Config(),
                open_previews=False,
                input_func=lambda _: next(answers),
            )

            self.assertEqual(counts["lightning"], 1)
            self.assertEqual(counts["pending"], 1)
            document = json.loads(labels_path.read_text())
            self.assertEqual(len(document["items"]), 1)
            preview = Path(next(iter(document["items"].values()))["preview"])
            image = cv2.imread(str(preview))
            self.assertIsNotNone(image)
            assert image is not None
            self.assertEqual(image.shape[1], 5 * 640)

            labels_path, counts = review_candidates(
                root,
                Config(),
                open_previews=False,
                input_func=lambda _: "n",
            )

            self.assertEqual(counts["lightning"], 1)
            self.assertEqual(counts["not-lightning"], 1)
            self.assertEqual(counts["pending"], 0)
            self.assertEqual(len(json.loads(labels_path.read_text())["items"]), 2)

    def test_custom_labels_path_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = self.create_run(root)
            labels = root / "private" / "labels.json"

            result, counts = review_candidates(
                run,
                Config(),
                labels_path=labels,
                open_previews=False,
                input_func=lambda _: "u",
            )

            self.assertEqual(result, labels)
            self.assertEqual(counts["uncertain"], 2)
            self.assertEqual(counts["pending"], 0)


if __name__ == "__main__":
    unittest.main()
