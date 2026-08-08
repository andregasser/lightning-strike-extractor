from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from lightning_extractor.dataset_export import export_frame_handoff


class DatasetExportTests(unittest.TestCase):
    def test_exports_frames_with_source_provenance_without_training_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "storm.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 24))
            for index in range(5):
                writer.write(np.full((24, 32, 3), index * 20, dtype=np.uint8))
            writer.release()
            run = root / "videos" / "storm-source-analysis" / "results"
            run.mkdir(parents=True)
            run_root = run.parent
            (run_root / "run.json").write_text(json.dumps({"source_id": "source-a"}))
            (run_root / "source.json").write_text(
                json.dumps({"path": str(video), "video": {"fps": 10.0}})
            )
            (run / "candidates.json").write_text(
                json.dumps(
                    [
                        {
                            "rank": 1,
                            "event_id": "event-1",
                            "frame_number": 2,
                            "time": 0.2,
                            "geometry_score": 10.0,
                            "line_segments": 2,
                            "bright_area": 20.0,
                        }
                    ]
                )
            )
            manifest = export_frame_handoff(root, root / "frames", context_frames=1)
            self.assertEqual(manifest["frames"][1]["source_id"], "source-a")
            self.assertEqual(manifest["frames"][1]["frame_number"], 2)
            self.assertTrue((root / "frames" / manifest["frames"][1]["file_name"]).is_file())
            self.assertNotIn("categories", manifest)


if __name__ == "__main__":
    unittest.main()
