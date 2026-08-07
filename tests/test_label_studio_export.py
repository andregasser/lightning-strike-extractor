from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tools.model_development.export_label_studio import export_label_studio


class LabelStudioExportTests(unittest.TestCase):
    def create_dataset(self, root: Path) -> Path:
        dataset = root / "dataset"
        images = dataset / "images" / "source-a"
        images.mkdir(parents=True)
        cv2.imwrite(str(images / "frame.jpg"), np.zeros((50, 100, 3), dtype=np.uint8))
        annotations = dataset / "annotations"
        annotations.mkdir()
        (annotations / "proposals.json").write_text(
            json.dumps(
                {
                    "info": {"proposal_model": {"version": "bootstrap-test"}},
                    "images": [
                        {
                            "id": 1,
                            "file_name": "source-a/frame.jpg",
                            "source_id": "source-a",
                            "width": 100,
                            "height": 50,
                        }
                    ],
                    "annotations": [
                        {
                            "id": 1,
                            "image_id": 1,
                            "category_id": 1,
                            "bbox": [10, 5, 20, 10],
                            "attributes": {"proposal_score": 0.8},
                        }
                    ],
                    "categories": [{"id": 1, "name": "lightning_channel"}],
                }
            )
        )
        return dataset

    def test_exports_percentage_predictions_and_images(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = self.create_dataset(root)
            output = root / "label-studio"

            manifest = export_label_studio(dataset, output)

            self.assertEqual(manifest["tasks"], 1)
            tasks = json.loads((output / "import" / "tasks.json").read_text())
            self.assertEqual(
                tasks[0]["data"]["image"],
                "http://localhost:8001/images/source-a__frame.jpg",
            )
            value = tasks[0]["predictions"][0]["result"][0]["value"]
            self.assertEqual(
                {key: value[key] for key in ("x", "y", "width", "height")},
                {"x": 10.0, "y": 10.0, "width": 20.0, "height": 20.0},
            )
            self.assertTrue((output / "serve" / "images" / "source-a__frame.jpg").is_file())
            self.assertIn(
                "RectangleLabels", (output / "project" / "label-config.xml").read_text()
            )
            self.assertTrue((output / "export-info.txt").is_file())
            self.assertEqual([path.name for path in (output / "import").iterdir()], ["tasks.json"])

    def test_refuses_to_overwrite_export(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = self.create_dataset(root)
            output = root / "label-studio"
            output.mkdir()

            with self.assertRaisesRegex(ValueError, "overwrite"):
                export_label_studio(dataset, output)


if __name__ == "__main__":
    unittest.main()
