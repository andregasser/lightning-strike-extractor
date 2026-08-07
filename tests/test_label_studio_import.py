from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tools.model_development.import_label_studio_dataset import import_label_studio_dataset


class LabelStudioImportTests(unittest.TestCase):
    def create_export(self, root: Path) -> tuple[Path, Path]:
        images_root = root / "images"
        tasks = []
        specifications = [
            (1, "source-a", "one.jpg", True),
            (2, "source-a", "two.jpg", False),
            (3, "source-b", "one.jpg", True),
            (4, "source-c", "one.jpg", True),
        ]
        for task_id, source_id, filename, positive in specifications:
            source_dir = images_root / source_id
            source_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(source_dir / filename), np.zeros((50, 100, 3), dtype=np.uint8)
            )
            result = (
                [
                    {
                        "id": f"box-{task_id}",
                        "type": "rectanglelabels",
                        "from_name": "label",
                        "to_name": "image",
                        "original_width": 100,
                        "original_height": 50,
                        "value": {
                            "x": 10,
                            "y": 20,
                            "width": 30,
                            "height": 40,
                            "rotation": 0,
                            "rectanglelabels": ["lightning_channel"],
                        },
                    }
                ]
                if positive
                else []
            )
            tasks.append(
                {
                    "id": task_id,
                    "data": {
                        "image": f"http://localhost:8001/images/{source_id}__{filename}",
                        "source_id": source_id,
                        "original_file_name": f"{source_id}/{filename}",
                    },
                    "annotations": [
                        {
                            "id": 100 + task_id,
                            "was_cancelled": False,
                            "result": result,
                        }
                    ],
                    "predictions": [
                        {"result": [{"id": "ignored-model-proposal"}]}
                    ],
                }
            )
        export_path = root / "label-studio.json"
        export_path.write_text(json.dumps(tasks))
        return export_path, images_root

    def test_imports_human_annotations_and_preserves_negatives(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            export_path, images_root = self.create_export(root)
            output = root / "verified"

            manifest = import_label_studio_dataset(export_path, images_root, output)

            self.assertEqual(set(manifest["source_assignments"].values()), {"train", "validation", "test"})
            totals = {
                key: sum(split[key] for split in manifest["splits"].values())
                for key in ("images", "annotations", "positive_images", "negative_images")
            }
            self.assertEqual(
                totals,
                {"images": 4, "annotations": 3, "positive_images": 3, "negative_images": 1},
            )
            for split in ("train", "validation", "test"):
                document = json.loads(
                    (output / "annotations" / f"instances_{split}.json").read_text()
                )
                self.assertTrue(
                    all(item["attributes"]["verified"] for item in document["annotations"])
                )

    def test_rejects_task_without_completed_annotation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            export_path, images_root = self.create_export(root)
            tasks = json.loads(export_path.read_text())
            tasks[0]["annotations"] = []
            export_path.write_text(json.dumps(tasks))

            with self.assertRaisesRegex(ValueError, "exactly one completed annotation"):
                import_label_studio_dataset(export_path, images_root, root / "verified")


if __name__ == "__main__":
    unittest.main()
