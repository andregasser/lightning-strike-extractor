from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import cv2
import numpy as np

from tools.model_development.import_cvat_dataset import import_cvat_dataset


class CvatImportTests(unittest.TestCase):
    def create_archive(self, root: Path, *, corrupt_image: str | None = None) -> Path:
        images = [
            {"id": 1, "file_name": "source-a__one.jpg", "width": 10, "height": 10},
            {"id": 2, "file_name": "source-a__two.jpg", "width": 10, "height": 10},
            {"id": 3, "file_name": "source-b__one.jpg", "width": 10, "height": 10},
            {"id": 4, "file_name": "source-c__one.jpg", "width": 10, "height": 10},
        ]
        annotations = [
            {
                "id": index,
                "image_id": index,
                "category_id": 1,
                "bbox": [1, 1, 5, 5],
                "area": 25,
                "segmentation": [],
                "iscrowd": 0,
                "attributes": {"verified": False},
            }
            for index in range(1, 5)
        ]
        document = {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": 1, "name": "lightning_channel"}],
        }
        archive_path = root / "corrected.zip"
        ok, encoded = cv2.imencode(".jpg", np.zeros((10, 10, 3), dtype=np.uint8))
        self.assertTrue(ok)
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("annotations/instances_default.json", json.dumps(document))
            for image in images:
                payload = b"not-an-image" if image["file_name"] == corrupt_image else encoded.tobytes()
                archive.writestr(f"images/default/{image['file_name']}", payload)
        return archive_path

    def test_imports_verified_source_grouped_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = self.create_archive(root)
            output = root / "verified"

            manifest = import_cvat_dataset(archive, output)

            assignments = manifest["source_assignments"]
            self.assertEqual(set(assignments), {"source-a", "source-b", "source-c"})
            self.assertEqual(set(assignments.values()), {"train", "validation", "test"})
            seen_sources: set[str] = set()
            for split in ("train", "validation", "test"):
                document = json.loads(
                    (output / "annotations" / f"instances_{split}.json").read_text()
                )
                split_sources = {image["source_id"] for image in document["images"]}
                self.assertTrue(seen_sources.isdisjoint(split_sources))
                seen_sources.update(split_sources)
                self.assertTrue(
                    all(
                        annotation["attributes"]["verified"]
                        for annotation in document["annotations"]
                    )
                )

    def test_rejects_filename_without_source_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "invalid.zip"
            document = {
                "images": [{"id": 1, "file_name": "frame.jpg", "width": 10, "height": 10}],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 5, 5]}
                ],
                "categories": [{"id": 1, "name": "lightning_channel"}],
            }
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("annotations/instances_default.json", json.dumps(document))
                archive.writestr("images/default/frame.jpg", b"image")

            with self.assertRaisesRegex(ValueError, "source ID"):
                import_cvat_dataset(archive_path, root / "output")

    def test_rejects_invalid_ratios_without_publishing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = self.create_archive(root)
            output = root / "verified"

            with self.assertRaisesRegex(ValueError, "sum to 1.0"):
                import_cvat_dataset(
                    archive,
                    output,
                    train_ratio=0.8,
                    validation_ratio=0.3,
                    test_ratio=0.1,
                )

            self.assertFalse(output.exists())

    def test_rejects_corrupt_image_without_publishing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = self.create_archive(root, corrupt_image="source-a__one.jpg")
            output = root / "verified"

            with self.assertRaisesRegex(ValueError, "cannot be decoded"):
                import_cvat_dataset(archive, output)

            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
