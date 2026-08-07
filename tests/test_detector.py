from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lightning_extractor.detector import (
    Detection,
    DetectionResult,
    load_model_manifest,
    write_detection_json,
)
from tools.model_development.preannotate_coco import preannotate
from tools.model_development.validate_coco import validate_coco_dataset


class DetectorRuntimeTests(unittest.TestCase):
    def test_manifest_fixes_the_runtime_model_configuration(self) -> None:
        manifest = load_model_manifest()
        self.assertEqual(manifest.class_name, "lightning_channel")
        self.assertEqual(manifest.backend, "grounding_dino")
        self.assertEqual(len(manifest.revision), 40)
        self.assertGreater(manifest.confidence_threshold, 0)

    def test_writes_model_identity_with_detection_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "frame.jpg"
            image.touch()
            output = root / "result.json"
            result = DetectionResult(
                image=image,
                width=1920,
                height=1080,
                model=load_model_manifest(),
                detections=(
                    Detection("lightning_channel", 0.75, (1.0, 2.0, 3.0, 4.0)),
                ),
            )
            write_detection_json(output, result)
            payload = json.loads(output.read_text())
            self.assertEqual(payload["model"]["class"], "lightning_channel")
            self.assertIn("version", payload["model"])
            self.assertIn("revision", payload["model"])
            self.assertNotIn("prompt", payload)
            self.assertEqual(payload["detections"][0]["box"], [1.0, 2.0, 3.0, 4.0])


class DinoDatasetDevelopmentTests(unittest.TestCase):
    def test_preannotations_include_source_and_unverified_score(self) -> None:
        class FakeDetector:
            manifest = load_model_manifest()

            def detect(self, image: Path) -> DetectionResult:
                return DetectionResult(
                    image=image,
                    width=100,
                    height=80,
                    model=self.manifest,
                    detections=(
                        Detection("lightning_channel", 0.6, (10.0, 5.0, 30.0, 65.0)),
                    ),
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            source = images / "storm-a"
            source.mkdir(parents=True)
            (source / "frame.jpg").touch()
            output = root / "annotations.json"

            document = preannotate(images, output, detector=FakeDetector())

            self.assertEqual(document["images"][0]["source_id"], "storm-a")
            annotation = document["annotations"][0]
            self.assertEqual(annotation["bbox"], [10.0, 5.0, 20.0, 60.0])
            self.assertEqual(annotation["segmentation"], [])
            self.assertFalse(annotation["attributes"]["verified"])
            self.assertEqual(annotation["attributes"]["proposal_score"], 0.6)

    def test_validates_positive_and_negative_coco_images(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "positive.jpg").touch()
            (root / "negative.jpg").touch()
            annotations = root / "annotations.json"
            annotations.write_text(
                json.dumps(
                    {
                        "images": [
                            {"id": 1, "file_name": "positive.jpg", "width": 100, "height": 80},
                            {"id": 2, "file_name": "negative.jpg", "width": 100, "height": 80},
                        ],
                        "categories": [{"id": 1, "name": "lightning_channel"}],
                        "annotations": [
                            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 5, 20, 60]}
                        ],
                    }
                )
            )
            self.assertEqual(
                validate_coco_dataset(annotations, root),
                {
                    "images": 2,
                    "annotations": 1,
                    "categories": 1,
                    "positive_images": 1,
                    "negative_images": 1,
                },
            )

    def test_rejects_box_outside_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "image.jpg").touch()
            annotations = root / "annotations.json"
            annotations.write_text(
                json.dumps(
                    {
                        "images": [{"id": 1, "file_name": "image.jpg", "width": 10, "height": 10}],
                        "categories": [{"id": 1, "name": "lightning_channel"}],
                        "annotations": [
                            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [9, 0, 2, 2]}
                        ],
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "outside image"):
                validate_coco_dataset(annotations, root)


if __name__ == "__main__":
    unittest.main()
