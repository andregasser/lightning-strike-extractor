from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from lse.detector import (
    Detection,
    DetectionResult,
    LightningDetector,
    load_model_manifest,
    write_detection_json,
)


class DetectorRuntimeTests(unittest.TestCase):
    def test_manifest_fixes_the_runtime_model_configuration(self) -> None:
        manifest = load_model_manifest()
        self.assertEqual(manifest.class_name, "lightning_channel")
        self.assertEqual(manifest.backend, "onnx")
        self.assertEqual(manifest.preprocessing.layout, "NCHW")
        self.assertEqual(manifest.outputs.box_format, "xyxy")
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
            self.assertEqual(payload["model"]["classes"], ["lightning_channel"])
            self.assertIn("version", payload["model"])
            self.assertEqual(payload["model"]["backend"], "onnx")
            self.assertNotIn("prompt", payload)
            self.assertEqual(payload["detections"][0]["box"], [1.0, 2.0, 3.0, 4.0])

    def test_onnx_runtime_restores_letterboxed_boxes(self) -> None:
        class Output:
            def __init__(self, name: str) -> None:
                self.name = name

        class Session:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                pass

            def get_outputs(self) -> list[Output]:
                return [Output("boxes"), Output("scores"), Output("class_ids")]

            def get_inputs(self) -> list[Output]:
                return [Output("images")]

            def run(self, _outputs: object, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
                self.input = inputs["images"]
                return [
                    np.array([[64, 160, 576, 480]], dtype=np.float32),
                    np.array([0.9], dtype=np.float32),
                    np.array([0], dtype=np.int64),
                ]

        class Runtime:
            InferenceSession = Session

            @staticmethod
            def get_available_providers() -> list[str]:
                return ["CPUExecutionProvider"]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.onnx"
            model.touch()
            image = root / "frame.jpg"
            cv2.imwrite(str(image), np.zeros((100, 200, 3), dtype=np.uint8))
            with patch("lse.detector._runtime_import", return_value=Runtime):
                detector = LightningDetector(model_path=model)
                result = detector.detect(image)
            self.assertEqual(detector.session.input.shape, (1, 3, 640, 640))
            self.assertEqual(len(result.detections), 1)
            x0, y0, x1, y1 = result.detections[0].box
            self.assertAlmostEqual(x0, 20.0)
            self.assertAlmostEqual(y0, 0.0)
            self.assertAlmostEqual(x1, 180.0)
            self.assertAlmostEqual(y1, 100.0)

if __name__ == "__main__":
    unittest.main()
