from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .training import _training_imports


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_onnx_release(
    checkpoint: Path,
    training_report: Path,
    evaluation_report: Path,
    output: Path,
    *,
    version: str,
    input_size: int = 640,
    opset: int = 17,
) -> dict[str, Any]:
    if output.exists():
        raise ValueError(f"Refusing to overwrite ONNX release: {output}")
    if input_size <= 0:
        raise ValueError("input_size must be positive")
    training = json.loads(training_report.read_text())
    evaluation = json.loads(evaluation_report.read_text())
    if evaluation.get("split") not in {"validation", "test"}:
        raise ValueError("A validation or test evaluation is required for release")
    torch, torchvision = _training_imports()
    try:
        import onnx
        import onnxruntime
    except ImportError as error:
        raise RuntimeError("ONNX dependencies are missing; run `uv sync --extra train`") from error

    base = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=None, weights_backbone=None, num_classes=2
    )
    base.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    base.eval()

    class ExportWrapper(torch.nn.Module):
        def __init__(self, model: Any) -> None:
            super().__init__()
            self.model = model

        def forward(self, images: Any) -> tuple[Any, Any, Any]:
            result = self.model([images[0]])[0]
            return result["boxes"], result["scores"], result["labels"] - 1

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staged = Path(temporary) / output.name
        staged.mkdir()
        artifact = staged / "model.onnx"
        sample = torch.zeros((1, 3, input_size, input_size), dtype=torch.float32)
        wrapper = ExportWrapper(base)
        with torch.no_grad():
            reference = wrapper(sample)
        torch.onnx.export(
            wrapper,
            sample,
            artifact,
            input_names=["images"],
            output_names=["boxes", "scores", "class_ids"],
            dynamic_axes={"boxes": {0: "detections"}, "scores": {0: "detections"}, "class_ids": {0: "detections"}},
            opset_version=opset,
        )
        onnx.checker.check_model(onnx.load(artifact))
        session = onnxruntime.InferenceSession(str(artifact), providers=["CPUExecutionProvider"])
        actual = session.run(None, {"images": sample.numpy()})
        if len(actual) != 3:
            raise RuntimeError("Exported ONNX model does not expose the detector contract")
        for expected, observed in zip(reference, actual):
            if expected.shape != observed.shape or not np.allclose(expected.numpy(), observed, rtol=1e-3, atol=1e-4):
                raise RuntimeError("ONNX parity check failed")
        manifest = {
            "schema_version": 1,
            "name": "lightning-channel-detector",
            "version": version,
            "release_status": "candidate",
            "backend": "onnx",
            "artifact": "model.onnx",
            "artifact_sha256": _sha256(artifact),
            "classes": ["lightning_channel"],
            "input_name": "images",
            "preprocessing": {"width": input_size, "height": input_size, "color_order": "RGB", "layout": "NCHW", "scale": 1 / 255, "mean": [0, 0, 0], "std": [1, 1, 1], "resize_mode": "letterbox"},
            "outputs": {"boxes": "boxes", "scores": "scores", "class_ids": "class_ids", "box_format": "xyxy"},
            "confidence_threshold": evaluation["score_threshold"],
            "nms_threshold": 0.5,
            "onnx_opset": opset,
            "minimum_onnxruntime_version": "1.20",
            "dataset_release": training["dataset_release"],
            "evaluation": evaluation,
        }
        (staged / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (staged / "checksums.json").write_text(json.dumps({"model.onnx": manifest["artifact_sha256"]}, indent=2) + "\n")
        os.replace(staged, output)
    return manifest
