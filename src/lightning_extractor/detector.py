from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class Preprocessing:
    width: int
    height: int
    color_order: str
    layout: str
    scale: float
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    resize_mode: str


@dataclass(frozen=True, slots=True)
class Outputs:
    boxes: str
    scores: str
    class_ids: str
    box_format: str


@dataclass(frozen=True, slots=True)
class ModelManifest:
    schema_version: int
    name: str
    version: str
    release_status: str
    backend: str
    artifact: str
    artifact_sha256: str | None
    classes: tuple[str, ...]
    input_name: str
    preprocessing: Preprocessing
    outputs: Outputs
    confidence_threshold: float
    nms_threshold: float
    onnx_opset: int
    minimum_onnxruntime_version: str
    dataset_release: str | None

    @property
    def class_name(self) -> str:
        return self.classes[0]

    def public_metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "release_status": self.release_status,
            "backend": self.backend,
            "artifact_sha256": self.artifact_sha256,
            "classes": list(self.classes),
            "confidence_threshold": self.confidence_threshold,
            "dataset_release": self.dataset_release,
        }


@dataclass(frozen=True, slots=True)
class Detection:
    label: str
    score: float
    box: tuple[float, float, float, float]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DetectionResult:
    image: Path
    width: int
    height: int
    model: ModelManifest
    detections: tuple[Detection, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "image": str(self.image.resolve()),
            "width": self.width,
            "height": self.height,
            "model": self.model.public_metadata(),
            "detections": [item.as_dict() for item in self.detections],
        }


def load_model_manifest(path: Path | None = None) -> ModelManifest:
    data = json.loads(
        path.read_text()
        if path is not None
        else files("lightning_extractor").joinpath("model_manifest.json").read_text()
    )
    preprocessing_data = data.pop("preprocessing")
    preprocessing = Preprocessing(
        **{
            **preprocessing_data,
            "mean": tuple(preprocessing_data["mean"]),
            "std": tuple(preprocessing_data["std"]),
        }
    )
    outputs = Outputs(**data.pop("outputs"))
    classes = tuple(data.pop("classes"))
    manifest = ModelManifest(
        **data,
        classes=classes,
        preprocessing=preprocessing,
        outputs=outputs,
    )
    if manifest.schema_version != 1 or manifest.backend != "onnx":
        raise RuntimeError("Detector manifest must use ONNX contract schema 1")
    if manifest.classes != ("lightning_channel",):
        raise RuntimeError("Detector must expose only the lightning_channel class")
    if manifest.preprocessing.layout != "NCHW" or manifest.preprocessing.color_order != "RGB":
        raise RuntimeError("Only RGB NCHW detector inputs are supported")
    if manifest.preprocessing.resize_mode != "letterbox":
        raise RuntimeError("Only letterbox detector resizing is supported")
    if manifest.preprocessing.width <= 0 or manifest.preprocessing.height <= 0:
        raise RuntimeError("Detector input dimensions must be positive")
    if manifest.preprocessing.scale <= 0 or any(value == 0 for value in manifest.preprocessing.std):
        raise RuntimeError("Detector preprocessing scale and standard deviations must be non-zero")
    if manifest.outputs.box_format != "xyxy":
        raise RuntimeError("Only xyxy detector boxes are supported")
    if not 0 <= manifest.confidence_threshold <= 1 or not 0 <= manifest.nms_threshold <= 1:
        raise RuntimeError("Detector manifest contains invalid thresholds")
    if manifest.release_status == "production" and not manifest.artifact_sha256:
        raise RuntimeError("Production detector manifests require an artifact SHA-256")
    return manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_import() -> Any:
    try:
        import onnxruntime
    except ImportError as error:
        raise RuntimeError("Detector runtime is not installed; run `uv sync --extra detector`") from error
    return onnxruntime


def _letterbox(image: np.ndarray, width: int, height: int) -> tuple[np.ndarray, float, int, int]:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized_width = max(1, round(image.shape[1] * scale))
    resized_height = max(1, round(image.shape[0] * scale))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    left = (width - resized_width) // 2
    top = (height - resized_height) // 2
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas, scale, left, top


class LightningDetector:
    """Inference-only ONNX session for the released closed-set detector."""

    def __init__(self, model_path: Path | None = None, manifest_path: Path | None = None) -> None:
        self.manifest = load_model_manifest(manifest_path)
        if model_path is None:
            resource = files("lightning_extractor").joinpath(self.manifest.artifact)
            context = as_file(resource)
            self._artifact_context = context
            model_path = context.__enter__()
        else:
            self._artifact_context = None
        self.model_path = model_path.resolve()
        if not self.model_path.is_file():
            raise RuntimeError(f"Released ONNX model is not installed: {self.model_path}")
        if self.manifest.artifact_sha256 and _sha256(self.model_path) != self.manifest.artifact_sha256:
            raise RuntimeError("Released ONNX model checksum does not match its manifest")
        runtime = _runtime_import()
        self.session = runtime.InferenceSession(
            str(self.model_path), providers=runtime.get_available_providers()
        )
        input_names = {item.name for item in self.session.get_inputs()}
        if self.manifest.input_name not in input_names:
            raise RuntimeError(f"ONNX input does not match manifest: {self.manifest.input_name}")
        names = {item.name for item in self.session.get_outputs()}
        expected = {
            self.manifest.outputs.boxes,
            self.manifest.outputs.scores,
            self.manifest.outputs.class_ids,
        }
        if not expected <= names:
            raise RuntimeError(f"ONNX outputs do not match manifest: missing {sorted(expected - names)}")

    def detect(self, image_path: Path) -> DetectionResult:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Image cannot be decoded: {image_path}")
        height, width = image.shape[:2]
        settings = self.manifest.preprocessing
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        prepared, scale, left, top = _letterbox(rgb, settings.width, settings.height)
        tensor = prepared.astype(np.float32) * settings.scale
        tensor = (tensor - np.asarray(settings.mean, dtype=np.float32)) / np.asarray(
            settings.std, dtype=np.float32
        )
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        values = self.session.run(None, {self.manifest.input_name: tensor})
        by_name = dict(zip((item.name for item in self.session.get_outputs()), values))
        boxes = np.asarray(by_name[self.manifest.outputs.boxes]).reshape(-1, 4)
        scores = np.asarray(by_name[self.manifest.outputs.scores]).reshape(-1)
        class_ids = np.asarray(by_name[self.manifest.outputs.class_ids]).reshape(-1)
        if not len(boxes) == len(scores) == len(class_ids):
            raise RuntimeError("ONNX detector returned inconsistent output lengths")
        detections: list[Detection] = []
        for box, score, class_id in zip(boxes, scores, class_ids):
            if float(score) < self.manifest.confidence_threshold:
                continue
            index = int(class_id)
            if not 0 <= index < len(self.manifest.classes):
                raise RuntimeError(f"ONNX detector returned invalid class id: {index}")
            x0, y0, x1, y1 = (float(value) for value in box)
            restored = (
                min(width, max(0.0, (x0 - left) / scale)),
                min(height, max(0.0, (y0 - top) / scale)),
                min(width, max(0.0, (x1 - left) / scale)),
                min(height, max(0.0, (y1 - top) / scale)),
            )
            if restored[2] > restored[0] and restored[3] > restored[1]:
                detections.append(Detection(self.manifest.classes[index], float(score), restored))
        detections.sort(key=lambda item: item.score, reverse=True)
        return DetectionResult(image_path, width, height, self.manifest, tuple(detections))


def detect_image(image_path: Path) -> DetectionResult:
    return LightningDetector().detect(image_path)


def write_detection_json(path: Path, result: DetectionResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.as_dict(), indent=2, ensure_ascii=False) + "\n")


def render_detections(image_path: Path, output_path: Path, detections: tuple[Detection, ...]) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Image cannot be decoded: {image_path}")
    line_width = max(2, round(min(image.shape[:2]) / 300))
    for detection in detections:
        x0, y0, x1, y1 = (round(value) for value in detection.box)
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 210, 255), line_width)
        cv2.putText(image, f"{detection.score:.3f}", (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 210, 255), 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Could not write detector preview: {output_path}")
