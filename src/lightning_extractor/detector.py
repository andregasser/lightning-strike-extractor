from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelManifest:
    name: str
    version: str
    release_status: str
    backend: str
    artifact: str
    revision: str
    artifact_sha256: str | None
    class_name: str
    prompt: str
    confidence_threshold: float
    text_threshold: float

    def public_metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "release_status": self.release_status,
            "backend": self.backend,
            "revision": self.revision,
            "artifact_sha256": self.artifact_sha256,
            "class": self.class_name,
            "confidence_threshold": self.confidence_threshold,
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


def load_model_manifest() -> ModelManifest:
    resource = files("lightning_extractor").joinpath("model_manifest.json")
    data = json.loads(resource.read_text())
    manifest = ModelManifest(**data)
    if manifest.class_name != "lightning_channel":
        raise RuntimeError("Detector manifest must expose the lightning_channel class")
    if not 0 <= manifest.confidence_threshold <= 1 or not 0 <= manifest.text_threshold <= 1:
        raise RuntimeError("Detector manifest contains invalid thresholds")
    if manifest.backend != "grounding_dino":
        raise RuntimeError(f"Unsupported detector backend: {manifest.backend}")
    if manifest.release_status == "production" and not manifest.artifact_sha256:
        raise RuntimeError("Production detector manifests require an artifact SHA-256")
    return manifest


def _detector_imports() -> tuple[Any, Any, Any]:
    try:
        import torch
        from PIL import Image
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    except ImportError as error:
        raise RuntimeError(
            "Detector runtime is not installed; run `uv sync --extra detector`"
        ) from error
    return torch, Image, (AutoModelForZeroShotObjectDetection, AutoProcessor)


def _choose_device(torch: Any) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LightningDetector:
    """Reusable inference session for the single released detector manifest."""

    def __init__(self) -> None:
        self.manifest = load_model_manifest()
        self.torch, self.Image, classes = _detector_imports()
        model_class, processor_class = classes
        self.device = _choose_device(self.torch)
        self.processor = processor_class.from_pretrained(
            self.manifest.artifact,
            revision=self.manifest.revision,
        )
        self.model = model_class.from_pretrained(
            self.manifest.artifact,
            revision=self.manifest.revision,
        ).to(self.device)
        self.model.eval()

    def detect(self, image_path: Path) -> DetectionResult:
        if not image_path.is_file():
            raise ValueError(f"Image does not exist: {image_path}")
        image = self.Image.open(image_path).convert("RGB")
        inputs = self.processor(
            images=image,
            text=self.manifest.prompt,
            return_tensors="pt",
        ).to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        processed = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.manifest.confidence_threshold,
            text_threshold=self.manifest.text_threshold,
            target_sizes=[(image.height, image.width)],
        )[0]
        detections = tuple(
            sorted(
                (
                    Detection(
                        label=self.manifest.class_name,
                        score=float(score.item()),
                        box=tuple(float(value) for value in box.tolist()),
                    )
                    for box, score in zip(processed["boxes"], processed["scores"])
                ),
                key=lambda item: item.score,
                reverse=True,
            )
        )
        return DetectionResult(
            image=image_path,
            width=image.width,
            height=image.height,
            model=self.manifest,
            detections=detections,
        )


def detect_image(image_path: Path) -> DetectionResult:
    return LightningDetector().detect(image_path)


def write_detection_json(path: Path, result: DetectionResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.as_dict(), indent=2, ensure_ascii=False) + "\n")


def render_detections(image_path: Path, output_path: Path, detections: tuple[Detection, ...]) -> None:
    _, Image, _ = _detector_imports()
    from PIL import ImageDraw

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    line_width = max(2, round(min(image.size) / 300))
    for detection in detections:
        draw.rectangle(detection.box, outline=(255, 210, 0), width=line_width)
        x0, y0, _, _ = detection.box
        draw.text((x0 + line_width, max(0, y0 - 14)), f"{detection.score:.3f}", fill=(255, 210, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
