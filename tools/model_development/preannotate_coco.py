from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import cv2


class ProposalDetector(Protocol):
    manifest: Any

    def detect(self, image: Path) -> Any: ...


@dataclass(frozen=True)
class BlankManifest:
    version: str = "none"

    def public_metadata(self) -> dict[str, object]:
        return {"name": "manual-annotation", "version": self.version, "backend": "none"}


@dataclass(frozen=True)
class BlankResult:
    width: int
    height: int
    detections: tuple[()] = ()


class BlankProposalDetector:
    """Create empty tasks for annotation without a product or bootstrap model."""

    manifest = BlankManifest()

    def detect(self, image: Path) -> BlankResult:
        pixels = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if pixels is None:
            raise ValueError(f"Image cannot be decoded: {image}")
        height, width = pixels.shape[:2]
        return BlankResult(width, height)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def discover_images(root: Path) -> list[Path]:
    if not root.is_dir():
        raise ValueError(f"Image root does not exist: {root}")
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
    )


def source_id(root: Path, image: Path) -> str:
    relative = image.relative_to(root)
    return relative.parts[0] if len(relative.parts) > 1 else image.stem


def preannotate(root: Path, output: Path, *, detector: ProposalDetector | None = None) -> dict:
    if output.exists():
        raise ValueError(f"Refusing to overwrite existing annotations: {output}")
    images = discover_images(root)
    if not images:
        raise ValueError(f"No supported images found below: {root}")
    session = detector or BlankProposalDetector()
    coco_images: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    annotation_id = 1
    for image_id, image_path in enumerate(images, 1):
        result = session.detect(image_path)
        coco_images.append(
            {
                "id": image_id,
                "file_name": image_path.relative_to(root).as_posix(),
                "width": result.width,
                "height": result.height,
                "source_id": source_id(root, image_path),
            }
        )
        for detection in result.detections:
            x0, y0, x1, y1 = detection.box
            width, height = max(0.0, x1 - x0), max(0.0, y1 - y0)
            if width == 0 or height == 0:
                continue
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x0, y0, width, height],
                    "area": width * height,
                    "segmentation": [],
                    "iscrowd": 0,
                    "attributes": {
                        "proposal_score": detection.score,
                        "proposal_model_version": result.model.version,
                        "verified": False,
                    },
                }
            )
            annotation_id += 1
    document = {
        "info": {
            "description": "Unverified lightning-channel proposals",
            "created_at": datetime.now(UTC).isoformat(),
            "proposal_model": session.manifest.public_metadata(),
        },
        "images": coco_images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "lightning_channel", "supercategory": "weather"}],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create unverified COCO box proposals")
    parser.add_argument("images", type=Path, help="Image root; first subdirectory identifies source")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    document = preannotate(args.images, args.output)
    print(
        f"images: {len(document['images'])}\n"
        f"proposals: {len(document['annotations'])}\n"
        f"output: {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
