from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from .coco import CATEGORY, load_verified_coco

SPLITS = ("train", "validation", "test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_boxes(boxes: list[list[float]]) -> tuple[tuple[float, ...], ...]:
    return tuple(sorted(tuple(round(float(value), 6) for value in box) for box in boxes))


def build_release(campaigns: list[Path], output: Path, *, release_id: str) -> dict[str, Any]:
    if not campaigns:
        raise ValueError("At least one verified campaign is required")
    if not release_id or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789.-_" for character in release_id):
        raise ValueError("release_id must contain only lowercase letters, digits, dot, dash, or underscore")
    output = output.resolve()
    if output.exists():
        raise ValueError(f"Refusing to overwrite dataset release: {output}")

    records: dict[str, dict[str, Any]] = {}
    campaign_metadata: list[dict[str, Any]] = []
    source_splits: dict[str, str] = {}
    for campaign_path in campaigns:
        campaign = campaign_path.resolve()
        manifest_path = campaign / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"Campaign has no manifest: {campaign}")
        campaign_manifest = json.loads(manifest_path.read_text())
        campaign_metadata.append(
            {
                "campaign": campaign.name,
                "manifest_sha256": sha256(manifest_path),
                "declared_schema_version": campaign_manifest.get("schema_version"),
            }
        )
        for split in SPLITS:
            annotations_path = campaign / "annotations" / f"instances_{split}.json"
            images_root = campaign / "images" / split
            document = load_verified_coco(annotations_path, images_root)
            boxes_by_image: dict[int, list[list[float]]] = defaultdict(list)
            for annotation in document["annotations"]:
                boxes_by_image[annotation["image_id"]].append(annotation["bbox"])
            for image in document["images"]:
                source_id = image["source_id"]
                previous_split = source_splits.setdefault(source_id, split)
                if previous_split != split:
                    raise ValueError(f"Source {source_id} appears in both {previous_split} and {split}")
                image_path = images_root / image["file_name"]
                image_hash = sha256(image_path)
                boxes = _canonical_boxes(boxes_by_image[image["id"]])
                existing = records.get(image_hash)
                if existing and existing["boxes"] != boxes:
                    raise ValueError(f"Conflicting annotations for image SHA-256 {image_hash}")
                if existing and existing["split"] != split:
                    raise ValueError(f"Identical image appears in both {existing['split']} and {split}")
                if not existing:
                    records[image_hash] = {
                        "source": image_path,
                        "source_id": source_id,
                        "split": split,
                        "width": image["width"],
                        "height": image["height"],
                        "boxes": boxes,
                    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staged = Path(temporary) / output.name
        split_summary: dict[str, dict[str, int]] = {}
        output_files: list[dict[str, str]] = []
        for split in SPLITS:
            selected = sorted(
                ((digest, record) for digest, record in records.items() if record["split"] == split),
                key=lambda item: item[0],
            )
            images: list[dict[str, Any]] = []
            annotations: list[dict[str, Any]] = []
            image_dir = staged / "images" / split
            image_dir.mkdir(parents=True, exist_ok=True)
            for image_id, (digest, record) in enumerate(selected, 1):
                suffix = record["source"].suffix.casefold() or ".jpg"
                filename = f"{digest}{suffix}"
                destination = image_dir / filename
                shutil.copyfile(record["source"], destination)
                images.append(
                    {
                        "id": image_id,
                        "file_name": filename,
                        "width": record["width"],
                        "height": record["height"],
                        "source_id": record["source_id"],
                        "sha256": digest,
                    }
                )
                output_files.append({"path": destination.relative_to(staged).as_posix(), "sha256": digest})
                for box in record["boxes"]:
                    annotations.append(
                        {
                            "id": len(annotations) + 1,
                            "image_id": image_id,
                            "category_id": 1,
                            "bbox": list(box),
                            "area": box[2] * box[3],
                            "segmentation": [],
                            "iscrowd": 0,
                            "attributes": {"verified": True},
                        }
                    )
            document = {"info": {"dataset_release": release_id, "split": split}, "images": images, "annotations": annotations, "categories": [CATEGORY]}
            annotation_path = staged / "annotations" / f"instances_{split}.json"
            annotation_path.parent.mkdir(parents=True, exist_ok=True)
            annotation_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
            output_files.append({"path": annotation_path.relative_to(staged).as_posix(), "sha256": sha256(annotation_path)})
            split_summary[split] = {"images": len(images), "annotations": len(annotations), "sources": len({item["source_id"] for item in images})}
        manifest = {
            "schema_version": 1,
            "release_id": release_id,
            "category_schema": [CATEGORY],
            "campaigns": campaign_metadata,
            "source_assignments": dict(sorted(source_splits.items())),
            "splits": split_summary,
            "files": sorted(output_files, key=lambda item: item["path"]),
        }
        (staged / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(staged, output)
    return manifest
