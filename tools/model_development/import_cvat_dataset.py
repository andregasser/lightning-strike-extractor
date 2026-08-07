from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np

from tools.model_development.dataset_splits import (
    SPLITS,
    assign_sources_to_splits,
    validate_split_ratios,
)
from tools.model_development.validate_coco import validate_coco_dataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_id(image: dict[str, Any]) -> str:
    explicit = image.get("source_id")
    filename = str(image.get("file_name", ""))
    source, separator, _ = Path(filename).name.partition("__")
    if not separator or not source:
        raise ValueError(f"CVAT image filename does not preserve a source ID: {filename}")
    if explicit is not None and explicit != source:
        raise ValueError(f"CVAT image has conflicting source identities: {filename}")
    return source


def _validated_image(payload: bytes, image: dict[str, Any]) -> None:
    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError(f"CVAT image cannot be decoded: {image['file_name']}")
    height, width = decoded.shape[:2]
    if width != image.get("width") or height != image.get("height"):
        raise ValueError(f"CVAT image dimensions do not match COCO metadata: {image['file_name']}")


def _coco_member(archive: zipfile.ZipFile) -> tuple[str, str]:
    candidates = sorted(
        name
        for name in archive.namelist()
        if PurePosixPath(name).parts[:1] == ("annotations",)
        and PurePosixPath(name).name.startswith("instances_")
        and name.endswith(".json")
    )
    if len(candidates) != 1:
        raise ValueError("CVAT archive must contain exactly one instances_<subset>.json")
    member = candidates[0]
    subset = PurePosixPath(member).stem.removeprefix("instances_")
    if not subset:
        raise ValueError("CVAT annotation subset name cannot be empty")
    return member, subset


def import_cvat_dataset(
    archive_path: Path,
    output: Path,
    *,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.2,
    test_ratio: float = 0.1,
) -> dict[str, Any]:
    """Validate a corrected CVAT export and publish source-grouped COCO splits."""
    archive_path = archive_path.resolve()
    output = output.resolve()
    if not archive_path.is_file():
        raise ValueError(f"CVAT archive does not exist: {archive_path}")
    if output.exists():
        raise ValueError(f"Refusing to overwrite imported dataset: {output}")
    ratios = validate_split_ratios(train_ratio, validation_ratio, test_ratio)

    with zipfile.ZipFile(archive_path) as archive:
        members = archive.namelist()
        if len(members) != len(set(members)):
            raise ValueError("CVAT archive contains duplicate member names")
        annotation_member, subset = _coco_member(archive)
        document = json.loads(archive.read(annotation_member))
        if not isinstance(document, dict):
            raise TypeError("CVAT COCO annotations must be a JSON object")
        images = document.get("images")
        annotations = document.get("annotations")
        categories = document.get("categories")
        if not all(isinstance(value, list) for value in (images, annotations, categories)):
            raise ValueError("CVAT COCO file needs images, annotations, and categories lists")
        if not images:
            raise ValueError("CVAT dataset contains no images")
        if not annotations:
            raise ValueError("CVAT dataset contains no verified lightning annotations")
        if (
            len(categories) != 1
            or not isinstance(categories[0], dict)
            or categories[0].get("name") != "lightning_channel"
        ):
            raise ValueError("CVAT dataset must contain exactly the lightning_channel category")

        images_by_id: dict[int, dict[str, Any]] = {}
        filenames: set[str] = set()
        source_counts: dict[str, int] = defaultdict(int)
        archive_members: dict[int, str] = {}
        for image in images:
            if not isinstance(image, dict) or not isinstance(image.get("id"), int):
                raise TypeError("Every CVAT image needs an integer id")
            image_id = image["id"]
            if image_id in images_by_id:
                raise ValueError(f"Duplicate CVAT image id: {image_id}")
            filename = image.get("file_name")
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise ValueError(f"CVAT image file_name must be a plain filename: {filename}")
            if filename in filenames:
                raise ValueError(f"Duplicate CVAT image filename: {filename}")
            filenames.add(filename)
            member = f"images/{subset}/{filename}"
            if member not in members:
                raise ValueError(f"CVAT archive is missing image: {member}")
            source = _source_id(image)
            image["source_id"] = source
            images_by_id[image_id] = image
            archive_members[image_id] = member
            source_counts[source] += 1

        assignments = assign_sources_to_splits(dict(source_counts), ratios)
        annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in annotations:
            if not isinstance(annotation, dict):
                raise TypeError("Every CVAT annotation must be an object")
            image_id = annotation.get("image_id")
            if image_id not in images_by_id:
                raise ValueError(f"Annotation references unknown image id: {image_id}")
            attributes = annotation.setdefault("attributes", {})
            if not isinstance(attributes, dict):
                raise TypeError("CVAT annotation attributes must be an object")
            attributes["verified"] = True
            annotations_by_image[image_id].append(annotation)

        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
            staged = Path(temporary) / output.name
            split_summaries: dict[str, dict[str, int]] = {}
            for split in SPLITS:
                split_images = [
                    image
                    for image in images
                    if assignments[image["source_id"]] == split
                ]
                image_dir = staged / "images" / split
                image_dir.mkdir(parents=True, exist_ok=True)
                remapped_images: list[dict[str, Any]] = []
                remapped_annotations: list[dict[str, Any]] = []
                for new_image_id, image in enumerate(split_images, 1):
                    original_id = image["id"]
                    filename = str(image["file_name"])
                    payload = archive.read(archive_members[original_id])
                    _validated_image(payload, image)
                    (image_dir / filename).write_bytes(payload)
                    remapped = {**image, "id": new_image_id}
                    remapped_images.append(remapped)
                    for annotation in annotations_by_image[original_id]:
                        remapped_annotations.append(
                            {
                                **annotation,
                                "id": len(remapped_annotations) + 1,
                                "image_id": new_image_id,
                            }
                        )
                split_document = {
                    "info": {
                        **(document.get("info") if isinstance(document.get("info"), dict) else {}),
                        "description": f"Verified lightning channels: {split}",
                    },
                    "images": remapped_images,
                    "annotations": remapped_annotations,
                    "categories": categories,
                }
                annotation_path = staged / "annotations" / f"instances_{split}.json"
                annotation_path.parent.mkdir(parents=True, exist_ok=True)
                annotation_path.write_text(
                    json.dumps(split_document, indent=2, ensure_ascii=False) + "\n"
                )
                split_summaries[split] = validate_coco_dataset(annotation_path, image_dir)

            manifest = {
                "schema_version": 1,
                "created_at": datetime.now(UTC).isoformat(),
                "source_archive": str(archive_path),
                "source_archive_sha256": _sha256(archive_path),
                "ratios": ratios,
                "source_assignments": assignments,
                "splits": split_summaries,
            }
            (staged / "manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
            )
            os.replace(staged, output)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Import a corrected CVAT COCO export and create source-grouped splits"
    )
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    args = parser.parse_args(argv)
    manifest = import_cvat_dataset(
        args.archive,
        args.output,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"output: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
