from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import cv2

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


def _completed_annotation(task: dict[str, Any]) -> dict[str, Any]:
    annotations = task.get("annotations")
    if not isinstance(annotations, list):
        raise TypeError(f"Label Studio task {task.get('id')} has no annotations list")
    completed = [
        annotation
        for annotation in annotations
        if isinstance(annotation, dict) and not annotation.get("was_cancelled", False)
    ]
    if len(completed) != 1:
        raise ValueError(
            f"Label Studio task {task.get('id')} must have exactly one completed annotation"
        )
    return completed[0]


def _box_from_result(result: dict[str, Any], width: int, height: int) -> list[float]:
    if result.get("type") != "rectanglelabels":
        raise ValueError(f"Unsupported Label Studio result type: {result.get('type')}")
    if result.get("from_name") != "label" or result.get("to_name") != "image":
        raise ValueError("Label Studio rectangle does not match label-config.xml")
    if result.get("original_width") != width or result.get("original_height") != height:
        raise ValueError("Label Studio rectangle dimensions do not match the source image")
    value = result.get("value")
    if not isinstance(value, dict):
        raise TypeError("Label Studio rectangle value must be an object")
    if value.get("rectanglelabels") != ["lightning_channel"]:
        raise ValueError("Label Studio rectangle must use the lightning_channel label")
    if float(value.get("rotation", 0)) != 0:
        raise ValueError("Rotated Label Studio rectangles are not supported")
    percentages = [value.get(key) for key in ("x", "y", "width", "height")]
    if not all(isinstance(item, (int, float)) for item in percentages):
        raise TypeError("Label Studio rectangle coordinates must be numeric")
    x, y, box_width, box_height = (float(item) for item in percentages)
    if x < 0 or y < 0 or box_width <= 0 or box_height <= 0:
        raise ValueError("Label Studio rectangle has invalid coordinates")
    if x + box_width > 100 + 1e-6 or y + box_height > 100 + 1e-6:
        raise ValueError("Label Studio rectangle lies outside the image")
    return [
        x / 100 * width,
        y / 100 * height,
        box_width / 100 * width,
        box_height / 100 * height,
    ]


def import_label_studio_dataset(
    export_path: Path,
    images_root: Path,
    output: Path,
    *,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.2,
    test_ratio: float = 0.1,
) -> dict[str, Any]:
    """Convert completed Label Studio tasks to verified, source-grouped COCO splits."""
    export_path = export_path.resolve()
    images_root = images_root.resolve()
    output = output.resolve()
    if not export_path.is_file():
        raise ValueError(f"Label Studio export does not exist: {export_path}")
    if not images_root.is_dir():
        raise ValueError(f"Source image root does not exist: {images_root}")
    if output.exists():
        raise ValueError(f"Refusing to overwrite imported dataset: {output}")
    ratios = validate_split_ratios(train_ratio, validation_ratio, test_ratio)
    tasks = json.loads(export_path.read_text())
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Label Studio export must be a non-empty task list")

    records: list[dict[str, Any]] = []
    source_counts: dict[str, int] = defaultdict(int)
    seen_task_ids: set[int] = set()
    positive_images = 0
    for task in tasks:
        if not isinstance(task, dict) or not isinstance(task.get("id"), int):
            raise TypeError("Every Label Studio task needs an integer id")
        task_id = task["id"]
        if task_id in seen_task_ids:
            raise ValueError(f"Duplicate Label Studio task id: {task_id}")
        seen_task_ids.add(task_id)
        data = task.get("data")
        if not isinstance(data, dict):
            raise TypeError(f"Label Studio task {task_id} data must be an object")
        source_id = data.get("source_id")
        original_name = data.get("original_file_name")
        image_url = data.get("image")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(f"Label Studio task {task_id} has no source_id")
        if not isinstance(original_name, str) or not original_name:
            raise ValueError(f"Label Studio task {task_id} has no original_file_name")
        if not isinstance(image_url, str) or not image_url:
            raise ValueError(f"Label Studio task {task_id} has no image URL")
        relative = Path(original_name)
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != source_id:
            raise ValueError(f"Label Studio task {task_id} has unsafe source image metadata")
        expected_served_name = f"{source_id}__{relative.name}"
        served_name = Path(unquote(urlparse(image_url).path)).name
        if served_name != expected_served_name:
            raise ValueError(f"Label Studio task {task_id} image URL does not match its source")
        image_path = images_root / relative
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Label Studio source image cannot be decoded: {image_path}")
        height, width = image.shape[:2]
        annotation = _completed_annotation(task)
        results = annotation.get("result")
        if not isinstance(results, list):
            raise TypeError(f"Label Studio task {task_id} result must be a list")
        boxes = [_box_from_result(result, width, height) for result in results]
        positive_images += bool(boxes)
        records.append(
            {
                "task_id": task_id,
                "annotation_id": annotation.get("id"),
                "source_id": source_id,
                "source_path": image_path,
                "file_name": expected_served_name,
                "width": width,
                "height": height,
                "boxes": boxes,
            }
        )
        source_counts[source_id] += 1
    if positive_images == 0:
        raise ValueError("Label Studio export contains no verified lightning annotations")

    file_names = [record["file_name"] for record in records]
    if len(file_names) != len(set(file_names)):
        raise ValueError("Label Studio export produces duplicate image filenames")
    assignments = assign_sources_to_splits(dict(source_counts), ratios)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staged = Path(temporary) / output.name
        split_summaries: dict[str, dict[str, int]] = {}
        for split in SPLITS:
            split_records = [
                record for record in records if assignments[record["source_id"]] == split
            ]
            image_dir = staged / "images" / split
            image_dir.mkdir(parents=True, exist_ok=True)
            coco_images: list[dict[str, Any]] = []
            coco_annotations: list[dict[str, Any]] = []
            for image_id, record in enumerate(split_records, 1):
                shutil.copy2(record["source_path"], image_dir / record["file_name"])
                coco_images.append(
                    {
                        "id": image_id,
                        "file_name": record["file_name"],
                        "width": record["width"],
                        "height": record["height"],
                        "source_id": record["source_id"],
                        "label_studio_task_id": record["task_id"],
                    }
                )
                for box in record["boxes"]:
                    coco_annotations.append(
                        {
                            "id": len(coco_annotations) + 1,
                            "image_id": image_id,
                            "category_id": 1,
                            "bbox": box,
                            "area": box[2] * box[3],
                            "segmentation": [],
                            "iscrowd": 0,
                            "attributes": {
                                "verified": True,
                                "label_studio_annotation_id": record["annotation_id"],
                            },
                        }
                    )
            document = {
                "info": {"description": f"Verified lightning channels: {split}"},
                "images": coco_images,
                "annotations": coco_annotations,
                "categories": [
                    {"id": 1, "name": "lightning_channel", "supercategory": "weather"}
                ],
            }
            annotation_path = staged / "annotations" / f"instances_{split}.json"
            annotation_path.parent.mkdir(parents=True, exist_ok=True)
            annotation_path.write_text(json.dumps(document, indent=2) + "\n")
            split_summaries[split] = validate_coco_dataset(annotation_path, image_dir)

        manifest = {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "source_format": "Label Studio JSON",
            "source_export": str(export_path),
            "source_export_sha256": _sha256(export_path),
            "ratios": ratios,
            "source_assignments": assignments,
            "splits": split_summaries,
        }
        (staged / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(staged, output)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Import completed Label Studio tasks into source-grouped COCO splits"
    )
    parser.add_argument("export", type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    args = parser.parse_args(argv)
    manifest = import_label_studio_dataset(
        args.export,
        args.images,
        args.output,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
    )
    print(json.dumps(manifest, indent=2))
    print(f"output: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
