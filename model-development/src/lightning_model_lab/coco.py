from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CATEGORY = {"id": 1, "name": "lightning_channel", "supercategory": "weather"}


def load_verified_coco(annotation_path: Path, images_root: Path) -> dict[str, Any]:
    if not annotation_path.is_file():
        raise ValueError(f"Annotation file does not exist: {annotation_path}")
    document = json.loads(annotation_path.read_text())
    if not isinstance(document, dict):
        raise TypeError("COCO annotations must be an object")
    images = document.get("images")
    annotations = document.get("annotations")
    categories = document.get("categories")
    if not all(isinstance(value, list) for value in (images, annotations, categories)):
        raise ValueError("COCO annotations require images, annotations, and categories lists")
    if categories != [CATEGORY]:
        raise ValueError("Dataset must use the canonical lightning_channel category schema")
    image_by_id: dict[int, dict[str, Any]] = {}
    for image in images:
        if not isinstance(image, dict) or not isinstance(image.get("id"), int):
            raise TypeError("Every image requires an integer id")
        if image["id"] in image_by_id:
            raise ValueError(f"Duplicate image id: {image['id']}")
        source_id = image.get("source_id")
        filename = image.get("file_name")
        width, height = image.get("width"), image.get("height")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(f"Image {image['id']} requires source_id")
        if not isinstance(filename, str) or not filename or Path(filename).is_absolute():
            raise ValueError(f"Image {image['id']} has invalid file_name")
        if ".." in Path(filename).parts:
            raise ValueError(f"Image {image['id']} has unsafe file_name")
        if not isinstance(width, int) or width <= 0 or not isinstance(height, int) or height <= 0:
            raise ValueError(f"Image {image['id']} has invalid dimensions")
        if not (images_root / filename).is_file():
            raise ValueError(f"Referenced image does not exist: {images_root / filename}")
        image_by_id[image["id"]] = image
    seen_annotations: set[int] = set()
    for annotation in annotations:
        if not isinstance(annotation, dict) or not isinstance(annotation.get("id"), int):
            raise TypeError("Every annotation requires an integer id")
        if annotation["id"] in seen_annotations:
            raise ValueError(f"Duplicate annotation id: {annotation['id']}")
        seen_annotations.add(annotation["id"])
        if annotation.get("image_id") not in image_by_id or annotation.get("category_id") != 1:
            raise ValueError("Annotation references an unknown image or category")
        if annotation.get("attributes", {}).get("verified") is not True:
            raise ValueError(f"Annotation {annotation['id']} is not human-verified")
        box = annotation.get("bbox")
        if not isinstance(box, list) or len(box) != 4 or not all(
            isinstance(value, (int, float)) for value in box
        ):
            raise ValueError(f"Annotation {annotation['id']} has an invalid bbox")
        x, y, width, height = (float(value) for value in box)
        image = image_by_id[annotation["image_id"]]
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError(f"Annotation {annotation['id']} has an invalid bbox")
        if x + width > image["width"] or y + height > image["height"]:
            raise ValueError(f"Annotation {annotation['id']} lies outside its image")
    return document
