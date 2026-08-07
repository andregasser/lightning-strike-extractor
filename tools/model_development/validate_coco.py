from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def validate_coco_dataset(annotation_path: Path, images_root: Path) -> dict[str, int]:
    if not annotation_path.is_file():
        raise ValueError(f"Annotation file does not exist: {annotation_path}")
    data = json.loads(annotation_path.read_text())
    if not isinstance(data, dict):
        raise TypeError("COCO annotations must be a JSON object")
    images = data.get("images")
    annotations = data.get("annotations")
    categories = data.get("categories")
    if not all(isinstance(value, list) for value in (images, annotations, categories)):
        raise ValueError("COCO file must contain images, annotations, and categories lists")

    image_by_id: dict[int, dict[str, Any]] = {}
    for image in images:
        if not isinstance(image, dict) or not isinstance(image.get("id"), int):
            raise TypeError("Every COCO image needs an integer id")
        if image["id"] in image_by_id:
            raise ValueError(f"Duplicate image id: {image['id']}")
        filename = image.get("file_name")
        width, height = image.get("width"), image.get("height")
        if not isinstance(filename, str) or not filename:
            raise ValueError(f"Image {image['id']} has no file_name")
        if not isinstance(width, int) or width <= 0 or not isinstance(height, int) or height <= 0:
            raise ValueError(f"Image {image['id']} has invalid dimensions")
        if not (images_root / filename).is_file():
            raise ValueError(f"Referenced image does not exist: {images_root / filename}")
        image_by_id[image["id"]] = image

    category_ids = {
        item.get("id")
        for item in categories
        if isinstance(item, dict) and isinstance(item.get("id"), int)
    }
    if not category_ids:
        raise ValueError("COCO dataset has no valid categories")
    positive_images: set[int] = set()
    for annotation in annotations:
        if not isinstance(annotation, dict):
            raise TypeError("Every COCO annotation must be an object")
        image_id, category_id = annotation.get("image_id"), annotation.get("category_id")
        if image_id not in image_by_id:
            raise ValueError(f"Annotation references unknown image id: {image_id}")
        if category_id not in category_ids:
            raise ValueError(f"Annotation references unknown category id: {category_id}")
        box = annotation.get("bbox")
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError("Every annotation bbox must be [x, y, width, height]")
        if not all(isinstance(value, (int, float)) for value in box):
            raise ValueError("Bounding-box values must be numeric")
        x, y, width, height = (float(value) for value in box)
        image = image_by_id[image_id]
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError(f"Annotation has invalid bbox: {box}")
        if x + width > image["width"] or y + height > image["height"]:
            raise ValueError(f"Annotation bbox lies outside image {image_id}: {box}")
        positive_images.add(image_id)

    return {
        "images": len(images),
        "annotations": len(annotations),
        "categories": len(category_ids),
        "positive_images": len(positive_images),
        "negative_images": len(images) - len(positive_images),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a COCO lightning training set")
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--images", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(validate_coco_dataset(args.annotations, args.images), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
