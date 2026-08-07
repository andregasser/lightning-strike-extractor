from __future__ import annotations

import argparse
import json
import os
import tempfile
import zipfile
from copy import deepcopy
from pathlib import Path

from tools.model_development.validate_coco import validate_coco_dataset


def package_cvat_dataset(dataset: Path, output: Path) -> dict[str, object]:
    """Build a CVAT-compatible COCO archive from a prepared dataset."""
    dataset = dataset.resolve()
    output = output.resolve()
    annotations_path = dataset / "annotations" / "proposals.json"
    images_root = dataset / "images"
    validation = validate_coco_dataset(annotations_path, images_root)
    if output.exists():
        raise ValueError(f"Refusing to overwrite existing CVAT archive: {output}")

    document = json.loads(annotations_path.read_text())
    cvat_document = deepcopy(document)
    archive_names: set[str] = set()
    image_files: list[tuple[Path, str]] = []
    for image in cvat_document["images"]:
        original_name = str(image["file_name"])
        source_id = str(image.get("source_id") or Path(original_name).parent.name)
        archive_name = f"{source_id}__{Path(original_name).name}"
        if archive_name in archive_names:
            raise ValueError(f"Duplicate CVAT image name: {archive_name}")
        archive_names.add(archive_name)
        source_path = images_root / original_name
        image["file_name"] = archive_name
        image_files.append((source_path, f"images/default/{archive_name}"))

    for annotation in cvat_document["annotations"]:
        annotation.setdefault("segmentation", [])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.stem}-", dir=output.parent) as temporary:
        temporary_archive = Path(temporary) / output.name
        with zipfile.ZipFile(temporary_archive, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                "annotations/instances_default.json",
                json.dumps(cvat_document, indent=2, ensure_ascii=False) + "\n",
            )
            for source_path, archive_path in image_files:
                archive.write(source_path, archive_path)
        os.replace(temporary_archive, output)
    return {
        "archive": str(output),
        "format": "COCO 1.0",
        "images": len(image_files),
        "annotations": validation["annotations"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Package a prepared dataset for CVAT")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(package_cvat_dataset(args.dataset, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
