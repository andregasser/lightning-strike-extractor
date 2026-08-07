from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

from tools.model_development.validate_coco import validate_coco_dataset

LABEL_CONFIG = """<View>
  <Header value="Draw a tight box around every visible lightning channel."/>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="lightning_channel" background="#FFD200"/>
  </RectangleLabels>
</View>
"""


def _prediction(annotation: dict, image: dict, result_id: str) -> dict:
    x, y, width, height = (float(value) for value in annotation["bbox"])
    image_width = float(image["width"])
    image_height = float(image["height"])
    return {
        "id": result_id,
        "type": "rectanglelabels",
        "from_name": "label",
        "to_name": "image",
        "original_width": image["width"],
        "original_height": image["height"],
        "image_rotation": 0,
        "value": {
            "rotation": 0,
            "x": x / image_width * 100,
            "y": y / image_height * 100,
            "width": width / image_width * 100,
            "height": height / image_height * 100,
            "rectanglelabels": ["lightning_channel"],
        },
        "score": annotation.get("attributes", {}).get("proposal_score"),
    }


def export_label_studio(
    dataset: Path,
    output: Path,
    *,
    image_base_url: str = "http://localhost:8001/images",
) -> dict[str, object]:
    """Convert prepared COCO proposals into Label Studio prediction tasks."""
    dataset = dataset.resolve()
    output = output.resolve()
    annotations_path = dataset / "annotations" / "proposals.json"
    images_root = dataset / "images"
    validation = validate_coco_dataset(annotations_path, images_root)
    if output.exists():
        raise ValueError(f"Refusing to overwrite Label Studio export: {output}")
    if not image_base_url.startswith(("http://", "https://")):
        raise ValueError("image_base_url must be an HTTP or HTTPS URL")

    document = json.loads(annotations_path.read_text())
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in document["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staged = Path(temporary) / output.name
        image_output = staged / "serve" / "images"
        tasks: list[dict] = []
        destination_names: set[str] = set()
        for image in document["images"]:
            source = images_root / image["file_name"]
            destination_name = f"{image['source_id']}__{Path(image['file_name']).name}"
            if destination_name in destination_names:
                raise ValueError(f"Duplicate Label Studio image name: {destination_name}")
            destination_names.add(destination_name)
            destination = image_output / destination_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            proposals = [
                _prediction(annotation, image, f"proposal-{image['id']}-{index}")
                for index, annotation in enumerate(annotations_by_image[image["id"]], 1)
            ]
            task = {
                "id": image["id"],
                "data": {
                    "image": f"{image_base_url.rstrip('/')}/{quote(destination_name)}",
                    "source_id": image["source_id"],
                    "original_file_name": image["file_name"],
                },
            }
            if proposals:
                scores = [item["score"] for item in proposals if item["score"] is not None]
                task["predictions"] = [
                    {
                        "model_version": document["info"]["proposal_model"]["version"],
                        "score": sum(scores) / len(scores) if scores else None,
                        "result": proposals,
                    }
                ]
            tasks.append(task)

        task_path = staged / "import" / "tasks.json"
        task_path.parent.mkdir(parents=True, exist_ok=True)
        task_path.write_text(
            json.dumps(tasks, indent=2, ensure_ascii=False) + "\n"
        )
        config_path = staged / "project" / "label-config.xml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(LABEL_CONFIG)
        manifest = {
            "schema_version": 1,
            "format": "Label Studio JSON predictions",
            "image_base_url": image_base_url,
            "tasks": len(tasks),
            "predictions": validation["annotations"],
            "positive_images": validation["positive_images"],
            "negative_images": validation["negative_images"],
        }
        # Keep export metadata out of *.json so Label Studio directory imports do
        # not mistake it for a second task file with incompatible data keys.
        (staged / "export-info.txt").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
        )
        os.replace(staged, output)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export prepared lightning proposals as Label Studio tasks"
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--image-base-url",
        default="http://localhost:8001/images",
        help="URL prefix used by Label Studio to load exported images",
    )
    args = parser.parse_args(argv)
    manifest = export_label_studio(
        args.dataset,
        args.output,
        image_base_url=args.image_base_url,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"output: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
