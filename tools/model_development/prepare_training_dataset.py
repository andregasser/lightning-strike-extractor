from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from lightning_extractor.detector import LightningDetector
from tools.model_development.export_training_frames import export_training_frames
from tools.model_development.package_cvat_dataset import package_cvat_dataset
from tools.model_development.preannotate_coco import preannotate
from tools.model_development.validate_coco import validate_coco_dataset


def prepare_training_dataset(
    runs_root: Path,
    output: Path,
    *,
    max_events_per_video: int = 100,
    context_frames: int = 2,
    jpeg_quality: int = 95,
    detector: LightningDetector | None = None,
) -> dict:
    """Export frames, propose COCO boxes, and validate one atomic dataset."""
    output = output.resolve()
    if output.exists():
        raise ValueError(f"Refusing to overwrite existing dataset: {output}")

    # Load the fixed detector before extracting frames so missing model dependencies
    # or artifacts fail without doing potentially expensive video work.
    session = detector or LightningDetector()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staged = Path(temporary) / output.name
        manifest = export_training_frames(
            runs_root,
            staged,
            max_events_per_video=max_events_per_video,
            context_frames=context_frames,
            jpeg_quality=jpeg_quality,
        )
        annotations = staged / "annotations" / "proposals.json"
        proposals = preannotate(staged / "images", annotations, detector=session)
        validation = validate_coco_dataset(annotations, staged / "images")
        cvat = package_cvat_dataset(staged, staged / "cvat" / "import.zip")
        report = {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "dataset_manifest": "manifest.json",
            "annotations": "annotations/proposals.json",
            "sources": len(manifest["sources"]),
            "frames": len(manifest["frames"]),
            "proposal_model": proposals["info"]["proposal_model"],
            "validation": validation,
            "cvat": {
                **cvat,
                "archive": "cvat/import.zip",
            },
        }
        (staged / "preparation.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        )
        os.replace(staged, output)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export, preannotate, and validate a lightning training dataset"
    )
    parser.add_argument("runs", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-events-per-video", type=int, default=100)
    parser.add_argument("--context-frames", type=int, default=2)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    args = parser.parse_args(argv)
    report = prepare_training_dataset(
        args.runs,
        args.output,
        max_events_per_video=args.max_events_per_video,
        context_frames=args.context_frames,
        jpeg_quality=args.jpeg_quality,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"output: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
