from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .training import CocoDetectionDataset, _training_imports


def _iou(first: Any, second: Any) -> float:
    x0, y0 = max(float(first[0]), float(second[0])), max(float(first[1]), float(second[1]))
    x1, y1 = min(float(first[2]), float(second[2])), min(float(first[3]), float(second[3]))
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    first_area = max(0.0, float(first[2] - first[0])) * max(0.0, float(first[3] - first[1]))
    second_area = max(0.0, float(second[2] - second[0])) * max(0.0, float(second[3] - second[1]))
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def evaluate(
    release: Path,
    checkpoint: Path,
    output: Path,
    *,
    split: str = "validation",
    score_threshold: float = 0.25,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    torch, torchvision = _training_imports()
    dataset = CocoDetectionDataset(release, split)
    if not dataset:
        raise ValueError(f"{split} split is empty")
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=None, weights_backbone=None, num_classes=2
    )
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    true_positives = false_positives = false_negatives = 0
    with torch.no_grad():
        for image, target in dataset:
            prediction = model([image])[0]
            predicted = [
                box for box, score, label in zip(
                    prediction["boxes"], prediction["scores"], prediction["labels"]
                ) if float(score) >= score_threshold and int(label) == 1
            ]
            expected = list(target["boxes"])
            matched: set[int] = set()
            for box in predicted:
                candidates = [(index, _iou(box, truth)) for index, truth in enumerate(expected) if index not in matched]
                best = max(candidates, key=lambda item: item[1], default=None)
                if best and best[1] >= iou_threshold:
                    matched.add(best[0])
                    true_positives += 1
                else:
                    false_positives += 1
            false_negatives += len(expected) - len(matched)
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0.0
    report = {
        "schema_version": 1,
        "split": split,
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report
