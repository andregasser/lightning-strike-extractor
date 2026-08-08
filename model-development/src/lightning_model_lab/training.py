from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _training_imports() -> tuple[Any, Any]:
    try:
        import torch
        import torchvision
    except ImportError as error:
        raise RuntimeError("Training dependencies are missing; run `uv sync --extra train`") from error
    return torch, torchvision


class CocoDetectionDataset:
    def __init__(self, release: Path, split: str) -> None:
        self.release = release.resolve()
        self.split = split
        document = json.loads(
            (self.release / "annotations" / f"instances_{split}.json").read_text()
        )
        annotations: dict[int, list[dict[str, Any]]] = {}
        for annotation in document["annotations"]:
            annotations.setdefault(annotation["image_id"], []).append(annotation)
        self.rows = [(image, annotations.get(image["id"], [])) for image in document["images"]]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Any, dict[str, Any]]:
        torch, _ = _training_imports()
        image, annotations = self.rows[index]
        pixels = cv2.imread(str(self.release / "images" / self.split / image["file_name"]))
        if pixels is None:
            raise RuntimeError(f"Could not decode training image: {image['file_name']}")
        pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1))).float() / 255
        boxes = [
            [box[0], box[1], box[0] + box[2], box[1] + box[3]]
            for box in (annotation["bbox"] for annotation in annotations)
        ]
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.ones((len(boxes),), dtype=torch.int64),
            "image_id": torch.tensor(image["id"]),
        }
        return tensor, target


def train(release: Path, output: Path, *, epochs: int = 10, seed: int = 17) -> dict[str, Any]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    manifest = json.loads((release / "manifest.json").read_text())
    if output.exists():
        raise ValueError(f"Refusing to overwrite training output: {output}")
    torch, torchvision = _training_imports()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset = CocoDetectionDataset(release, "train")
    if not dataset:
        raise ValueError("Training split is empty")
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, collate_fn=lambda batch: tuple(zip(*batch))
    )
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=None, weights_backbone=None, num_classes=2
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    losses: list[float] = []
    for _ in range(epochs):
        epoch_loss = 0.0
        for images, targets in loader:
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
            loss = sum(model(images, targets).values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
        losses.append(epoch_loss / len(loader))
    output.mkdir(parents=True)
    checkpoint = output / "checkpoint.pt"
    torch.save(model.state_dict(), checkpoint)
    report = {
        "schema_version": 1,
        "dataset_release": manifest["release_id"],
        "architecture": "fasterrcnn_mobilenet_v3_large_320_fpn",
        "epochs": epochs,
        "seed": seed,
        "losses": losses,
        "checkpoint": checkpoint.name,
    }
    (output / "training.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
