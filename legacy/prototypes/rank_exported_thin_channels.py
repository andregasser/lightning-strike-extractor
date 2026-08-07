#!/usr/bin/env python3
"""Rank exported review frames by thin locally bright channel-like structures."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np


SOURCE = Path("analysis/all_event_frame_review")
OUTPUT = Path("analysis/thin_channel_ranking.csv")


def score_frame(path: Path) -> tuple[float, int, float]:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    small = cv2.resize(image, (960, 540), interpolation=cv2.INTER_AREA)
    # Remove slow cloud illumination while preserving narrow bright channels.
    background = cv2.morphologyEx(
        small, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    )
    residual = cv2.subtract(small, background)
    residual[:25] = 0
    residual[-55:] = 0
    _, mask = cv2.threshold(residual, 18, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    best = 0.0
    accepted = 0
    best_length = 0.0
    for x, y, width, height, area in stats[1:count]:
        length = float(max(width, height))
        thickness = area / max(length, 1.0)
        aspect = length / max(float(min(width, height)), 1.0)
        if length < 8 or thickness > 9 or aspect < 1.8:
            continue
        accepted += 1
        component_score = length * aspect / (1.0 + thickness)
        if component_score > best:
            best = component_score
            best_length = length
    return best, accepted, best_length


def main() -> None:
    rows = []
    images = sorted(
        path for path in SOURCE.glob("[0-9][0-9][0-9][0-9]_*.jpg")
        if not path.name.startswith("contact_sheet")
    )
    for number, path in enumerate(images, 1):
        score, components, length = score_frame(path)
        rows.append(
            {
                "filename": path.name,
                "score": score,
                "components": components,
                "best_length": length,
            }
        )
        if number % 100 == 0:
            print(f"processed {number}/{len(images)}", flush=True)
    rows.sort(key=lambda row: float(row["score"]), reverse=True)
    with OUTPUT.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
