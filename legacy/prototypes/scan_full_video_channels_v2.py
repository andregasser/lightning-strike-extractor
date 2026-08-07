#!/usr/bin/env python3
"""Scan every video frame for lightning channels, independent of flash events."""

from __future__ import annotations

import csv
from collections import deque
from pathlib import Path

import cv2
import numpy as np


VIDEO = "GX010422.mp4"
OUT = Path("analysis/full_video_channel_scan_v2")


def fast_score(gray: np.ndarray, temporal_median: np.ndarray) -> tuple[float, int, float]:
    local = cv2.GaussianBlur(gray, (0, 0), 4.0)
    ridge = cv2.subtract(gray, local)
    temporal = cv2.subtract(gray, temporal_median)
    mask = ((ridge >= 6) & (temporal >= 3) & (gray >= 55)).astype(np.uint8) * 255
    mask[:12] = 0
    mask[-22:] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    values = []
    longest = 0
    for _, _, width, height, area in stats[1:count]:
        length = max(width, height)
        short = max(min(width, height), 1)
        thickness = area / max(length, 1)
        aspect = length / short
        if length < 6 or thickness > 7 or aspect < 1.6:
            continue
        values.append(length * np.sqrt(aspect) / (1 + thickness / 3))
        longest = max(longest, length)
    values.sort(reverse=True)
    cloud_area = float(np.count_nonzero(temporal >= 7)) / temporal.size
    return sum(values[:4]) / (1 + 8 * cloud_area), longest, cloud_area


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(VIDEO)
    fps = capture.get(cv2.CAP_PROP_FPS)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    window: deque[tuple[int, np.ndarray]] = deque(maxlen=7)
    rows = []
    read_count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        index = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        gray = cv2.resize(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            (480, 270),
            interpolation=cv2.INTER_AREA,
        )
        window.append((index, gray))
        read_count += 1
        if len(window) == 7:
            center_index, center = window[3]
            median = np.median(
                np.stack([item[1] for item in window]), axis=0
            ).astype(np.uint8)
            score, longest, cloud_area = fast_score(center, median)
            if score > 5:
                rows.append(
                    {
                        "frame": center_index,
                        "time": center_index / fps,
                        "score": score,
                        "longest": longest,
                        "cloud_area": cloud_area,
                    }
                )
        if read_count % 10000 == 0:
            print(f"scanned {read_count}/{total} frames", flush=True)
    capture.release()

    # Keep local maxima at least 0.20 s apart, then retain a broad shortlist.
    rows.sort(key=lambda row: float(row["score"]), reverse=True)
    selected = []
    for row in rows:
        if all(abs(float(row["time"]) - float(old["time"])) >= 0.20 for old in selected):
            selected.append(row)
        if len(selected) == 1200:
            break
    for rank, row in enumerate(selected, 1):
        row["rank"] = rank
    fields = ["rank", "frame", "time", "score", "longest", "cloud_area"]
    with (OUT / "shortlist.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    print(f"wrote {len(selected)} full-video candidates to {OUT / 'shortlist.csv'}")


if __name__ == "__main__":
    main()
