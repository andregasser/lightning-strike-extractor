#!/usr/bin/env python3
"""Select distinct full-video peaks from the FFmpeg channel metric."""

from __future__ import annotations

import csv
import re
from pathlib import Path


SOURCE = Path("analysis/channel_metrics.txt")
OUTPUT = Path("analysis/full_video_channel_peaks.csv")
LIMIT = 600
MIN_SEPARATION = 0.12


def main() -> None:
    current_time = 0.0
    samples: list[tuple[float, float]] = []
    time_pattern = re.compile(r"pts_time:([0-9.]+)")
    value_pattern = re.compile(r"YAVG=([0-9.]+)")
    for line in SOURCE.read_text().splitlines():
        time_match = time_pattern.search(line)
        if time_match:
            current_time = float(time_match.group(1))
            continue
        value_match = value_pattern.search(line)
        if value_match:
            samples.append((float(value_match.group(1)), current_time))

    selected: list[tuple[float, float]] = []
    for score, timestamp in sorted(samples, reverse=True):
        if all(abs(timestamp - old_time) >= MIN_SEPARATION for _, old_time in selected):
            selected.append((score, timestamp))
        if len(selected) == LIMIT:
            break

    with OUTPUT.open("w", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(["rank", "timestamp", "channel_metric"])
        for rank, (score, timestamp) in enumerate(selected, 1):
            writer.writerow([rank, timestamp, score])
    print(f"selected {len(selected)} full-video peaks")


if __name__ == "__main__":
    main()
