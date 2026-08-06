#!/usr/bin/env python3
"""Rank lightning candidates by visible thin-channel geometry."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np


VIDEO = "GX010422.mp4"
CANDIDATES = Path("analysis/lightning_candidates.csv")
OUTPUT = Path("analysis/channel_geometry_ranking.csv")


def frame_score(previous: np.ndarray, current: np.ndarray) -> tuple[float, int, float]:
    previous = cv2.resize(previous, (960, 540))
    current = cv2.resize(current, (960, 540))
    previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)

    temporal = cv2.subtract(gray, previous_gray)
    fine = cv2.subtract(gray, cv2.GaussianBlur(gray, (0, 0), 5.0))
    channel = cv2.bitwise_and(fine, temporal)
    channel = cv2.threshold(channel, 10, 255, cv2.THRESH_BINARY)[1]
    channel = cv2.morphologyEx(
        channel, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)
    )

    edges = cv2.Canny(channel, 30, 100)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=12, minLineLength=12, maxLineGap=8
    )
    lengths: list[float] = []
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            lengths.append(float(np.hypot(x2 - x1, y2 - y1)))

    line_count = len(lengths)
    total_length = sum(sorted(lengths, reverse=True)[:30])
    bright_area = float(np.count_nonzero(temporal > 20))
    # Thin, multi-segment structures score well; broad cloud illumination is
    # penalized by its bright area.
    score = total_length * (1.0 + min(line_count, 20) / 10.0)
    score /= 1.0 + bright_area / 12000.0
    return score, line_count, bright_area


def main() -> None:
    with CANDIDATES.open() as source:
        candidates = list(csv.DictReader(source))

    capture = cv2.VideoCapture(VIDEO)
    fps = capture.get(cv2.CAP_PROP_FPS)
    results = []
    for candidate in candidates:
        peak = float(candidate["peak_time"])
        start_frame = max(0, round((peak - 0.20) * fps))
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ok, previous = capture.read()
        best = (0.0, 0, 0.0, peak)
        for offset in range(1, 41):
            ok, current = capture.read()
            if not ok:
                break
            score, count, area = frame_score(previous, current)
            timestamp = (start_frame + offset) / fps
            if score > best[0]:
                best = (score, count, area, timestamp)
            previous = current
        results.append(
            {
                "source_rank": int(candidate["rank"]),
                "event_peak": peak,
                "best_frame_time": best[3],
                "geometry_score": best[0],
                "line_segments": best[1],
                "bright_area": best[2],
            }
        )

    capture.release()
    results.sort(key=lambda row: float(row["geometry_score"]), reverse=True)
    for rank, result in enumerate(results, 1):
        result["geometry_rank"] = rank

    fields = [
        "geometry_rank",
        "source_rank",
        "event_peak",
        "best_frame_time",
        "geometry_score",
        "line_segments",
        "bright_area",
    ]
    with OUTPUT.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    for result in results[:30]:
        print(
            f"#{result['geometry_rank']:02d} "
            f"t={result['best_frame_time']:.2f}s "
            f"score={result['geometry_score']:.1f} "
            f"segments={result['line_segments']}"
        )


if __name__ == "__main__":
    main()
