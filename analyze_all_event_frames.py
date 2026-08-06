#!/usr/bin/env python3
"""Rank every frame around detected flash events, not just one frame per event."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from analyze_channel_geometry import frame_score


VIDEO = "GX010422.mp4"
CANDIDATES = Path("analysis/lightning_candidates.csv")
OUTPUT = Path("analysis/all_event_frame_ranking.csv")


def main() -> None:
    with CANDIDATES.open() as source:
        events = list(csv.DictReader(source))

    capture = cv2.VideoCapture(VIDEO)
    fps = capture.get(cv2.CAP_PROP_FPS)
    rows: list[dict[str, float | int]] = []

    for event_number, event in enumerate(events, 1):
        peak = float(event["peak_time"])
        # A visible return stroke can be extremely brief. Cover 0.8 seconds
        # around every detected event and retain several distinct frames.
        start_frame = max(0, round((peak - 0.40) * fps))
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ok, previous = capture.read()
        if not ok:
            continue

        local: list[dict[str, float | int]] = []
        for offset in range(1, 81):
            ok, current = capture.read()
            if not ok:
                break
            score, segments, area = frame_score(previous, current)
            local.append(
                {
                    "source_rank": int(event["rank"]),
                    "event_peak": peak,
                    "frame_time": (start_frame + offset) / fps,
                    "geometry_score": score,
                    "line_segments": segments,
                    "bright_area": area,
                }
            )
            previous = current

        # Keep up to six temporally distinct alternatives per event. This
        # avoids losing a one-frame channel to a brighter cloud-only frame.
        selected: list[dict[str, float | int]] = []
        for row in sorted(local, key=lambda item: float(item["geometry_score"]), reverse=True):
            if all(abs(float(row["frame_time"]) - float(old["frame_time"])) >= 0.025 for old in selected):
                selected.append(row)
            if len(selected) == 6:
                break
        rows.extend(selected)
        if event_number % 25 == 0:
            print(f"processed {event_number}/{len(events)} events", flush=True)

    capture.release()
    rows.sort(key=lambda item: float(item["geometry_score"]), reverse=True)
    for rank, row in enumerate(rows, 1):
        row["global_rank"] = rank

    fields = [
        "global_rank",
        "source_rank",
        "event_peak",
        "frame_time",
        "geometry_score",
        "line_segments",
        "bright_area",
    ]
    with OUTPUT.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} ranked event frames to {OUTPUT}")


if __name__ == "__main__":
    main()
