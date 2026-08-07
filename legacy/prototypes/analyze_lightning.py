#!/usr/bin/env python3
"""Rank likely lightning events from FFmpeg signalstats metadata."""

from __future__ import annotations

import csv
import math
import re
import statistics
from collections import deque
from pathlib import Path


SOURCE = Path("analysis/frame_metrics.txt")
CSV_OUT = Path("analysis/lightning_candidates.csv")
FPS = 100.0

FRAME_RE = re.compile(r"frame:(\d+).*pts_time:([0-9.]+)")


def percentile(values: list[float], percentile_value: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile_value
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def read_metrics() -> tuple[list[float], list[float], list[float], list[float]]:
    times: list[float] = []
    yavg: list[float] = []
    yhigh: list[float] = []
    ydif: list[float] = []
    current: dict[str, float] = {}

    def flush() -> None:
        if {"time", "YAVG", "YHIGH", "YDIF"} <= current.keys():
            times.append(current["time"])
            yavg.append(current["YAVG"])
            yhigh.append(current["YHIGH"])
            ydif.append(current["YDIF"])

    with SOURCE.open() as input_file:
        for line in input_file:
            if line.startswith("frame:"):
                flush()
                current = {}
                match = FRAME_RE.search(line)
                if match:
                    current["time"] = float(match.group(2))
            elif line.startswith("lavfi.signalstats."):
                key, value = line.strip().split("=", 1)
                short_key = key.rsplit(".", 1)[-1]
                if short_key in {"YAVG", "YHIGH", "YDIF"}:
                    current[short_key] = float(value)
        flush()
    return times, yavg, yhigh, ydif


def main() -> None:
    times, yavg, yhigh, ydif = read_metrics()
    # Compare each frame with the median of the preceding 0.30 seconds. This
    # suppresses slow exposure changes while retaining single-frame flashes.
    baseline_window: deque[float] = deque(maxlen=30)
    raw: list[dict[str, float]] = []
    for index, value in enumerate(yavg):
        baseline = statistics.median(baseline_window) if baseline_window else value
        rise = value - baseline
        high_rise = yhigh[index] - (
            statistics.median(yhigh[max(0, index - 30):index]) if index else yhigh[index]
        )
        raw.append(
            {
                "index": float(index),
                "time": times[index],
                "rise": rise,
                "high_rise": high_rise,
                "ydif": ydif[index],
                "yavg": value,
            }
        )
        baseline_window.append(value)

    rise_cutoff = max(1.0, percentile([row["rise"] for row in raw], 0.995))
    diff_cutoff = max(0.8, percentile(ydif, 0.995))
    hits = [
        row
        for row in raw
        if row["rise"] >= rise_cutoff
        and (row["ydif"] >= diff_cutoff or row["high_rise"] >= 3.0)
    ]

    # Collapse detections less than 0.75 seconds apart into one lightning event.
    groups: list[list[dict[str, float]]] = []
    for hit in hits:
        if not groups or hit["time"] - groups[-1][-1]["time"] > 0.75:
            groups.append([hit])
        else:
            groups[-1].append(hit)

    events = []
    for group in groups:
        peak = max(
            group,
            key=lambda row: row["rise"] * 2.0
            + row["ydif"]
            + max(0.0, row["high_rise"]) * 0.25,
        )
        score = (
            peak["rise"] * 2.0
            + peak["ydif"]
            + max(0.0, peak["high_rise"]) * 0.25
        )
        events.append(
            {
                "rank": 0,
                "peak_time": peak["time"],
                "first_time": group[0]["time"],
                "last_time": group[-1]["time"],
                "score": score,
                "rise": peak["rise"],
                "ydif": peak["ydif"],
                "high_rise": peak["high_rise"],
                "peak_yavg": peak["yavg"],
                "hit_frames": len(group),
            }
        )

    events.sort(key=lambda event: event["score"], reverse=True)
    for rank, event in enumerate(events, 1):
        event["rank"] = rank

    CSV_OUT.parent.mkdir(exist_ok=True)
    with CSV_OUT.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(events[0].keys()))
        writer.writeheader()
        writer.writerows(events)

    print(f"frames={len(times)} rise_cutoff={rise_cutoff:.3f} diff_cutoff={diff_cutoff:.3f}")
    print(f"candidate_events={len(events)} output={CSV_OUT}")
    for event in events[:30]:
        print(
            f"#{event['rank']:02d} t={event['peak_time']:8.2f}s "
            f"score={event['score']:7.2f} rise={event['rise']:6.2f} "
            f"ydif={event['ydif']:6.2f} frames={event['hit_frames']}"
        )


if __name__ == "__main__":
    main()
