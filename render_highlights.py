#!/usr/bin/env python3
"""Render a chronological 4K reel containing every detected lightning event."""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path


SOURCE = "GX010422.mp4"
OUTPUT = "output/florida_lightning_all_flashes_4k.mp4"
PRE_ROLL = 1.0
POST_ROLL = 1.5
SOURCE_DURATION = 3072.0


def merged_event_windows(events: list[dict[str, str]]) -> list[tuple[float, float, int]]:
    """Merge overlapping clip windows while retaining the event count."""
    windows: list[list[float | int]] = []
    for event in sorted(events, key=lambda row: float(row["peak_time"])):
        peak = float(event["peak_time"])
        start = max(0.0, peak - PRE_ROLL)
        end = min(SOURCE_DURATION, peak + POST_ROLL)
        if windows and start <= float(windows[-1][1]):
            windows[-1][1] = max(float(windows[-1][1]), end)
            windows[-1][2] = int(windows[-1][2]) + 1
        else:
            windows.append([start, end, 1])
    return [(float(start), float(end), int(count)) for start, end, count in windows]


def main() -> None:
    with Path("analysis/lightning_candidates.csv").open() as input_file:
        events = list(csv.DictReader(input_file))
    windows = merged_event_windows(events)

    command = ["ffmpeg", "-hide_banner", "-y"]
    for start, end, _ in windows:
        command.extend(["-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", SOURCE])

    filters: list[str] = []
    concat_labels: list[str] = []
    for index, (start, end, _) in enumerate(windows):
        duration = end - start
        filters.append(
            f"[{index}:v]trim=start=0:end={duration:.3f},"
            f"setpts=PTS-STARTPTS,fps=25,format=yuv420p[v{index}]"
        )
        filters.append(
            f"[{index}:a]atrim=start=0:end={duration:.3f},"
            f"asetpts=PTS-STARTPTS,aresample=48000[a{index}]"
        )
        concat_labels.extend([f"[v{index}]", f"[a{index}]"])

    filters.append(
        "".join(concat_labels)
        + f"concat=n={len(windows)}:v=1:a=1[vout][aout]"
    )
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[vout]",
            "-map",
            "[aout]",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-tag:v",
            "avc1",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-colorspace",
            "bt709",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            OUTPUT,
        ]
    )

    total_duration = sum(end - start for start, end, _ in windows)
    print(
        f"Rendering all {len(events)} detected lightning events as "
        f"{len(windows)} chronological clips ({total_duration:.1f}s)."
    )
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
