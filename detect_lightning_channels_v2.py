#!/usr/bin/env python3
"""Spatiotemporal lightning-channel detector.

Unlike the first pass, this ranks thin, locally bright structures directly and
uses a temporal median to suppress static lights, cloud texture and raindrops.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


VIDEO = "GX010422.mp4"
EVENTS = Path("analysis/lightning_candidates.csv")
OUT = Path("analysis/channel_detector_v2")


def channel_score(
    frame: np.ndarray, temporal_median: np.ndarray
) -> tuple[float, float, int, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (960, 540), interpolation=cv2.INTER_AREA)
    local_background = cv2.GaussianBlur(gray, (0, 0), 7.0)
    ridge = cv2.subtract(gray, local_background)
    temporal = cv2.subtract(gray, temporal_median)

    # A channel must be locally line-like and newly brighter than its temporal
    # surroundings. Cloud-wide illumination therefore contributes little.
    mask = ((ridge >= 7) & (temporal >= 4) & (gray >= 65)).astype(np.uint8) * 255
    mask[:28] = 0
    mask[-45:] = 0
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )

    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    component_scores: list[float] = []
    longest = 0.0
    accepted = 0
    for x, y, width, height, area in stats[1:count]:
        length = float(max(width, height))
        short = float(max(min(width, height), 1))
        thickness = area / max(length, 1.0)
        aspect = length / short
        if length < 10 or thickness > 10 or aspect < 1.7:
            continue
        roi = ridge[y : y + height, x : x + width]
        strength = float(np.percentile(roi[roi > 0], 85)) if np.any(roi > 0) else 0
        component_scores.append(
            length * np.sqrt(aspect) * (1.0 + strength / 16.0)
            / (1.0 + thickness / 3.0)
        )
        accepted += 1
        longest = max(longest, length)

    # Reward multiple separated channel pieces, but prevent bright cloud edges
    # from winning merely because they cover a large area.
    component_scores.sort(reverse=True)
    geometry = sum(component_scores[:4])
    cloud_area = float(np.count_nonzero(temporal >= 8)) / temporal.size
    penalty = 1.0 + 7.0 * cloud_area
    return geometry / penalty, cloud_area, accepted, longest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-limit", type=int)
    parser.add_argument("--keep", type=int, default=350)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    with EVENTS.open() as source:
        events = list(csv.DictReader(source))
    if args.event_limit:
        events = events[: args.event_limit]

    capture = cv2.VideoCapture(VIDEO)
    fps = capture.get(cv2.CAP_PROP_FPS)
    candidates: list[dict[str, float | int | np.ndarray]] = []
    for event_number, event in enumerate(events, 1):
        peak = float(event["peak_time"])
        start = max(0, round((peak - 0.55) * fps))
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for _ in range(111):
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        if len(frames) < 7:
            continue

        reduced = [
            cv2.resize(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                (960, 540),
                interpolation=cv2.INTER_AREA,
            )
            for frame in frames
        ]
        # A 9-frame rolling median is long enough to reject a single return
        # stroke but short enough to follow changing illumination.
        for index in range(4, len(frames) - 4):
            temporal_median = np.median(
                np.stack(reduced[index - 4 : index + 5]), axis=0
            ).astype(np.uint8)
            score, cloud_area, components, longest = channel_score(
                frames[index], temporal_median
            )
            if score <= 0:
                continue
            candidates.append(
                {
                    "event": event_number,
                    "time": (start + index) / fps,
                    "score": score,
                    "cloud_area": cloud_area,
                    "components": components,
                    "longest": longest,
                    "frame": frames[index],
                }
            )
        if event_number % 20 == 0:
            print(f"processed {event_number}/{len(events)}", flush=True)
    capture.release()

    candidates.sort(key=lambda row: float(row["score"]), reverse=True)
    selected = []
    for row in candidates:
        if all(abs(float(row["time"]) - float(old["time"])) >= 0.04 for old in selected):
            selected.append(row)
        if len(selected) == args.keep:
            break

    fields = ["rank", "event", "time", "score", "cloud_area", "components", "longest"]
    with (OUT / "ranking.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(selected, 1):
            writer.writerow({"rank": rank, **{key: row[key] for key in fields[1:]}})

    thumbnails = []
    for rank, row in enumerate(selected, 1):
        frame = row.pop("frame")
        path = OUT / f"{rank:04d}_{float(row['time']):08.2f}s.jpg"
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 96])
        thumb = cv2.resize(frame, (768, 432), interpolation=cv2.INTER_AREA)
        cv2.rectangle(thumb, (0, 0), (290, 42), (0, 0, 0), -1)
        cv2.putText(
            thumb,
            f"#{rank} {float(row['time']):.2f}s",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        thumbnails.append(thumb)
    for number, offset in enumerate(range(0, len(thumbnails), 25), 1):
        batch = thumbnails[offset : offset + 25]
        while len(batch) < 25:
            batch.append(np.zeros_like(thumbnails[0]))
        sheet = np.vstack(
            [np.hstack(batch[row : row + 5]) for row in range(0, 25, 5)]
        )
        cv2.imwrite(str(OUT / f"contact_sheet_{number:02d}.jpg"), sheet)
    print(f"exported {len(selected)} ranked candidates to {OUT}")


if __name__ == "__main__":
    main()
