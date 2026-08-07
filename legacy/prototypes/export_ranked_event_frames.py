#!/usr/bin/env python3
"""Export ranked alternate frames and readable 5x5 review sheets."""

from __future__ import annotations

import csv
import argparse
from pathlib import Path

import cv2
import numpy as np


VIDEO = "GX010422.mp4"
RANKING = Path("analysis/all_event_frame_ranking.csv")
OUT = Path("analysis/all_event_frame_review")
LIMIT = 450


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1, help="First global rank")
    parser.add_argument("--limit", type=int, default=LIMIT, help="Number of frames")
    parser.add_argument("--sheet-offset", type=int, default=0)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    with RANKING.open() as source:
        all_rows = list(csv.DictReader(source))
        rows = all_rows[args.start - 1 : args.start - 1 + args.limit]

    capture = cv2.VideoCapture(VIDEO)
    fps = capture.get(cv2.CAP_PROP_FPS)
    thumbnails: list[np.ndarray] = []
    for row in rows:
        rank = int(row["global_rank"])
        timestamp = float(row["frame_time"])
        capture.set(cv2.CAP_PROP_POS_FRAMES, round(timestamp * fps))
        ok, frame = capture.read()
        if not ok:
            continue
        filename = OUT / f"{rank:04d}_{timestamp:08.2f}s.jpg"
        cv2.imwrite(str(filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 96])

        thumb = cv2.resize(frame, (768, 432), interpolation=cv2.INTER_AREA)
        cv2.rectangle(thumb, (0, 0), (255, 42), (0, 0, 0), -1)
        cv2.putText(
            thumb,
            f"#{rank}  {timestamp:.2f}s",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        thumbnails.append(thumb)
    capture.release()

    for sheet_number, start in enumerate(
        range(0, len(thumbnails), 25), args.sheet_offset + 1
    ):
        batch = thumbnails[start : start + 25]
        while len(batch) < 25:
            batch.append(np.zeros_like(thumbnails[0]))
        sheet = np.vstack(
            [np.hstack(batch[row : row + 5]) for row in range(0, 25, 5)]
        )
        cv2.imwrite(
            str(OUT / f"contact_sheet_{sheet_number:02d}.jpg"),
            sheet,
            [cv2.IMWRITE_JPEG_QUALITY, 94],
        )
    print(f"exported {len(thumbnails)} frames and {sheet_number} contact sheets")


if __name__ == "__main__":
    main()
