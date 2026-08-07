#!/usr/bin/env python3
"""Export full-resolution candidates from the independent full-video scan."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np


VIDEO = "GX010422.mp4"
RANKING = Path("analysis/full_video_channel_scan_v2/shortlist.csv")
OUT = Path("analysis/full_video_channel_scan_v2/review")
LIMIT = 400


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with RANKING.open() as source:
        rows = list(csv.DictReader(source))[:LIMIT]

    capture = cv2.VideoCapture(VIDEO)
    thumbnails: list[np.ndarray] = []
    for row in rows:
        rank = int(row["rank"])
        frame_number = int(row["frame"])
        timestamp = float(row["time"])
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = capture.read()
        if not ok:
            continue

        cv2.imwrite(
            str(OUT / f"{rank:04d}_{timestamp:08.2f}s_f{frame_number:06d}.jpg"),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 97],
        )
        thumb = cv2.resize(frame, (768, 432), interpolation=cv2.INTER_AREA)
        cv2.rectangle(thumb, (0, 0), (330, 44), (0, 0, 0), -1)
        cv2.putText(
            thumb,
            f"#{rank}  {timestamp:.2f}s  f{frame_number}",
            (10, 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        thumbnails.append(thumb)
    capture.release()

    for sheet_number, start in enumerate(range(0, len(thumbnails), 25), 1):
        batch = thumbnails[start : start + 25]
        while len(batch) < 25:
            batch.append(np.zeros_like(thumbnails[0]))
        sheet = np.vstack(
            [np.hstack(batch[row : row + 5]) for row in range(0, 25, 5)]
        )
        cv2.imwrite(
            str(OUT / f"contact_sheet_{sheet_number:02d}.jpg"),
            sheet,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
    print(f"exported {len(thumbnails)} frames")


if __name__ == "__main__":
    main()
