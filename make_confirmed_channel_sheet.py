#!/usr/bin/env python3
"""Build a labelled overview of all manually confirmed channel frames."""

from pathlib import Path

import cv2
import numpy as np


SOURCE = Path("output/confirmed_visible_lightning_channels")
OUTPUT = SOURCE / "contact_sheet_confirmed.jpg"


def main() -> None:
    images = sorted(
        path
        for path in SOURCE.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and path.name != OUTPUT.name
    )
    thumbs: list[np.ndarray] = []
    for path in images:
        frame = cv2.imread(str(path))
        thumb = cv2.resize(frame, (768, 432), interpolation=cv2.INTER_AREA)
        cv2.rectangle(thumb, (0, 0), (768, 42), (0, 0, 0), -1)
        cv2.putText(
            thumb,
            path.stem,
            (10, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        thumbs.append(thumb)

    while len(thumbs) % 4:
        thumbs.append(np.zeros_like(thumbs[0]))
    sheet = np.vstack(
        [np.hstack(thumbs[start : start + 4]) for start in range(0, len(thumbs), 4)]
    )
    cv2.imwrite(str(OUTPUT), sheet, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"wrote {OUTPUT} with {len(images)} confirmed frames")


if __name__ == "__main__":
    main()
