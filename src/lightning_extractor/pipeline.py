from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .config import Config
from .detection import detect_flashes, rank_event_frames
from .models import CandidateFrame, FlashEvent
from .probe import probe_video


def _source_id(video: Path) -> str:
    stat = video.stat()
    identity = f"{video.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode()
    return hashlib.sha256(identity).hexdigest()[:8]


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        path.write_text("")
        return
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def export_stills(
    video: Path, candidates: list[CandidateFrame], output: Path, config: Config
) -> int:
    output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    thumbnails: list[np.ndarray] = []
    selected = candidates[: config.export.top]
    for candidate in selected:
        capture.set(cv2.CAP_PROP_POS_FRAMES, candidate.frame_number)
        ok, frame = capture.read()
        if not ok:
            continue
        filename = output / f"{candidate.rank:04d}_{candidate.time:09.2f}s.jpg"
        cv2.imwrite(str(filename), frame, [cv2.IMWRITE_JPEG_QUALITY, config.export.jpeg_quality])
        width = 640
        height = round(frame.shape[0] * width / frame.shape[1])
        thumb = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        cv2.rectangle(thumb, (0, 0), (300, 38), (0, 0, 0), -1)
        cv2.putText(thumb, f"#{candidate.rank}  {candidate.time:.2f}s", (8, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        thumbnails.append(thumb)
    capture.release()
    columns = config.export.contact_sheet_columns
    if thumbnails:
        blank = np.zeros_like(thumbnails[0])
        while len(thumbnails) % columns:
            thumbnails.append(blank.copy())
        sheet = np.vstack([
            np.hstack(thumbnails[index:index + columns])
            for index in range(0, len(thumbnails), columns)
        ])
        cv2.imwrite(str(output.parent / "contact-sheet.jpg"), sheet,
                    [cv2.IMWRITE_JPEG_QUALITY, 94])
    return len(selected)


def analyze(
    video: Path,
    runs_root: Path,
    config: Config,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
) -> Path:
    video = video.resolve()
    source = probe_video(video)
    fps = float(source["video"]["fps"])  # type: ignore[index]
    if fps <= 0:
        raise RuntimeError("The video frame rate could not be determined")
    run = runs_root / f"{video.stem}-{_source_id(video)}"
    results = run / "results"
    exports = run / "exports"
    results.mkdir(parents=True, exist_ok=True)
    exports.mkdir(parents=True, exist_ok=True)
    _write_json(run / "source.json", source)
    _write_json(run / "config.json", config.as_dict())
    _write_json(run / "run.json", {
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "start_seconds": start_seconds,
        "end_seconds": end_seconds,
        "source_id": _source_id(video),
    })
    events = detect_flashes(video, fps, config, start_seconds, end_seconds)
    _write_json(results / "events.json", [event.as_dict() for event in events])
    _write_csv(results / "events.csv", (event.as_dict() for event in events))
    candidates = rank_event_frames(video, fps, events, config)
    _write_json(results / "candidates.json", [item.as_dict() for item in candidates])
    _write_csv(results / "candidates.csv", (item.as_dict() for item in candidates))
    still_count = export_stills(video, candidates, exports / "stills", config)
    _write_json(results / "summary.json", {
        "events": len(events),
        "candidate_frames": len(candidates),
        "exported_stills": still_count,
        "best_geometry_score": candidates[0].geometry_score if candidates else 0.0,
    })
    _write_json(run / "run.json", {
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "start_seconds": start_seconds,
        "end_seconds": end_seconds,
        "source_id": _source_id(video),
    })
    return run
