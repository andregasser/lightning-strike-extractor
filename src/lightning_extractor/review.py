from __future__ import annotations

import json
import os
import webbrowser
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from .config import Config
from .models import CandidateFrame
from .pipeline import select_export_candidates

LABELS = {"y": "lightning", "n": "not-lightning", "u": "uncertain"}


@dataclass(slots=True)
class ReviewItem:
    key: str
    video_run: Path
    source: Path
    video_name: str
    candidate: CandidateFrame


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _video_runs(path: Path) -> list[Path]:
    path = path.resolve()
    if (path / "results" / "candidates.json").is_file():
        return [path]
    videos = path / "videos"
    if videos.is_dir():
        return sorted(
            run
            for run in videos.iterdir()
            if (run / "results" / "candidates.json").is_file()
        )
    raise ValueError(f"No analysis video runs found below: {path}")


def discover_review_items(path: Path, config: Config) -> list[ReviewItem]:
    items: list[ReviewItem] = []
    for video_run in _video_runs(path):
        source_path = video_run / "source.json"
        if not source_path.is_file():
            raise ValueError(f"Run source metadata does not exist: {source_path}")
        source = json.loads(source_path.read_text())
        video = Path(str(source["path"])).expanduser()
        if not video.is_file():
            raise ValueError(f"Source video does not exist: {video}")
        candidates = [
            CandidateFrame(**row)
            for row in json.loads(
                (video_run / "results" / "candidates.json").read_text()
            )
        ]
        for candidate in select_export_candidates(candidates, config):
            key = f"{video_run.name}:{candidate.event_id}"
            items.append(
                ReviewItem(
                    key,
                    video_run,
                    video.resolve(),
                    str(source.get("name", video.name)),
                    candidate,
                )
            )
    return items


def _label_frame(frame: np.ndarray, text: str, peak: bool) -> np.ndarray:
    width = 640
    height = max(1, round(frame.shape[0] * width / frame.shape[1]))
    result = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(result, (0, 0), (width, 44), (0, 0, 0), -1)
    cv2.putText(
        result,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if peak:
        cv2.rectangle(result, (2, 2), (width - 3, height - 3), (0, 255, 255), 5)
    return result


def build_review_preview(item: ReviewItem) -> Path:
    destination = item.video_run / "review" / "previews" / (
        f"{item.candidate.event_id}_{item.candidate.time:010.3f}s.jpg"
    )
    if destination.is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(item.source))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {item.source}")
    start = max(0, item.candidate.frame_number - 2)
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames: dict[int, np.ndarray] = {}
    try:
        for frame_number in range(start, item.candidate.frame_number + 3):
            ok, frame = capture.read()
            if not ok:
                break
            frames[frame_number] = frame
    finally:
        capture.release()
    strips: list[np.ndarray] = []
    for relative in range(-2, 3):
        frame_number = max(0, item.candidate.frame_number + relative)
        frame = frames.get(frame_number)
        if frame is None:
            raise RuntimeError(
                f"Could not decode review frame {frame_number} from {item.source}"
            )
        label = "PEAK" if relative == 0 else f"{relative:+d} frame"
        strips.append(_label_frame(frame, label, relative == 0))
    preview = np.hstack(strips)
    temporary = destination.with_name(f".{destination.name}.tmp.jpg")
    if not cv2.imwrite(str(temporary), preview, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise RuntimeError(f"Could not write review preview: {destination}")
    os.replace(temporary, destination)
    return destination


def review_candidates(
    path: Path,
    config: Config,
    *,
    labels_path: Path | None = None,
    open_previews: bool = True,
    include_reviewed: bool = False,
    input_func: Callable[[str], str] = input,
    opener: Callable[[str], object] = webbrowser.open,
) -> tuple[Path, dict[str, int]]:
    root = path.resolve()
    labels_path = labels_path or root / "labels" / "review.json"
    document: dict[str, object] = (
        json.loads(labels_path.read_text())
        if labels_path.is_file()
        else {"schema_version": 1, "run": str(root), "items": {}}
    )
    labels = document.setdefault("items", {})
    if not isinstance(labels, dict):
        raise TypeError(f"Invalid review labels: {labels_path}")
    items = discover_review_items(root, config)
    pending = items if include_reviewed else [item for item in items if item.key not in labels]
    for index, item in enumerate(pending, 1):
        preview = build_review_preview(item)
        print(
            f"[{index}/{len(pending)}] {item.video_name}  {item.candidate.event_id}  "
            f"{item.candidate.time:.3f}s\n{preview.resolve()}"
        )
        if open_previews:
            opener(preview.resolve().as_uri())
        while True:
            answer = input_func("Lightning channel? [y]es/[n]o/[u]ncertain/[q]uit: ").strip().lower()
            if answer in LABELS or answer == "q":
                break
            print("Please enter y, n, u, or q.")
        if answer == "q":
            break
        labels[item.key] = {
            "video_run": item.video_run.name,
            "video": item.video_name,
            "event_id": item.candidate.event_id,
            "frame_number": item.candidate.frame_number,
            "time": item.candidate.time,
            "rank": item.candidate.rank,
            "label": LABELS[answer],
            "reviewed_at": _utc_now(),
            "preview": str(preview.resolve()),
            "metrics": asdict(item.candidate),
        }
        document["updated_at"] = _utc_now()
        _atomic_json(labels_path, document)
    counts = {
        label: sum(
            isinstance(value, dict) and value.get("label") == label
            for value in labels.values()
        )
        for label in LABELS.values()
    }
    counts["pending"] = sum(item.key not in labels for item in items)
    document["updated_at"] = _utc_now()
    _atomic_json(labels_path, document)
    return labels_path, counts
