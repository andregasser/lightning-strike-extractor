from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from .config import Config
from .detection import detect_flashes, rank_event_frames
from .models import CandidateFrame, FlashEvent
from .probe import probe_video
from .progress import ProgressReporter

MINIMUM_FREE_BYTES = 100 * 1024 * 1024


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _source_id(video: Path) -> str:
    stat = video.stat()
    identity = f"{video.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode()
    return hashlib.sha256(identity).hexdigest()[:8]


def run_identity(
    video: Path,
    source: dict[str, object],
    config: Config,
    start_seconds: float,
    end_seconds: float | None,
) -> str:
    value = {
        "source_id": _source_id(video),
        "source_size": source["size_bytes"],
        "config": config.as_dict(),
        "start_seconds": start_seconds,
        "end_seconds": end_seconds,
        "tool_version": __version__,
    }
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()[:10]


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def _write_json(path: Path, value: object) -> None:
    _atomic_write_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _read_json(path: Path) -> object:
    return json.loads(path.read_text())


def _write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    materialized = list(rows)
    temporary = path.with_name(f".{path.name}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not materialized:
        temporary.write_text("")
    else:
        with temporary.open("w", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=list(materialized[0]))
            writer.writeheader()
            writer.writerows(materialized)
    os.replace(temporary, path)


def _update_state(run: Path, **changes: object) -> dict[str, object]:
    path = run / "run.json"
    state = dict(_read_json(path)) if path.exists() else {}
    state.update(changes)
    state["updated_at"] = _utc_now()
    _write_json(path, state)
    return state


def _ensure_disk_space(path: Path, required: int = MINIMUM_FREE_BYTES) -> None:
    free = shutil.disk_usage(path).free
    required = max(required, MINIMUM_FREE_BYTES)
    if free < required:
        raise RuntimeError(
            f"Insufficient free disk space at {path}: {free / 1024 / 1024:.0f} MiB "
            f"available, at least {required / 1024 / 1024:.0f} MiB required"
        )


def _events_from_json(path: Path) -> list[FlashEvent]:
    return [FlashEvent(**row) for row in _read_json(path)]  # type: ignore[arg-type]


def _candidates_from_json(path: Path) -> list[CandidateFrame]:
    return [CandidateFrame(**row) for row in _read_json(path)]  # type: ignore[arg-type]


def export_stills(
    video: Path,
    candidates: list[CandidateFrame],
    output: Path,
    config: Config,
    resume: bool = False,
) -> int:
    output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {video}")
    thumbnails: list[np.ndarray] = []
    selected = candidates[: config.export.top]
    reporter = ProgressReporter("export", "frames")
    exported = 0
    for index, candidate in enumerate(selected, 1):
        filename = output / f"{candidate.rank:04d}_{candidate.time:09.2f}s.jpg"
        frame = cv2.imread(str(filename)) if resume and filename.exists() else None
        if frame is None:
            capture.set(cv2.CAP_PROP_POS_FRAMES, candidate.frame_number)
            ok, frame = capture.read()
            if not ok:
                reporter.update(index, len(selected))
                continue
            temporary = filename.with_name(f".{filename.name}.tmp.jpg")
            if not cv2.imwrite(
                str(temporary), frame, [cv2.IMWRITE_JPEG_QUALITY, config.export.jpeg_quality]
            ):
                raise RuntimeError(f"Could not write still image: {filename}")
            os.replace(temporary, filename)
        exported += 1
        width = 640
        height = round(frame.shape[0] * width / frame.shape[1])
        thumb = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        cv2.rectangle(thumb, (0, 0), (300, 38), (0, 0, 0), -1)
        cv2.putText(
            thumb,
            f"#{candidate.rank}  {candidate.time:.2f}s",
            (8, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        thumbnails.append(thumb)
        reporter.update(index, len(selected))
    capture.release()
    if selected:
        reporter.update(len(selected), len(selected), force=True)
    columns = config.export.contact_sheet_columns
    if thumbnails:
        blank = np.zeros_like(thumbnails[0])
        while len(thumbnails) % columns:
            thumbnails.append(blank.copy())
        sheet = np.vstack(
            [
                np.hstack(thumbnails[index : index + columns])
                for index in range(0, len(thumbnails), columns)
            ]
        )
        contact_sheet = output.parent / "contact-sheet.jpg"
        temporary = contact_sheet.with_name(f".{contact_sheet.name}.tmp.jpg")
        if not cv2.imwrite(str(temporary), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94]):
            raise RuntimeError(f"Could not write contact sheet: {contact_sheet}")
        os.replace(temporary, contact_sheet)
    return exported


def analyze(
    video: Path,
    runs_root: Path,
    config: Config,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
    resume: bool = False,
) -> Path:
    video = video.resolve()
    source = probe_video(video)
    fps = float(source["video"]["fps"])  # type: ignore[index]
    if fps <= 0:
        raise RuntimeError("The video frame rate could not be determined")
    identity = run_identity(video, source, config, start_seconds, end_seconds)
    run = runs_root / f"{video.stem}-{_source_id(video)}-{identity}"
    results = run / "results"
    exports = run / "exports"
    cache = run / "cache"
    existed = run.exists()
    if existed and not resume:
        raise RuntimeError(f"Run already exists: {run}. Use --resume to continue it")
    results.mkdir(parents=True, exist_ok=True)
    exports.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    video_metadata = source["video"]  # type: ignore[assignment]
    export_reserve = (
        config.export.top
        * int(video_metadata.get("width", 1920))  # type: ignore[union-attr]
        * int(video_metadata.get("height", 1080))  # type: ignore[union-attr]
    )
    _ensure_disk_space(run, export_reserve)

    state_path = run / "run.json"
    if not state_path.exists():
        _write_json(
            state_path,
            {
                "status": "pending",
                "phase": "pending",
                "created_at": _utc_now(),
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
                "source_id": _source_id(video),
                "run_id": identity,
                "tool_version": __version__,
            },
        )
        _write_json(run / "source.json", source)
        _write_json(run / "config.json", config.as_dict())
    elif resume:
        state = dict(_read_json(state_path))
        if state.get("run_id") != identity:
            raise RuntimeError(f"Run identity mismatch: {run}")
        if state.get("status") == "complete":
            return run

    try:
        events_path = results / "events.json"
        if resume and events_path.exists():
            events = _events_from_json(events_path)
        else:
            _update_state(run, status="running", phase="flash-scan", error=None)
            start_frame = round(start_seconds * fps)
            source_end_frame = round(float(source["duration_seconds"]) * fps)
            end_frame = (
                min(round(end_seconds * fps), source_end_frame)
                if end_seconds is not None
                else source_end_frame
            )
            flash_reporter = ProgressReporter("flash scan", "frames")
            events = detect_flashes(
                video,
                fps,
                config,
                start_seconds,
                end_seconds,
                progress=flash_reporter.update,
                checkpoint_dir=cache / "flash-scan",
                resume=resume,
            )
            flash_reporter.update(end_frame - start_frame, end_frame - start_frame, force=True)
            _write_json(events_path, [event.as_dict() for event in events])
            _write_csv(results / "events.csv", (event.as_dict() for event in events))

        candidates_path = results / "candidates.json"
        if resume and candidates_path.exists():
            candidates = _candidates_from_json(candidates_path)
        else:
            _update_state(run, status="running", phase="channel-ranking", event_count=len(events))
            channel_checkpoint = cache / "channel-ranking.json"
            completed_events = 0
            partial_candidates: list[CandidateFrame] = []
            if resume and channel_checkpoint.exists():
                checkpoint_value = dict(_read_json(channel_checkpoint))
                completed_events = int(checkpoint_value["completed_events"])
                partial_candidates = [
                    CandidateFrame(**row)
                    for row in checkpoint_value["candidates"]  # type: ignore[arg-type]
                ]

            def save_channel_checkpoint(
                completed: int, current_candidates: list[CandidateFrame]
            ) -> None:
                _write_json(
                    channel_checkpoint,
                    {
                        "completed_events": completed,
                        "candidates": [candidate.as_dict() for candidate in current_candidates],
                    },
                )

            channel_reporter = ProgressReporter(
                "channel ranking", "events", initial_completed=completed_events
            )
            candidates = rank_event_frames(
                video,
                fps,
                events,
                config,
                progress=channel_reporter.update,
                completed_events=completed_events,
                existing_candidates=partial_candidates,
                checkpoint=save_channel_checkpoint,
            )
            channel_reporter.update(len(events), len(events), force=True)
            _write_json(candidates_path, [item.as_dict() for item in candidates])
            _write_csv(results / "candidates.csv", (item.as_dict() for item in candidates))

        _update_state(run, status="running", phase="export", candidate_count=len(candidates))
        still_count = export_stills(video, candidates, exports / "stills", config, resume=resume)
        _write_json(
            results / "summary.json",
            {
                "events": len(events),
                "candidate_frames": len(candidates),
                "exported_stills": still_count,
                "best_geometry_score": candidates[0].geometry_score if candidates else 0.0,
            },
        )
        _update_state(run, status="complete", phase="complete", completed_at=_utc_now(), error=None)
        return run
    except KeyboardInterrupt:
        _update_state(
            run, status="interrupted", interrupted_at=_utc_now(), error="Interrupted by user"
        )
        raise
    except Exception as error:
        _update_state(
            run,
            status="failed",
            failed_at=_utc_now(),
            error=f"{type(error).__name__}: {error}",
        )
        raise
