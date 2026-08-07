from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from .config import Config
from .detection import detect_flashes, frame_channel_metrics, rank_event_frames
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


def resolve_run_path(
    video: Path,
    runs_root: Path,
    config: Config,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
    source: dict[str, object] | None = None,
) -> Path:
    video = video.resolve()
    source = source or probe_video(video)
    identity = run_identity(video, source, config, start_seconds, end_seconds)
    return runs_root / f"{video.stem}-{_source_id(video)}-{identity}"


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


def has_visible_channel_peak(row: CandidateFrame, config: Config) -> bool:
    clean_local_channel = (
        row.channel_length >= config.export.minimum_channel_length
        and row.line_segments >= config.export.minimum_clean_line_segments
        and row.bright_area <= config.export.maximum_clean_channel_bright_area
    )
    return (
        row.geometry_score >= config.export.minimum_peak_geometry_score
        or row.channel_length >= config.export.minimum_peak_channel_length
        or clean_local_channel
    )


def select_export_candidates(
    candidates: list[CandidateFrame], config: Config
) -> list[CandidateFrame]:
    def has_confirmed_geometry(row: CandidateFrame) -> bool:
        base_geometry = (
            row.geometry_score >= config.export.minimum_geometry_score
            and row.channel_length >= config.export.minimum_channel_length
            and row.line_segments >= config.export.minimum_line_segments
            and row.channel_strength >= config.export.minimum_channel_strength
            and 0.0 < row.channel_thickness <= config.export.maximum_channel_thickness
        )
        # A merely thresholded cloud edge can be thin and connected. Require
        # additional evidence that the component is either geometrically very
        # strong, genuinely long, or isolated from a diffuse frame-wide flash.
        return base_geometry and (
            row.geometry_score >= config.export.minimum_strong_geometry_score
            or row.channel_length >= config.export.minimum_long_channel_length
            or row.bright_area <= config.export.maximum_clean_channel_bright_area
        )

    def is_qualified(row: CandidateFrame, confirmed_frames: set[int]) -> bool:
        # A bright or saturated return stroke may use the mask of an adjacent
        # frame, but temporal overlap alone must never manufacture geometry.
        # The exact template frame therefore has to pass the same strict
        # channel test on its own.
        return has_confirmed_geometry(row) or (
            row.channel_template_frame_number != row.frame_number
            and row.channel_template_frame_number in confirmed_frames
        )

    if config.export.one_frame_per_event:
        events: dict[str, list[CandidateFrame]] = {}
        for candidate in candidates:
            events.setdefault(candidate.event_id, []).append(candidate)
        selected = []
        for rows in events.values():
            confirmed_frames = {
                row.frame_number for row in rows if has_confirmed_geometry(row)
            }
            qualified_rows = [
                row
                for row in rows
                if is_qualified(row, confirmed_frames)
            ]
            if not qualified_rows:
                continue
            best_geometry = max(row.geometry_score for row in qualified_rows)
            plausible = [
                row
                for row in qualified_rows
                if row.geometry_score
                >= best_geometry * config.export.minimum_winner_geometry_ratio
            ]
            winner = max(
                plausible,
                key=lambda row: (
                    row.frame_quality or row.geometry_score,
                    row.multiframe_support,
                    row.geometry_score,
                ),
            )
            if has_visible_channel_peak(winner, config):
                selected.append(winner)
        selected.sort(
            key=lambda row: row.multiframe_quality
            or row.frame_quality
            or row.geometry_score,
            reverse=True,
        )
        return selected[: config.export.top]

    selected: list[CandidateFrame] = []
    confirmed_frames = {
        row.frame_number for row in candidates if has_confirmed_geometry(row)
    }
    for candidate in candidates:
        if not is_qualified(candidate, confirmed_frames) or not has_visible_channel_peak(
            candidate, config
        ):
            continue
        selected.append(candidate)
        if len(selected) >= config.export.top:
            break
    return selected


def _contact_thumbnail(
    frame: np.ndarray,
    label: str,
    details: str = "",
    peak: bool = False,
) -> np.ndarray:
    width = 640
    height = round(frame.shape[0] * width / frame.shape[1])
    thumbnail = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(thumbnail, (0, 0), (width, 62), (0, 0, 0), -1)
    cv2.putText(
        thumbnail,
        label,
        (8, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if details:
        cv2.putText(
            thumbnail,
            details,
            (8, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
    if peak:
        cv2.rectangle(
            thumbnail,
            (2, 2),
            (width - 3, height - 3),
            (0, 255, 255),
            5,
        )
    return thumbnail


def _channel_overlay(frame: np.ndarray, channel_mask: np.ndarray | None) -> np.ndarray:
    overlay = frame.copy()
    if channel_mask is None or not np.any(channel_mask):
        return overlay
    mask = cv2.resize(
        channel_mask,
        (frame.shape[1], frame.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    selected = mask > 0
    colour = np.zeros_like(frame)
    colour[:, :] = (255, 0, 255)
    overlay[selected] = cv2.addWeighted(frame, 0.25, colour, 0.75, 0)[selected]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
    return overlay


def _append_contact_sequence(
    capture: cv2.VideoCapture,
    frame: np.ndarray,
    candidate: CandidateFrame,
    config: Config,
    thumbnails: list[np.ndarray],
    overlay: np.ndarray | None = None,
) -> None:
    context = max(0, config.export.contact_sheet_context_frames)
    stride = max(1, config.export.contact_sheet_context_stride)
    total_frames = round(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    for relative in range(-context, context + 1):
        target = candidate.frame_number + relative * stride
        context_frame: np.ndarray | None
        if relative == 0:
            context_frame = frame
        elif target < 0 or (total_frames > 0 and target >= total_frames):
            context_frame = None
        else:
            capture.set(cv2.CAP_PROP_POS_FRAMES, target)
            ok, context_frame = capture.read()
            if not ok:
                context_frame = None
        if context_frame is None:
            context_frame = np.zeros_like(frame)
        if relative == 0:
            label = f"PEAK #{candidate.rank}  {candidate.event_id}  {candidate.time:.3f}s"
            details = (
                f"Q {candidate.frame_quality:.0f}  G {candidate.geometry_score:.0f}  "
                f"L {candidate.channel_length:.0f}  Y {candidate.channel_luminance:.1f}  "
                f"S {candidate.channel_strength:.1f}  "
                f"B {candidate.branch_points}  "
                f"MF {max(candidate.multiframe_support, candidate.peak_multiframe_support):.2f}"
            )
        else:
            time = candidate.time + relative * stride / fps if fps > 0 else candidate.time
            label = f"{relative * stride:+d}f  {time:.3f}s"
            details = ""
        thumbnails.append(
            _contact_thumbnail(context_frame, label, details, peak=relative == 0)
        )
        if relative == 0 and config.export.contact_sheet_include_overlay:
            overlay_frame = overlay if overlay is not None else np.zeros_like(frame)
            thumbnails.append(
                _contact_thumbnail(
                    overlay_frame,
                    "DETECTED CHANNEL",
                    "Magenta mask with yellow outline",
                )
            )


def _write_jpeg(path: Path, frame: np.ndarray, quality: int) -> None:
    temporary = path.with_name(f".{path.name}.tmp.jpg")
    if not cv2.imwrite(str(temporary), frame, [cv2.IMWRITE_JPEG_QUALITY, quality]):
        raise RuntimeError(f"Could not write still image: {path}")
    os.replace(temporary, path)


def _event_export_directory(output: Path, candidate: CandidateFrame) -> Path:
    return output.parent / "events" / f"{candidate.event_id}_{candidate.time:010.3f}s"


def _export_event_frames(
    capture: cv2.VideoCapture,
    peak_frame: np.ndarray,
    candidate: CandidateFrame,
    destination: Path,
    config: Config,
    resume: bool,
) -> None:
    if not config.export.event_frames_enabled:
        return
    destination.mkdir(parents=True, exist_ok=True)
    context = max(0, config.export.contact_sheet_context_frames)
    stride = max(1, config.export.contact_sheet_context_stride)
    total_frames = round(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    for relative in range(-context, context + 1):
        label = "peak" if relative == 0 else f"{relative * stride:+03d}"
        path = destination / f"frame_{label}.jpg"
        if resume and path.exists():
            continue
        target = candidate.frame_number + relative * stride
        if total_frames > 0:
            target = min(max(target, 0), total_frames - 1)
        if relative == 0:
            frame = peak_frame
        else:
            capture.set(cv2.CAP_PROP_POS_FRAMES, target)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(
                    f"Could not read context frame {target} for {candidate.event_id}"
                )
        _write_jpeg(path, frame, config.export.jpeg_quality)


def _export_slow_motion(
    video: Path,
    candidate: CandidateFrame,
    destination: Path,
    config: Config,
    resume: bool,
) -> None:
    if not config.export.slow_motion_enabled:
        return
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "slow-motion.mp4"
    if resume and output.exists():
        return
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError("ffmpeg is required for slow-motion event exports")
    before = max(0.0, config.export.slow_motion_before_seconds)
    after = max(0.0, config.export.slow_motion_after_seconds)
    duration = before + after
    factor = max(1.0, config.export.slow_motion_factor)
    fps = max(1, config.export.slow_motion_output_fps)
    start = max(0.0, candidate.time - before)
    temporary = output.with_name(f".{output.name}.tmp.mp4")
    command = [
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.6f}",
        "-t",
        f"{duration:.6f}",
        "-i",
        str(video),
        "-map",
        "0:v:0",
        "-an",
        "-dn",
        "-sn",
        "-vf",
        f"setpts={factor:.6f}*PTS,fps={fps}",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(config.export.slow_motion_crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            result.stderr.strip() or f"Could not export slow motion for {candidate.event_id}"
        )
    os.replace(temporary, output)


def export_stills(
    video: Path,
    candidates: list[CandidateFrame],
    output: Path,
    config: Config,
    resume: bool = False,
    progress_mode: str = "auto",
    label: str | None = None,
) -> int:
    output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {video}")
    thumbnails: list[np.ndarray] = []
    selected = select_export_candidates(candidates, config)
    reporter = ProgressReporter("export", "frames", mode=progress_mode, label=label)
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
            _write_jpeg(filename, frame, config.export.jpeg_quality)
        exported += 1
        event_directory = _event_export_directory(output, candidate)
        _export_event_frames(
            capture,
            frame,
            candidate,
            event_directory,
            config,
            resume,
        )
        _export_slow_motion(
            video,
            candidate,
            event_directory,
            config,
            resume,
        )
        overlay = None
        if (
            config.export.contact_sheet_include_overlay
            and candidate.background_frame_number >= 0
        ):
            capture.set(cv2.CAP_PROP_POS_FRAMES, candidate.background_frame_number)
            ok, background = capture.read()
            if ok:
                template = frame
                if candidate.channel_template_frame_number >= 0:
                    capture.set(
                        cv2.CAP_PROP_POS_FRAMES,
                        candidate.channel_template_frame_number,
                    )
                    template_ok, template_frame = capture.read()
                    if template_ok:
                        template = template_frame
                metrics = frame_channel_metrics(background, template, config)
                overlay = _channel_overlay(frame, metrics.channel_mask)
        _append_contact_sequence(
            capture,
            frame,
            candidate,
            config,
            thumbnails,
            overlay,
        )
        reporter.update(index, len(selected))
    capture.release()
    if selected:
        reporter.update(len(selected), len(selected), force=True)
    context = max(0, config.export.contact_sheet_context_frames)
    columns = 2 * context + 1 if context else config.export.contact_sheet_columns
    if config.export.contact_sheet_include_overlay:
        columns += 1
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
    progress_mode: str = "auto",
    label: str | None = None,
) -> Path:
    video = video.resolve()
    source = probe_video(video)
    fps = float(source["video"]["fps"])  # type: ignore[index]
    if fps <= 0:
        raise RuntimeError("The video frame rate could not be determined")
    identity = run_identity(video, source, config, start_seconds, end_seconds)
    run = resolve_run_path(video, runs_root, config, start_seconds, end_seconds, source)
    label = label or video.name
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
            flash_reporter = ProgressReporter(
                "flash scan", "frames", mode=progress_mode, label=label
            )
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
                "channel ranking",
                "events",
                initial_completed=completed_events,
                mode=progress_mode,
                label=label,
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
        still_count = export_stills(
            video,
            candidates,
            exports / "stills",
            config,
            resume=resume,
            progress_mode=progress_mode,
            label=label,
        )
        _write_json(
            results / "summary.json",
            {
                "events": len(events),
                "candidate_frames": len(candidates),
                "exported_stills": still_count,
                "best_geometry_score": (
                    max(item.geometry_score for item in candidates) if candidates else 0.0
                ),
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
