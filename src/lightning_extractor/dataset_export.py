from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2


@dataclass(frozen=True, slots=True)
class CandidateFrame:
    rank: int
    event_id: str
    frame_number: int
    time: float
    geometry_score: float
    line_segments: int
    bright_area: float
    channel_length: float = 0.0
    channel_strength: float = 0.0
    channel_luminance: float = 0.0
    branch_points: int = 0
    channel_thickness: float = 0.0
    frame_quality: float = 0.0
    multiframe_support: float = 0.0
    multiframe_quality: float = 0.0
    peak_multiframe_support: float = 0.0
    channel_template_frame_number: int = -1
    background_frame_number: int = -1


@dataclass(frozen=True, slots=True)
class FrameRequest:
    frame_number: int
    event_id: str
    relative_frame: int
    candidate: CandidateFrame
    review_label: str | None


def _video_runs(path: Path) -> list[Path]:
    root = path.resolve()
    if (root / "results" / "candidates.json").is_file():
        return [root]
    videos = root / "videos"
    if videos.is_dir():
        runs = sorted(
            run for run in videos.iterdir() if (run / "results" / "candidates.json").is_file()
        )
        if runs:
            return runs
    raise ValueError(f"No analysis video runs found below: {root}")


def _review_labels(root: Path) -> dict[str, dict]:
    labels_path = root.resolve() / "labels" / "review.json"
    if not labels_path.is_file():
        return {}
    document = json.loads(labels_path.read_text())
    items = document.get("items", {}) if isinstance(document, dict) else {}
    return {key: value for key, value in items.items() if isinstance(value, dict)}


def _candidate_score(candidate: CandidateFrame) -> tuple[float, float, float]:
    return (
        candidate.frame_quality or candidate.geometry_score,
        candidate.multiframe_support,
        candidate.geometry_score,
    )


def select_frame_requests(
    candidates: list[CandidateFrame],
    *,
    run_name: str,
    labels: dict[str, dict],
    max_events: int,
    context_frames: int,
) -> list[FrameRequest]:
    events: dict[str, list[CandidateFrame]] = defaultdict(list)
    for candidate in candidates:
        events[candidate.event_id].append(candidate)
    winners = [max(rows, key=_candidate_score) for rows in events.values()]
    winners.sort(key=_candidate_score, reverse=True)
    requests: list[FrameRequest] = []
    for winner in winners[:max_events]:
        reviewed = labels.get(f"{run_name}:{winner.event_id}", {})
        label = (
            str(reviewed["label"])
            if reviewed.get("frame_number") == winner.frame_number and reviewed.get("label")
            else None
        )
        for relative in range(-context_frames, context_frames + 1):
            frame_number = winner.frame_number + relative
            if frame_number < 0:
                continue
            requests.append(
                FrameRequest(
                    frame_number=frame_number,
                    event_id=winner.event_id,
                    relative_frame=relative,
                    candidate=winner,
                    review_label=label if relative == 0 else None,
                )
            )
    return requests


def _source_id(run: Path, source: dict) -> str:
    run_path = run / "run.json"
    if run_path.is_file():
        state = json.loads(run_path.read_text())
        if isinstance(state, dict) and state.get("source_id"):
            return str(state["source_id"])
    identity = str(Path(str(source["path"])).resolve()).encode()
    return hashlib.sha256(identity).hexdigest()[:8]


def _decode_requested_frames(
    video: Path, frame_numbers: set[int]
) -> tuple[dict[int, object], set[int]]:
    if not frame_numbers:
        return {}, set()
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open source video: {video}")
    total_frames = round(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    out_of_range = {
        frame_number
        for frame_number in frame_numbers
        if frame_number < 0 or (total_frames > 0 and frame_number >= total_frames)
    }
    valid_frames = frame_numbers - out_of_range
    if not valid_frames:
        capture.release()
        return {}, out_of_range
    first, last = min(valid_frames), max(valid_frames)
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    decoded: dict[int, object] = {}
    try:
        for frame_number in range(first, last + 1):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not decode frame {frame_number} from {video}")
            if frame_number in valid_frames:
                decoded[frame_number] = frame
    finally:
        capture.release()
    return decoded, out_of_range


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_run_frames(
    run: Path,
    destination: Path,
    labels: dict[str, dict],
    *,
    max_events: int,
    context_frames: int,
    jpeg_quality: int,
    seen_frames: set[tuple[str, int]],
) -> tuple[list[dict], dict]:
    source = json.loads((run / "source.json").read_text())
    if not isinstance(source, dict) or not source.get("path"):
        raise ValueError(f"Invalid source metadata: {run / 'source.json'}")
    video = Path(str(source["path"])).expanduser().resolve()
    if not video.is_file():
        raise ValueError(f"Source video does not exist: {video}")
    candidates_data = json.loads((run / "results" / "candidates.json").read_text())
    if not isinstance(candidates_data, list):
        raise TypeError(f"Candidate data must be a list: {run}")
    candidates = [CandidateFrame(**row) for row in candidates_data]
    requests = select_frame_requests(
        candidates,
        run_name=run.name,
        labels=labels,
        max_events=max_events,
        context_frames=context_frames,
    )
    by_frame: dict[int, list[FrameRequest]] = defaultdict(list)
    for request in requests:
        by_frame[request.frame_number].append(request)
    source_id = _source_id(run, source)
    duplicate_frames = {
        frame_number for frame_number in by_frame if (source_id, frame_number) in seen_frames
    }
    for frame_number in duplicate_frames:
        del by_frame[frame_number]
    frames, out_of_range = _decode_requested_frames(video, set(by_frame))
    for frame_number in out_of_range:
        del by_frame[frame_number]
    image_dir = destination / "images" / source_id
    image_dir.mkdir(parents=True, exist_ok=True)
    fps = float(source.get("video", {}).get("fps", 0.0))
    rows: list[dict] = []
    for frame_number in sorted(by_frame):
        frame = frames[frame_number]
        filename = f"{run.name}_frame_{frame_number:012d}.jpg"
        image_path = image_dir / filename
        if not cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]):
            raise RuntimeError(f"Could not write handoff frame: {image_path}")
        seen_frames.add((source_id, frame_number))
        provenance = by_frame[frame_number]
        rows.append(
            {
                "file_name": image_path.relative_to(destination).as_posix(),
                "sha256": _file_sha256(image_path),
                "source_id": source_id,
                "source_video": str(video),
                "video_run": run.name,
                "frame_number": frame_number,
                "time": frame_number / fps if fps > 0 else None,
                "roles": [
                    {
                        "role": "event_peak" if item.relative_frame == 0 else "event_context",
                        "event_id": item.event_id,
                        "relative_frame": item.relative_frame,
                        "review_label": item.review_label,
                        "candidate": asdict(item.candidate),
                    }
                    for item in provenance
                ],
            }
        )
    return rows, {
        "source_id": source_id,
        "source_video": str(video),
        "video_run": run.name,
        "frames": len(rows),
        "events": len({request.event_id for request in requests}),
        "duplicate_frames_skipped": len(duplicate_frames),
        "out_of_range_frames_skipped": len(out_of_range),
    }


def export_frame_handoff(
    runs_root: Path,
    output: Path,
    *,
    max_events_per_video: int = 100,
    context_frames: int = 2,
    jpeg_quality: int = 95,
) -> dict:
    if max_events_per_video <= 0:
        raise ValueError("max_events_per_video must be positive")
    if context_frames < 0:
        raise ValueError("context_frames cannot be negative")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100")
    output = output.resolve()
    if output.exists():
        raise ValueError(f"Refusing to overwrite existing dataset: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    labels = _review_labels(runs_root)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        rows: list[dict] = []
        sources: list[dict] = []
        seen_frames: set[tuple[str, int]] = set()
        for run in _video_runs(runs_root):
            run_rows, source = _write_run_frames(
                run,
                staging,
                labels,
                max_events=max_events_per_video,
                context_frames=context_frames,
                jpeg_quality=jpeg_quality,
                seen_frames=seen_frames,
            )
            rows.extend(run_rows)
            sources.append(source)
        if not rows:
            raise ValueError(f"No candidate frames found below: {runs_root.resolve()}")
        manifest = {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "runs_root": str(runs_root.resolve()),
            "selection": {
                "max_events_per_video": max_events_per_video,
                "context_frames": context_frames,
                "jpeg_quality": jpeg_quality,
            },
            "sources": sources,
            "frames": rows,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
        )
        os.replace(staging, output)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export selected frames and provenance from analysis runs")
    parser.add_argument("runs", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-events-per-video", type=int, default=100)
    parser.add_argument("--context-frames", type=int, default=2)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    args = parser.parse_args(argv)
    manifest = export_frame_handoff(
        args.runs,
        args.output,
        max_events_per_video=args.max_events_per_video,
        context_frames=args.context_frames,
        jpeg_quality=args.jpeg_quality,
    )
    print(
        f"sources: {len(manifest['sources'])}\n"
        f"frames: {len(manifest['frames'])}\n"
        f"output: {args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
