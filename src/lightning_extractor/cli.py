from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

from . import __version__
from .batch import VideoJob, deduplicate_jobs, list_runs, load_manifest, run_batch
from .config import load_config
from .detector import detect_image, render_detections, write_detection_json
from .discovery import DiscoveryResult, discover_inputs
from .probe import ProbeError, probe_video
from .review import review_candidates


def _add_discovery_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--recursive", action="store_true", help="Search directories recursively")
    parser.add_argument("--include", action="append", default=[], help="Include matching filenames")
    parser.add_argument("--exclude", action="append", default=[], help="Exclude matching filenames")
    parser.add_argument(
        "--follow-symlinks", action="store_true", help="Follow directory symlinks while searching"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lightning", description="Extract lightning strikes from videos"
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser("inspect", help="Show metadata for one or more videos")
    inspect.add_argument("inputs", nargs="+", type=Path)
    _add_discovery_options(inspect)

    analyze = commands.add_parser("analyze", help="Analyze files, directories, or a manifest")
    analyze.add_argument("inputs", nargs="*", type=Path)
    analyze.add_argument("--manifest", type=Path, help="TOML batch manifest")
    _add_discovery_options(analyze)
    analyze.add_argument("--dry-run", action="store_true", help="Discover inputs without analyzing")
    analyze.add_argument("--output", type=Path, help="Root directory for video runs and batches")
    analyze.add_argument("--config", type=Path, help="Default TOML analysis configuration")
    analyze.add_argument("--start", type=float, help="Start time in seconds for all inputs")
    analyze.add_argument("--end", type=float, help="End time in seconds for all inputs")
    analyze.add_argument("--top", type=int, help="Number of stills to export per video")
    analyze.add_argument("--max-events", type=int, help="Maximum events to analyze per video")
    analyze.add_argument("--jobs", type=int, help="Maximum videos to process concurrently")
    analyze.add_argument(
        "--resume",
        action="store_true",
        default=None,
        help="Resume incomplete runs and skip completed runs",
    )
    analyze.add_argument(
        "--fail-fast",
        action="store_true",
        default=None,
        help="Stop scheduling work after the first failed video",
    )
    analyze.add_argument(
        "--progress",
        choices=("auto", "interactive", "plain", "json", "quiet"),
        help="Progress output format",
    )

    runs = commands.add_parser("runs", help="Inspect analysis run state")
    run_commands = runs.add_subparsers(dest="runs_command", required=True)
    run_list = run_commands.add_parser("list", help="List video runs")
    run_list.add_argument("--output", type=Path, default=Path("runs"))
    run_list.add_argument(
        "--status", choices=("pending", "running", "interrupted", "failed", "complete")
    )
    run_list.add_argument("--json", action="store_true")
    run_show = run_commands.add_parser("show", help="Show a run or batch JSON state")
    run_show.add_argument("path", type=Path)

    review = commands.add_parser("review", help="Manually label detected lightning channels")
    review.add_argument("path", type=Path, help="Runs root or one video run")
    review.add_argument("--config", type=Path, help="TOML selection configuration")
    review.add_argument("--labels", type=Path, help="Review JSON output path")
    review.add_argument(
        "--scope",
        choices=("selected", "all-events"),
        default="selected",
        help="Review exported selections or the best frame from every raw event",
    )
    review.add_argument(
        "--no-open",
        action="store_true",
        help="Print preview paths without opening the system image viewer",
    )
    review.add_argument(
        "--include-reviewed",
        action="store_true",
        help="Review and overwrite already labelled events",
    )

    detector = commands.add_parser("detector", help="Detect lightning with the released model")
    detector_commands = detector.add_subparsers(dest="detector_command", required=True)
    detect = detector_commands.add_parser("detect", help="Detect lightning in one image")
    detect.add_argument("image", type=Path)
    detect.add_argument("--output", type=Path, help="Write detection JSON")
    detect.add_argument("--preview", type=Path, help="Write image with detection boxes")
    return parser


def _print_discovery(result: DiscoveryResult, videos: list[Path], duplicates: int = 0) -> None:
    print(f"videos: {len(videos)}")
    print(f"ignored: {len(result.ignored)}")
    print(f"duplicates: {len(result.duplicates) + duplicates}")
    for video in videos:
        print(video)


def _validate_ranges(jobs: list[VideoJob]) -> None:
    for job in jobs:
        if job.start_seconds < 0:
            raise ValueError(f"Start time cannot be negative: {job.path}")
        if job.end_seconds is not None and job.end_seconds <= job.start_seconds:
            raise ValueError(f"End time must be greater than start time: {job.path}")


def _analyze(args: argparse.Namespace) -> int:
    if not args.inputs and args.manifest is None:
        raise ValueError("Provide at least one input or --manifest")
    fallback = load_config(args.config)
    jobs: list[VideoJob] = []
    discovery = DiscoveryResult()
    manifest = load_manifest(args.manifest, fallback) if args.manifest else None
    if manifest:
        jobs.extend(manifest.jobs)
        for found in manifest.discoveries:
            discovery.ignored.extend(found.ignored)
            discovery.duplicates.extend(found.duplicates)
    if args.inputs:
        discovery = discover_inputs(
            args.inputs,
            recursive=args.recursive,
            includes=args.include,
            excludes=args.exclude,
            follow_symlinks=args.follow_symlinks,
        )
        jobs.extend(VideoJob(video, copy.deepcopy(fallback)) for video in discovery.videos)

    if args.start is not None:
        for job in jobs:
            job.start_seconds = args.start
    if args.end is not None:
        for job in jobs:
            job.end_seconds = args.end
    for job in jobs:
        if args.top is not None:
            job.config.export.top = args.top
        if args.max_events is not None:
            job.config.analysis.max_events = args.max_events
    _validate_ranges(jobs)
    jobs, duplicate_jobs = deduplicate_jobs(jobs)
    if not jobs:
        _print_discovery(discovery, [], len(duplicate_jobs))
        return 3
    if args.dry_run:
        _print_discovery(discovery, [job.path for job in jobs], len(duplicate_jobs))
        return 0

    defaults = manifest.defaults if manifest else None
    output = args.output or (defaults.output if defaults else None) or Path("runs")
    worker_count = args.jobs or (defaults.jobs if defaults else None) or 1
    resume = args.resume if args.resume is not None else (defaults.resume if defaults else False)
    fail_fast = (
        args.fail_fast
        if args.fail_fast is not None
        else (defaults.fail_fast if defaults else False)
    )
    progress = args.progress or (defaults.progress if defaults else None) or "auto"
    result = run_batch(
        jobs,
        output,
        resume=bool(resume),
        worker_count=worker_count,
        fail_fast=bool(fail_fast),
        progress_mode=progress,
    )
    complete = sum(item.status == "complete" for item in result.items)
    skipped = sum(item.status == "skipped" for item in result.items)
    failed = sum(item.status == "failed" for item in result.items)
    if progress == "json":
        print(
            json.dumps(
                {
                    "type": "batch-complete",
                    "complete": complete,
                    "skipped": skipped,
                    "failed": failed,
                    "path": str(result.path),
                }
            )
        )
    elif progress != "quiet":
        print(
            f"batch complete: {complete} complete, {skipped} skipped, {failed} failed\n"
            f"results: {result.path}"
        )
    return result.exit_code


def _inspect(args: argparse.Namespace) -> int:
    discovery = discover_inputs(
        args.inputs,
        recursive=args.recursive,
        includes=args.include,
        excludes=args.exclude,
        follow_symlinks=args.follow_symlinks,
    )
    if not discovery.videos:
        return 3
    values = [probe_video(video) for video in discovery.videos]
    print(json.dumps(values[0] if len(values) == 1 else values, indent=2, ensure_ascii=False))
    return 0


def _runs(args: argparse.Namespace) -> int:
    if args.runs_command == "show":
        path = args.path
        if path.is_dir():
            candidates = [path / "run.json", path / "batch.json", path / "summary.json"]
            path = next((candidate for candidate in candidates if candidate.exists()), path)
        if not path.is_file():
            raise ValueError(f"Run state does not exist: {path}")
        print(json.dumps(json.loads(path.read_text()), indent=2, ensure_ascii=False))
        return 0
    rows = list_runs(args.output, args.status)
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        for row in rows:
            print(f"{row['kind']:6} {row.get('status', 'unknown'):12} {row['run']}")
        print(f"{len(rows)} run(s)")
    return 0


def _review(args: argparse.Namespace) -> int:
    labels, counts = review_candidates(
        args.path,
        load_config(args.config),
        labels_path=args.labels,
        open_previews=not args.no_open,
        include_reviewed=args.include_reviewed,
        scope=args.scope,
    )
    print(
        f"labels: {labels}\n"
        f"lightning: {counts['lightning']}\n"
        f"not-lightning: {counts['not-lightning']}\n"
        f"uncertain: {counts['uncertain']}\n"
        f"pending: {counts['pending']}"
    )
    return 0


def _detector(args: argparse.Namespace) -> int:
    result = detect_image(args.image)
    if args.output:
        write_detection_json(args.output, result)
    if args.preview:
        render_detections(args.image, args.preview, result.detections)
    print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "inspect":
            return _inspect(args)
        if args.command == "runs":
            return _runs(args)
        if args.command == "review":
            return _review(args)
        if args.command == "detector":
            return _detector(args)
        return _analyze(args)
    except KeyboardInterrupt:
        print("interrupted; progress written so far can be resumed", file=sys.stderr)
        return 130
    except ProbeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 4 if "ffprobe" in str(error).lower() else 2
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 5 if "disk space" in str(error).lower() else 2


if __name__ == "__main__":
    raise SystemExit(main())
