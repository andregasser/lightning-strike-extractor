from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from . import __version__
from .config import load_config
from .pipeline import analyze
from .probe import ProbeError, probe_video


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lightning", description="Extract lightning strikes from videos")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Show video metadata")
    inspect.add_argument("video", type=Path)
    run = commands.add_parser("analyze", help="Analyze a video and export its best frames")
    run.add_argument("video", type=Path)
    run.add_argument("--output", type=Path, default=Path("runs"), help="Root directory for runs")
    run.add_argument("--config", type=Path, help="TOML configuration")
    run.add_argument("--start", type=float, default=0.0, help="Start time in seconds")
    run.add_argument("--end", type=float, help="End time in seconds")
    run.add_argument("--top", type=int, help="Number of stills to export")
    run.add_argument("--max-events", type=int, help="Maximum events to analyze")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "inspect":
            print(json.dumps(probe_video(args.video), indent=2, ensure_ascii=False))
            return 0
        config = load_config(args.config)
        if args.top is not None:
            config.export.top = args.top
        if args.max_events is not None:
            config.analysis.max_events = args.max_events
        if args.end is not None and args.end <= args.start:
            raise ValueError("--end must be greater than --start")
        run = analyze(args.video, args.output, config, args.start, args.end)
        print(f"analysis complete: {run}")
        return 0
    except (OSError, ValueError, RuntimeError, ProbeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

