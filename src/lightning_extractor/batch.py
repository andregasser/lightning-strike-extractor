from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import sys
import time
import tomllib
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from . import __version__
from .config import Config, load_config
from .discovery import DiscoveryResult, discover_inputs
from .pipeline import analyze, resolve_run_path


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _update_batch_state(batch: Path, **changes: object) -> dict[str, object]:
    path = batch / "batch.json"
    state = json.loads(path.read_text()) if path.exists() else {}
    state.update(changes)
    state["updated_at"] = _utc_now()
    _atomic_json(path, state)
    return state


@dataclass(slots=True)
class VideoJob:
    path: Path
    config: Config
    start_seconds: float = 0.0
    end_seconds: float | None = None
    config_source: str | None = None

    def identity_dict(self) -> dict[str, object]:
        stat = self.path.stat()
        return {
            "path": str(self.path.resolve()),
            "size_bytes": stat.st_size,
            "modified_ns": stat.st_mtime_ns,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "config": self.config.as_dict(),
        }


@dataclass(slots=True)
class BatchDefaults:
    output: Path | None = None
    jobs: int | None = None
    resume: bool | None = None
    fail_fast: bool | None = None
    progress: str | None = None


@dataclass(slots=True)
class Manifest:
    jobs: list[VideoJob] = field(default_factory=list)
    discoveries: list[DiscoveryResult] = field(default_factory=list)
    defaults: BatchDefaults = field(default_factory=BatchDefaults)


@dataclass(slots=True)
class BatchItemResult:
    video: str
    status: str
    run: str | None
    events: int = 0
    candidate_frames: int = 0
    exported_stills: int = 0
    elapsed_seconds: float = 0.0
    error: str | None = None


@dataclass(slots=True)
class BatchResult:
    path: Path
    items: list[BatchItemResult]

    @property
    def exit_code(self) -> int:
        return 1 if any(item.status == "failed" for item in self.items) else 0


def _resolve_optional_path(value: object, base: Path) -> Path | None:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _patterns(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]  # type: ignore[union-attr]


def _job_config(value: object, base: Path, fallback: Config) -> tuple[Config, str | None]:
    path = _resolve_optional_path(value, base)
    if path is None:
        return copy.deepcopy(fallback), None
    return load_config(path), str(path)


def load_manifest(path: Path, fallback_config: Config | None = None) -> Manifest:
    path = path.resolve()
    base = path.parent
    with path.open("rb") as source:
        values = tomllib.load(source)
    batch_values = values.get("batch", {})
    default_config, default_source = _job_config(
        batch_values.get("config"), base, fallback_config or Config()
    )
    manifest = Manifest(
        defaults=BatchDefaults(
            output=_resolve_optional_path(batch_values.get("output"), base),
            jobs=int(batch_values["jobs"]) if "jobs" in batch_values else None,
            resume=bool(batch_values["resume"]) if "resume" in batch_values else None,
            fail_fast=bool(batch_values["fail_fast"]) if "fail_fast" in batch_values else None,
            progress=str(batch_values["progress"]) if "progress" in batch_values else None,
        )
    )
    for entry in values.get("video", []):
        video_path = _resolve_optional_path(entry.get("path"), base)
        if video_path is None or not video_path.is_file():
            raise ValueError(f"Manifest video does not exist: {video_path}")
        config, config_source = _job_config(entry.get("config"), base, default_config)
        manifest.jobs.append(
            VideoJob(
                video_path.resolve(),
                config,
                float(entry.get("start", 0.0)),
                float(entry["end"]) if "end" in entry else None,
                config_source or default_source,
            )
        )
    for entry in values.get("input", []):
        input_path = _resolve_optional_path(entry.get("path"), base)
        if input_path is None:
            raise ValueError("Manifest input requires a path")
        discovery = discover_inputs(
            [input_path],
            recursive=bool(entry.get("recursive", False)),
            includes=_patterns(entry.get("include")),
            excludes=_patterns(entry.get("exclude")),
            follow_symlinks=bool(entry.get("follow_symlinks", False)),
        )
        manifest.discoveries.append(discovery)
        config, config_source = _job_config(entry.get("config"), base, default_config)
        for video in discovery.videos:
            manifest.jobs.append(
                VideoJob(
                    video,
                    copy.deepcopy(config),
                    float(entry.get("start", 0.0)),
                    float(entry["end"]) if "end" in entry else None,
                    config_source or default_source,
                )
            )
    return manifest


def deduplicate_jobs(jobs: list[VideoJob]) -> tuple[list[VideoJob], list[VideoJob]]:
    unique: list[VideoJob] = []
    duplicates: list[VideoJob] = []
    identities: set[str] = set()
    for job in jobs:
        canonical = json.dumps(job.identity_dict(), sort_keys=True, separators=(",", ":"))
        identity = hashlib.sha256(canonical.encode()).hexdigest()
        if identity in identities:
            duplicates.append(job)
        else:
            identities.add(identity)
            unique.append(job)
    unique.sort(key=lambda item: item.path.as_posix().casefold())
    return unique, duplicates


def batch_identity(jobs: list[VideoJob]) -> str:
    value = {
        "tool_version": __version__,
        "jobs": [job.identity_dict() for job in jobs],
    }
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()[:12]


def _read_summary(run: Path) -> dict[str, object]:
    path = run / "results" / "summary.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _run_is_complete(run: Path) -> bool:
    state = run / "run.json"
    return state.exists() and json.loads(state.read_text()).get("status") == "complete"


def _write_batch_outputs(
    batch: Path, items: list[BatchItemResult], status: str, total: int | None = None
) -> None:
    ordered = sorted(items, key=lambda item: item.video.casefold())
    counts = {
        name: sum(item.status == name for item in ordered)
        for name in ("complete", "skipped", "failed")
    }
    counts["pending"] = max((total or len(ordered)) - len(ordered), 0)
    _atomic_json(
        batch / "summary.json",
        {"status": status, "counts": counts, "items": [asdict(item) for item in ordered]},
    )
    temporary = batch / ".summary.csv.tmp"
    with temporary.open("w", newline="") as destination:
        fields = list(asdict(BatchItemResult("", "", None)))
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(item) for item in ordered)
    os.replace(temporary, batch / "summary.csv")


def _announce(
    result: BatchItemResult,
    current: int,
    total: int,
    mode: str,
    started_at: float,
    worker_count: int,
) -> None:
    if mode == "quiet":
        return
    elapsed = time.monotonic() - started_at
    eta = elapsed / current * max(total - current, 0) / max(worker_count, 1)
    if mode == "json":
        print(
            json.dumps(
                {
                    "type": "batch-item",
                    "completed": current,
                    "total": total,
                    "elapsed_seconds": round(elapsed, 3),
                    "eta_seconds": round(eta, 3),
                    **asdict(result),
                }
            ),
            file=sys.stderr,
            flush=True,
        )
        return
    symbol = {"complete": "✓", "skipped": "↷", "failed": "✗"}[result.status]
    detail = f"{result.events} events" if result.status != "failed" else result.error
    print(
        f"batch {current}/{total}  {symbol} {Path(result.video).name}  {detail or ''}  "
        f"elapsed {elapsed / 60:.1f}m  ETA {eta / 60:.1f}m",
        file=sys.stderr,
        flush=True,
    )


def run_batch(
    jobs: list[VideoJob],
    output: Path,
    *,
    resume: bool = False,
    worker_count: int = 1,
    fail_fast: bool = False,
    progress_mode: str = "auto",
) -> BatchResult:
    if not jobs:
        raise ValueError("No videos to analyze")
    if worker_count < 1:
        raise ValueError("--jobs must be at least 1")
    jobs, _ = deduplicate_jobs(jobs)
    identity = batch_identity(jobs)
    batch = output / "batches" / f"batch-{identity}"
    video_runs = output / "videos"
    if batch.exists() and not resume:
        raise RuntimeError(f"Batch already exists: {batch}. Use --resume to continue it")
    batch.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        batch / "inputs.json",
        {"batch_id": identity, "items": [job.identity_dict() for job in jobs]},
    )
    existing_state = (
        json.loads((batch / "batch.json").read_text()) if (batch / "batch.json").exists() else {}
    )
    _update_batch_state(
        batch,
        batch_id=identity,
        status="running",
        created_at=existing_state.get("created_at", _utc_now()),
        tool_version=__version__,
        video_count=len(jobs),
        worker_count=worker_count,
    )
    results: list[BatchItemResult] = []
    batch_started = time.monotonic()

    def execute(job: VideoJob) -> BatchItemResult:
        started = time.monotonic()
        run: Path | None = None
        try:
            run = resolve_run_path(
                job.path, video_runs, job.config, job.start_seconds, job.end_seconds
            )
            if resume and _run_is_complete(run):
                summary = _read_summary(run)
                return BatchItemResult(
                    str(job.path),
                    "skipped",
                    str(run),
                    int(summary.get("events", 0)),
                    int(summary.get("candidate_frames", 0)),
                    int(summary.get("exported_stills", 0)),
                    time.monotonic() - started,
                )
            run = analyze(
                job.path,
                video_runs,
                job.config,
                job.start_seconds,
                job.end_seconds,
                resume=resume,
                progress_mode=progress_mode,
                label=job.path.name,
            )
            summary = _read_summary(run)
            return BatchItemResult(
                str(job.path),
                "complete",
                str(run),
                int(summary.get("events", 0)),
                int(summary.get("candidate_frames", 0)),
                int(summary.get("exported_stills", 0)),
                time.monotonic() - started,
            )
        except Exception as error:  # noqa: BLE001 - isolate failures to one video
            return BatchItemResult(
                str(job.path),
                "failed",
                str(run) if run is not None else None,
                elapsed_seconds=time.monotonic() - started,
                error=f"{type(error).__name__}: {error}",
            )

    try:
        if worker_count == 1:
            for job in jobs:
                result = execute(job)
                results.append(result)
                _announce(
                    result,
                    len(results),
                    len(jobs),
                    progress_mode,
                    batch_started,
                    worker_count,
                )
                _write_batch_outputs(batch, results, "running", len(jobs))
                if fail_fast and result.status == "failed":
                    break
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                job_iterator = iter(jobs)
                futures: dict[Future[BatchItemResult], VideoJob] = {}
                for job in job_iterator:
                    futures[executor.submit(execute, job)] = job
                    if len(futures) == worker_count:
                        break
                scheduling = True
                while futures:
                    completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in completed:
                        futures.pop(future)
                        result = future.result()
                        results.append(result)
                        _announce(
                            result,
                            len(results),
                            len(jobs),
                            progress_mode,
                            batch_started,
                            worker_count,
                        )
                        _write_batch_outputs(batch, results, "running", len(jobs))
                        if fail_fast and result.status == "failed":
                            scheduling = False
                    if scheduling:
                        for job in job_iterator:
                            futures[executor.submit(execute, job)] = job
                            if len(futures) == worker_count:
                                break
    except KeyboardInterrupt:
        _update_batch_state(
            batch,
            batch_id=identity,
            status="interrupted",
            interrupted_at=_utc_now(),
            tool_version=__version__,
            video_count=len(jobs),
            worker_count=worker_count,
        )
        _write_batch_outputs(batch, results, "interrupted", len(jobs))
        raise

    status = "failed" if any(item.status == "failed" for item in results) else "complete"
    _write_batch_outputs(batch, results, status, len(jobs))
    _update_batch_state(
        batch,
        batch_id=identity,
        status=status,
        completed_at=_utc_now(),
        tool_version=__version__,
        video_count=len(jobs),
        processed_count=len(results),
        worker_count=worker_count,
    )
    return BatchResult(batch, results)


def list_runs(output: Path, status: str | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for state_path in sorted((output / "videos").glob("*/run.json")):
        state = json.loads(state_path.read_text())
        if status and state.get("status") != status:
            continue
        rows.append({"kind": "video", "run": str(state_path.parent), **state})
    for state_path in sorted((output / "batches").glob("*/batch.json")):
        state = json.loads(state_path.read_text())
        if status and state.get("status") != status:
            continue
        rows.append({"kind": "batch", "run": str(state_path.parent), **state})
    rows.sort(key=lambda row: str(row["run"]))
    return rows
