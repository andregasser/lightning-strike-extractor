from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass, field
from pathlib import Path

VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
IGNORED_DIRECTORIES = {
    ".git",
    ".venv",
    "__pycache__",
    "analysis",
    "output",
    "previews",
    "runs",
}


@dataclass(slots=True)
class DiscoveryResult:
    videos: list[Path] = field(default_factory=list)
    ignored: list[Path] = field(default_factory=list)
    duplicates: list[Path] = field(default_factory=list)


def _matches(path: Path, patterns: list[str]) -> bool:
    value = path.as_posix()
    return any(
        fnmatch.fnmatch(path.name, pattern) or fnmatch.fnmatch(value, pattern)
        for pattern in patterns
    )


def _candidate(path: Path, includes: list[str], excludes: list[str]) -> bool:
    return (
        path.suffix.lower() in VIDEO_EXTENSIONS
        and (not includes or _matches(path, includes))
        and not _matches(path, excludes)
    )


def discover_inputs(
    inputs: list[Path],
    *,
    recursive: bool = False,
    includes: list[str] | None = None,
    excludes: list[str] | None = None,
    follow_symlinks: bool = False,
) -> DiscoveryResult:
    includes = includes or []
    excludes = excludes or []
    result = DiscoveryResult()
    discovered: list[Path] = []
    for raw_path in inputs:
        path = raw_path.expanduser()
        if not path.exists():
            raise ValueError(f"Input does not exist: {path}")
        if path.is_file():
            if _candidate(path, includes, excludes):
                discovered.append(path)
            else:
                result.ignored.append(path)
            continue
        if not path.is_dir():
            result.ignored.append(path)
            continue
        if recursive:
            for root, directories, files in os.walk(path, followlinks=follow_symlinks):
                directories[:] = sorted(
                    directory
                    for directory in directories
                    if directory not in IGNORED_DIRECTORIES
                    and not directory.startswith(".")
                    and (follow_symlinks or not (Path(root) / directory).is_symlink())
                )
                for filename in sorted(files):
                    candidate = Path(root) / filename
                    if _candidate(candidate, includes, excludes):
                        discovered.append(candidate)
                    elif candidate.suffix.lower() in VIDEO_EXTENSIONS:
                        result.ignored.append(candidate)
        else:
            for candidate in sorted(path.iterdir()):
                if candidate.is_file() and _candidate(candidate, includes, excludes):
                    discovered.append(candidate)
                elif candidate.is_file() and candidate.suffix.lower() in VIDEO_EXTENSIONS:
                    result.ignored.append(candidate)

    identities: set[tuple[int, int] | Path] = set()
    for path in discovered:
        resolved = path.resolve()
        stat = resolved.stat()
        identity: tuple[int, int] | Path = (stat.st_dev, stat.st_ino)
        if identity in identities:
            result.duplicates.append(path)
            continue
        identities.add(identity)
        result.videos.append(resolved)
    result.videos.sort(key=lambda item: item.as_posix().casefold())
    return result
