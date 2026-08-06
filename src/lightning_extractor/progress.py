from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--:--"
    rounded = round(seconds)
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


@dataclass(slots=True)
class ProgressReporter:
    phase: str
    unit: str
    initial_completed: int = 0
    minimum_interval: float = 2.0
    started_at: float = field(default_factory=time.monotonic)
    last_printed_at: float = 0.0

    def __post_init__(self) -> None:
        self.last_printed_at = self.started_at

    def update(self, completed: int, total: int, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_printed_at < self.minimum_interval:
            return
        elapsed = max(now - self.started_at, 1e-9)
        rate = max(completed - self.initial_completed, 0) / elapsed
        remaining = (total - completed) / rate if total > completed and rate > 0 else 0.0
        percent = completed / total * 100 if total else 100.0
        print(
            f"{self.phase}: {percent:6.2f}%  {completed:,}/{total:,} {self.unit}  "
            f"{rate:,.1f} {self.unit}/s  elapsed {format_duration(elapsed)}  "
            f"ETA {format_duration(remaining)}",
            file=sys.stderr,
            flush=True,
        )
        self.last_printed_at = now
