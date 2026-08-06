from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(slots=True)
class AnalysisConfig:
    width: int = 960
    baseline_seconds: float = 0.30
    event_gap_seconds: float = 0.75
    event_window_before: float = 0.40
    event_window_after: float = 0.40
    rise_percentile: float = 0.995
    diff_percentile: float = 0.995
    minimum_rise: float = 1.0
    minimum_difference: float = 0.8
    minimum_high_rise: float = 3.0
    max_events: int = 0
    keep_frames_per_event: int = 3
    checkpoint_seconds: float = 30.0


@dataclass(slots=True)
class ChannelConfig:
    ridge_threshold: int = 10
    bright_area_threshold: int = 20
    minimum_line_length: int = 12
    maximum_line_gap: int = 8


@dataclass(slots=True)
class ExportConfig:
    top: int = 50
    jpeg_quality: int = 96
    contact_sheet_columns: int = 5


@dataclass(slots=True)
class Config:
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def load_config(path: Path | None = None) -> Config:
    config = Config()
    if path is None:
        return config
    with path.open("rb") as source:
        values = tomllib.load(source)
    for section_name in ("analysis", "channel", "export"):
        target = getattr(config, section_name)
        for key, value in values.get(section_name, {}).items():
            if not hasattr(target, key):
                raise ValueError(f"Unknown setting: {section_name}.{key}")
            setattr(target, key, value)
    return config
