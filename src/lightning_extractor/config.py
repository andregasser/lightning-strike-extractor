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
    keep_frames_per_event: int = 0
    checkpoint_seconds: float = 30.0


@dataclass(slots=True)
class ChannelConfig:
    analysis_width: int = 1920
    stabilization_enabled: bool = True
    stabilization_width: int = 640
    stabilization_max_features: int = 1200
    stabilization_min_matches: int = 24
    stabilization_min_inlier_ratio: float = 0.45
    stabilization_ransac_threshold: float = 2.5
    stabilization_orb_max_residual: float = 3.0
    stabilization_ecc_enabled: bool = True
    stabilization_min_ecc_correlation: float = 0.90
    stabilization_max_translation_fraction: float = 0.08
    stabilization_max_rotation_degrees: float = 5.0
    stabilization_max_scale_change: float = 0.05
    stabilization_mask_aligned_edges: bool = True
    stabilization_edge_mask_dilation: int = 5
    multiframe_enabled: bool = True
    multiframe_width: int = 640
    multiframe_window_seconds: float = 0.06
    multiframe_dilation_pixels: int = 3
    multiframe_bonus_weight: float = 0.25
    multiframe_template_min_support: float = 0.5
    multiframe_peak_radius_frames: int = 2
    ridge_threshold: int = 10
    bright_area_threshold: int = 20
    minimum_line_length: int = 12
    maximum_line_gap: int = 8


@dataclass(slots=True)
class ExportConfig:
    top: int = 50
    minimum_geometry_score: float = 100.0
    minimum_supported_geometry_score: float = 25.0
    minimum_low_geometry_multiframe_support: float = 0.5
    minimum_channel_length: float = 40.0
    one_frame_per_event: bool = True
    minimum_winner_geometry_ratio: float = 0.0
    jpeg_quality: int = 96
    contact_sheet_columns: int = 5
    contact_sheet_context_frames: int = 2
    contact_sheet_context_stride: int = 1
    contact_sheet_include_overlay: bool = True
    event_frames_enabled: bool = True
    slow_motion_enabled: bool = True
    slow_motion_before_seconds: float = 0.25
    slow_motion_after_seconds: float = 0.25
    slow_motion_factor: float = 4.0
    slow_motion_output_fps: int = 25
    slow_motion_crf: int = 18


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
