from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class FlashEvent:
    event_id: str
    rank: int
    peak_time: float
    first_time: float
    last_time: float
    score: float
    rise: float
    difference: float
    high_rise: float
    hit_frames: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
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
    branch_points: int = 0
    channel_thickness: float = 0.0
    frame_quality: float = 0.0
    multiframe_support: float = 0.0
    multiframe_quality: float = 0.0
    background_frame_number: int = -1

    def as_dict(self) -> dict[str, object]:
        return asdict(self)
