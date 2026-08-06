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

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

