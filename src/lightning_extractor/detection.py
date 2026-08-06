from __future__ import annotations

from collections import deque
import math
from pathlib import Path
import statistics
from typing import Callable

import cv2
import numpy as np

from .config import Config
from .models import CandidateFrame, FlashEvent


Progress = Callable[[str], None]


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _gray(frame: np.ndarray, width: int) -> np.ndarray:
    height = max(1, round(frame.shape[0] * width / frame.shape[1]))
    return cv2.resize(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )


def detect_flashes(
    video: Path,
    fps: float,
    config: Config,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
    progress: Progress = print,
) -> list[FlashEvent]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {video}")
    start_frame = max(0, round(start_seconds * fps))
    end_frame = round(end_seconds * fps) if end_seconds is not None else None
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    window_size = max(2, round(config.analysis.baseline_seconds * fps))
    average_window: deque[float] = deque(maxlen=window_size)
    high_window: deque[float] = deque(maxlen=window_size)
    samples: list[dict[str, float]] = []
    previous: np.ndarray | None = None
    frame_number = start_frame
    while end_frame is None or frame_number < end_frame:
        ok, frame = capture.read()
        if not ok:
            break
        gray = _gray(frame, config.analysis.width)
        average = float(np.mean(gray))
        high = float(np.percentile(gray, 90))
        difference = float(np.mean(cv2.absdiff(gray, previous))) if previous is not None else 0.0
        baseline = statistics.median(average_window) if average_window else average
        high_baseline = statistics.median(high_window) if high_window else high
        samples.append(
            {
                "time": frame_number / fps,
                "rise": average - baseline,
                "high_rise": high - high_baseline,
                "difference": difference,
            }
        )
        average_window.append(average)
        high_window.append(high)
        previous = gray
        frame_number += 1
        if len(samples) % max(round(fps * 60), 1) == 0:
            progress(f"flash scan: {len(samples) / fps / 60:.1f} min processed")
    capture.release()

    rise_cutoff = max(
        config.analysis.minimum_rise,
        percentile([row["rise"] for row in samples], config.analysis.rise_percentile),
    )
    diff_cutoff = max(
        config.analysis.minimum_difference,
        percentile([row["difference"] for row in samples], config.analysis.diff_percentile),
    )
    hits = [
        row for row in samples
        if row["rise"] >= rise_cutoff
        and (
            row["difference"] >= diff_cutoff
            or row["high_rise"] >= config.analysis.minimum_high_rise
        )
    ]
    groups: list[list[dict[str, float]]] = []
    for hit in hits:
        if not groups or hit["time"] - groups[-1][-1]["time"] > config.analysis.event_gap_seconds:
            groups.append([hit])
        else:
            groups[-1].append(hit)
    events: list[FlashEvent] = []
    for group in groups:
        peak = max(
            group,
            key=lambda row: row["rise"] * 2 + row["difference"] + max(0.0, row["high_rise"]) * 0.25,
        )
        score = peak["rise"] * 2 + peak["difference"] + max(0.0, peak["high_rise"]) * 0.25
        events.append(
            FlashEvent("", 0, peak["time"], group[0]["time"], group[-1]["time"], score,
                       peak["rise"], peak["difference"], peak["high_rise"], len(group))
        )
    events.sort(key=lambda event: event.score, reverse=True)
    if config.analysis.max_events > 0:
        events = events[: config.analysis.max_events]
    for rank, event in enumerate(events, 1):
        event.rank = rank
        event.event_id = f"evt_{rank:06d}"
    progress(f"flash scan: {len(events)} events (rise >= {rise_cutoff:.2f}, diff >= {diff_cutoff:.2f})")
    return events


def frame_geometry_score(
    previous: np.ndarray, current: np.ndarray, config: Config
) -> tuple[float, int, float]:
    before = _gray(previous, config.analysis.width)
    gray = _gray(current, config.analysis.width)
    temporal = cv2.subtract(gray, before)
    ridge = cv2.subtract(gray, cv2.GaussianBlur(gray, (0, 0), 5.0))
    channel = cv2.bitwise_and(ridge, temporal)
    channel = cv2.threshold(channel, config.channel.ridge_threshold, 255, cv2.THRESH_BINARY)[1]
    channel = cv2.morphologyEx(channel, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    edges = cv2.Canny(channel, 30, 100)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=12,
        minLineLength=config.channel.minimum_line_length,
        maxLineGap=config.channel.maximum_line_gap,
    )
    lengths = [] if lines is None else [
        float(np.hypot(x2 - x1, y2 - y1)) for x1, y1, x2, y2 in lines.reshape(-1, 4)
    ]
    count = len(lengths)
    bright_area = float(np.count_nonzero(temporal > config.channel.bright_area_threshold))
    score = sum(sorted(lengths, reverse=True)[:30]) * (1.0 + min(count, 20) / 10.0)
    score /= 1.0 + bright_area / 12000.0
    return score, count, bright_area


def rank_event_frames(
    video: Path, fps: float, events: list[FlashEvent], config: Config, progress: Progress = print
) -> list[CandidateFrame]:
    capture = cv2.VideoCapture(str(video))
    candidates: list[CandidateFrame] = []
    for event_index, event in enumerate(events, 1):
        start = max(0, round((event.peak_time - config.analysis.event_window_before) * fps))
        count = max(2, round((config.analysis.event_window_before + config.analysis.event_window_after) * fps))
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        ok, previous = capture.read()
        if not ok:
            continue
        local: list[CandidateFrame] = []
        for offset in range(1, count + 1):
            ok, current = capture.read()
            if not ok:
                break
            score, segments, area = frame_geometry_score(previous, current, config)
            local.append(CandidateFrame(0, event.event_id, start + offset, (start + offset) / fps,
                                        score, segments, area))
            previous = current
        selected: list[CandidateFrame] = []
        for row in sorted(local, key=lambda item: item.geometry_score, reverse=True):
            if all(abs(row.time - old.time) >= max(0.025, 2 / fps) for old in selected):
                selected.append(row)
            if len(selected) == config.analysis.keep_frames_per_event:
                break
        candidates.extend(selected)
        if event_index % 25 == 0:
            progress(f"channel ranking: {event_index}/{len(events)} events")
    capture.release()
    candidates.sort(key=lambda item: item.geometry_score, reverse=True)
    for rank, candidate in enumerate(candidates, 1):
        candidate.rank = rank
    return candidates
