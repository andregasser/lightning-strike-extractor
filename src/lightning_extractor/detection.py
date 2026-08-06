from __future__ import annotations

import math
import os
import statistics
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import Config
from .models import CandidateFrame, FlashEvent

Progress = Callable[[int, int], None]


@dataclass(slots=True)
class ChannelMetrics:
    score: float
    line_segments: int
    bright_area: float
    channel_length: float
    channel_strength: float
    branch_points: int
    channel_thickness: float
    frame_quality: float


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
    progress: Progress | None = None,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
) -> list[FlashEvent]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {video}")
    start_frame = max(0, round(start_seconds * fps))
    end_frame = round(end_seconds * fps) if end_seconds is not None else None
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    expected_end = min(end_frame, source_frames) if end_frame is not None else source_frames
    window_size = max(2, round(config.analysis.baseline_seconds * fps))
    average_window: deque[float] = deque(maxlen=window_size)
    high_window: deque[float] = deque(maxlen=window_size)
    samples: list[dict[str, float]] = []
    previous: np.ndarray | None = None
    frame_number = start_frame
    chunk_samples: list[dict[str, float]] = []
    chunk_number = 0

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        chunks = sorted(checkpoint_dir.glob("chunk-*.npz")) if resume else []
        for chunk in chunks:
            with np.load(chunk) as data:
                samples.extend(
                    {
                        "time": float(time_value),
                        "rise": float(rise),
                        "high_rise": float(high_rise),
                        "difference": float(difference),
                    }
                    for time_value, rise, high_rise, difference in zip(
                        data["time"], data["rise"], data["high_rise"], data["difference"]
                    )
                )
        if chunks:
            with np.load(chunks[-1]) as data:
                frame_number = int(data["next_frame"])
                average_window.extend(float(value) for value in data["average_window"])
                high_window.extend(float(value) for value in data["high_window"])
                previous = data["previous_gray"].copy()
            chunk_number = len(chunks)

    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    checkpoint_frames = max(1, round(config.analysis.checkpoint_seconds * fps))

    def save_checkpoint() -> None:
        nonlocal chunk_number, chunk_samples
        if checkpoint_dir is None or not chunk_samples or previous is None:
            return
        destination = checkpoint_dir / f"chunk-{chunk_number:06d}.npz"
        temporary = destination.with_suffix(".npz.tmp")
        with temporary.open("wb") as output:
            np.savez_compressed(
                output,
                time=np.asarray([row["time"] for row in chunk_samples]),
                rise=np.asarray([row["rise"] for row in chunk_samples]),
                high_rise=np.asarray([row["high_rise"] for row in chunk_samples]),
                difference=np.asarray([row["difference"] for row in chunk_samples]),
                next_frame=np.asarray(frame_number),
                average_window=np.asarray(average_window),
                high_window=np.asarray(high_window),
                previous_gray=previous,
            )
        os.replace(temporary, destination)
        chunk_number += 1
        chunk_samples = []

    try:
        while frame_number < expected_end:
            ok, frame = capture.read()
            if not ok:
                break
            gray = _gray(frame, config.analysis.width)
            average = float(np.mean(gray))
            high = float(np.percentile(gray, 90))
            difference = (
                float(np.mean(cv2.absdiff(gray, previous))) if previous is not None else 0.0
            )
            baseline = statistics.median(average_window) if average_window else average
            high_baseline = statistics.median(high_window) if high_window else high
            sample = {
                "time": frame_number / fps,
                "rise": average - baseline,
                "high_rise": high - high_baseline,
                "difference": difference,
            }
            samples.append(sample)
            chunk_samples.append(sample)
            average_window.append(average)
            high_window.append(high)
            previous = gray
            frame_number += 1
            if len(chunk_samples) >= checkpoint_frames:
                save_checkpoint()
            if progress is not None:
                progress(frame_number - start_frame, max(expected_end - start_frame, 0))
    except KeyboardInterrupt:
        save_checkpoint()
        raise
    finally:
        capture.release()
    save_checkpoint()

    if not samples:
        raise RuntimeError("No video frames could be decoded in the selected time range")
    if frame_number + 1 < expected_end:
        raise RuntimeError(
            f"Video decoding stopped early at frame {frame_number:,}; "
            f"expected frame {expected_end:,}"
        )

    rise_cutoff = max(
        config.analysis.minimum_rise,
        percentile([row["rise"] for row in samples], config.analysis.rise_percentile),
    )
    diff_cutoff = max(
        config.analysis.minimum_difference,
        percentile([row["difference"] for row in samples], config.analysis.diff_percentile),
    )
    hits = [
        row
        for row in samples
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
            FlashEvent(
                "",
                0,
                peak["time"],
                group[0]["time"],
                group[-1]["time"],
                score,
                peak["rise"],
                peak["difference"],
                peak["high_rise"],
                len(group),
            )
        )
    events.sort(key=lambda event: event.score, reverse=True)
    if config.analysis.max_events > 0:
        events = events[: config.analysis.max_events]
    for rank, event in enumerate(events, 1):
        event.rank = rank
        event.event_id = f"evt_{rank:06d}"
    return events


def frame_geometry_score(
    previous: np.ndarray, current: np.ndarray, config: Config
) -> tuple[float, int, float]:
    metrics = frame_channel_metrics(previous, current, config)
    return metrics.score, metrics.line_segments, metrics.bright_area


def frame_channel_metrics(
    previous: np.ndarray, current: np.ndarray, config: Config
) -> ChannelMetrics:
    before = _gray(previous, config.channel.analysis_width)
    gray = _gray(current, config.channel.analysis_width)
    temporal = cv2.subtract(gray, before)
    ridge = cv2.subtract(gray, cv2.GaussianBlur(gray, (0, 0), 5.0))
    channel_response = cv2.min(ridge, temporal)
    channel = channel_response
    channel = cv2.threshold(channel, config.channel.ridge_threshold, 255, cv2.THRESH_BINARY)[1]
    # Do not erode thin channels: at analysis resolution a real lightning branch
    # may only be one pixel wide. Closing only bridges tiny compression gaps.
    channel = cv2.morphologyEx(channel, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    skeleton = _skeletonize(channel)
    bright_area = float(np.count_nonzero(temporal > config.channel.bright_area_threshold))
    score, best_component, length, strength, branches, thickness = _best_channel_component(
        channel, skeleton, channel_response
    )
    lines = cv2.HoughLinesP(
        best_component,
        1,
        np.pi / 180,
        threshold=12,
        minLineLength=config.channel.minimum_line_length,
        maxLineGap=config.channel.maximum_line_gap,
    )
    count = 0 if lines is None else len(lines)
    # Dense frame-wide edge fields are characteristic of camera motion, grass,
    # and textured clouds rather than one coherent lightning tree.
    clutter = float(np.count_nonzero(skeleton))
    score /= (1.0 + clutter / 5000.0) ** 2
    score /= 1.0 + bright_area / 100000.0
    frame_quality = (
        length
        * strength
        * (1.0 + min(branches, 10) * 0.15)
        / max(thickness, 1.0) ** 2
    )
    return ChannelMetrics(
        score,
        count,
        bright_area,
        length,
        strength,
        branches,
        thickness,
        frame_quality,
    )


def _best_channel_component(
    mask: np.ndarray, skeleton: np.ndarray, response: np.ndarray
) -> tuple[float, np.ndarray, float, float, int, float]:
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if component_count <= 1:
        return 0.0, np.zeros_like(mask), 0.0, 0.0, 0, 0.0

    skeleton_pixels = skeleton > 0
    skeleton_labels = labels[skeleton_pixels]
    lengths = np.bincount(skeleton_labels, minlength=component_count).astype(float)
    strength_sums = np.bincount(
        skeleton_labels,
        weights=response[skeleton_pixels],
        minlength=component_count,
    )
    areas = stats[:, cv2.CC_STAT_AREA].astype(float)
    thickness = areas / np.maximum(lengths, 1.0)
    strengths = strength_sums / np.maximum(lengths, 1.0)
    rough_scores = lengths * (strengths / 10.0) ** 2 / np.maximum(thickness, 1.0) ** 4
    rough_scores[0] = 0.0

    best_score = 0.0
    best_component = np.zeros_like(mask)
    best_metrics = (0.0, 0.0, 0, 0.0)
    # Branch analysis is only needed for the strongest thin components, not
    # thousands of tiny compression or foliage fragments.
    labels_to_check = np.argsort(rough_scores)[-32:]
    for label in labels_to_check:
        if lengths[label] < 8:
            continue
        component = np.where((labels == label) & skeleton_pixels, 255, 0).astype(np.uint8)
        branches = _branch_point_count(component)
        branch_multiplier = 1.0 + min(branches, 10) * 0.35
        component_score = rough_scores[label] * branch_multiplier
        if component_score > best_score:
            best_score = float(component_score)
            best_component = component
            best_metrics = (
                float(lengths[label]),
                float(strengths[label]),
                branches,
                float(thickness[label]),
            )
    return best_score, best_component, *best_metrics


def _skeletonize(mask: np.ndarray) -> np.ndarray:
    remaining = mask.copy()
    skeleton = np.zeros_like(mask)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(remaining):
        opened = cv2.morphologyEx(remaining, cv2.MORPH_OPEN, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(remaining, opened))
        remaining = cv2.erode(remaining, element)
    return skeleton


def _branch_point_count(skeleton: np.ndarray) -> int:
    binary = (skeleton > 0).astype(np.uint8)
    neighbours = cv2.filter2D(binary, cv2.CV_16S, np.ones((3, 3), np.uint8)) - binary
    junction_pixels = np.where((binary > 0) & (neighbours >= 3), 255, 0).astype(np.uint8)
    # One physical fork often spans several adjacent skeleton pixels.
    components, _ = cv2.connectedComponents(junction_pixels)
    return max(0, components - 1)


def rank_event_frames(
    video: Path,
    fps: float,
    events: list[FlashEvent],
    config: Config,
    progress: Progress | None = None,
    completed_events: int = 0,
    existing_candidates: list[CandidateFrame] | None = None,
    checkpoint: Callable[[int, list[CandidateFrame]], None] | None = None,
) -> list[CandidateFrame]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {video}")
    candidates = list(existing_candidates or [])
    for event_index, event in enumerate(events[completed_events:], completed_events + 1):
        start = max(0, round((event.first_time - config.analysis.event_window_before) * fps))
        end = round((event.last_time + config.analysis.event_window_after) * fps)
        count = max(
            2,
            end - start,
        )
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        ok, background = capture.read()
        if not ok:
            continue
        local: list[CandidateFrame] = []
        for offset in range(1, count + 1):
            ok, current = capture.read()
            if not ok:
                break
            metrics = frame_channel_metrics(background, current, config)
            local.append(
                CandidateFrame(
                    rank=0,
                    event_id=event.event_id,
                    frame_number=start + offset,
                    time=(start + offset) / fps,
                    geometry_score=metrics.score,
                    line_segments=metrics.line_segments,
                    bright_area=metrics.bright_area,
                    channel_length=metrics.channel_length,
                    channel_strength=metrics.channel_strength,
                    branch_points=metrics.branch_points,
                    channel_thickness=metrics.channel_thickness,
                    frame_quality=metrics.frame_quality,
                )
            )
        if config.analysis.keep_frames_per_event <= 0:
            selected = local
        else:
            selected = []
            for row in sorted(local, key=lambda item: item.geometry_score, reverse=True):
                if all(abs(row.time - old.time) >= max(0.025, 2 / fps) for old in selected):
                    selected.append(row)
                if len(selected) == config.analysis.keep_frames_per_event:
                    break
        candidates.extend(selected)
        if checkpoint is not None:
            checkpoint(event_index, candidates)
        if progress is not None:
            progress(event_index, len(events))
    capture.release()
    candidates.sort(key=lambda item: item.frame_quality, reverse=True)
    for rank, candidate in enumerate(candidates, 1):
        candidate.rank = rank
    return candidates
