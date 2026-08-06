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
    channel_luminance: float
    branch_points: int
    channel_thickness: float
    frame_quality: float
    channel_mask: np.ndarray | None = None


@dataclass(slots=True)
class StabilizationReference:
    gray: np.ndarray
    keypoints: tuple[object, ...]
    descriptors: np.ndarray | None
    scale: float


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
    previous: np.ndarray,
    current: np.ndarray,
    config: Config,
    stabilization_reference: StabilizationReference | None = None,
) -> ChannelMetrics:
    reference = stabilization_reference or _prepare_stabilization_reference(previous, config)
    before = reference.gray
    gray = _gray(current, config.channel.analysis_width)
    before, stabilized = _align_reference(reference, gray, config)
    temporal = cv2.subtract(gray, before)
    ridge = cv2.subtract(gray, cv2.GaussianBlur(gray, (0, 0), 5.0))
    channel_response = cv2.min(ridge, temporal)
    if stabilized and config.channel.stabilization_mask_aligned_edges:
        # Sub-pixel interpolation leaves thin residuals around static, high-contrast
        # structures. They cannot be lightning because they already existed in the
        # reference frame, so remove a narrow band around those aligned edges.
        static_edges = cv2.Canny(before, 50, 150)
        dilation = max(1, config.channel.stabilization_edge_mask_dilation)
        static_edges = cv2.dilate(
            static_edges,
            np.ones((dilation, dilation), np.uint8),
        )
        channel_response[static_edges > 0] = 0
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
    component_pixels = best_component > 0
    luminance = float(np.mean(gray[component_pixels])) if np.any(component_pixels) else 0.0
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
        * (luminance / 10.0) ** 2
        * (1.0 + min(branches, 10) * 0.15)
        / max(thickness, 1.0)
    )
    return ChannelMetrics(
        score,
        count,
        bright_area,
        length,
        strength,
        luminance,
        branches,
        thickness,
        frame_quality,
        best_component,
    )


def _prepare_stabilization_reference(
    frame: np.ndarray, config: Config
) -> StabilizationReference:
    gray = _gray(frame, config.channel.analysis_width)
    if not config.channel.stabilization_enabled:
        return StabilizationReference(gray, (), None, 1.0)
    width = min(config.channel.stabilization_width, gray.shape[1])
    height = max(1, round(gray.shape[0] * width / gray.shape[1]))
    small = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
    detector = cv2.ORB_create(nfeatures=config.channel.stabilization_max_features)
    keypoints, descriptors = detector.detectAndCompute(small, None)
    return StabilizationReference(
        gray,
        tuple(keypoints),
        descriptors,
        width / gray.shape[1],
    )


def _align_reference(
    reference: StabilizationReference, current: np.ndarray, config: Config
) -> tuple[np.ndarray, bool]:
    if not config.channel.stabilization_enabled:
        return reference.gray, False
    width = round(current.shape[1] * reference.scale)
    height = round(current.shape[0] * reference.scale)
    small = cv2.resize(current, (width, height), interpolation=cv2.INTER_AREA)
    matrix = _orb_affine(reference, small, config)
    if matrix is not None and (
        not _plausible_affine(matrix, width, height, config)
        or _alignment_residual(reference, small, matrix)
        > config.channel.stabilization_orb_max_residual
    ):
        matrix = None
    inverse = False
    if matrix is None and config.channel.stabilization_ecc_enabled:
        matrix = _ecc_affine(reference, small, config)
        inverse = matrix is not None
    if matrix is None or not _plausible_affine(matrix, width, height, config):
        return reference.gray, False
    full_matrix = matrix.copy()
    full_matrix[:, 2] /= reference.scale
    flags = cv2.INTER_LINEAR | (cv2.WARP_INVERSE_MAP if inverse else 0)
    aligned = cv2.warpAffine(
        reference.gray,
        full_matrix,
        (current.shape[1], current.shape[0]),
        flags=flags,
        borderMode=cv2.BORDER_REFLECT,
    )
    return aligned, True


def _alignment_residual(
    reference: StabilizationReference, current: np.ndarray, matrix: np.ndarray
) -> float:
    height, width = current.shape
    reference_small = cv2.resize(
        reference.gray,
        (width, height),
        interpolation=cv2.INTER_AREA,
    )
    aligned = cv2.warpAffine(
        reference_small,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return float(np.mean(cv2.absdiff(aligned, current)))


def _orb_affine(
    reference: StabilizationReference, current: np.ndarray, config: Config
) -> np.ndarray | None:
    if (
        reference.descriptors is None
        or len(reference.keypoints) < config.channel.stabilization_min_matches
    ):
        return None
    detector = cv2.ORB_create(nfeatures=config.channel.stabilization_max_features)
    keypoints, descriptors = detector.detectAndCompute(current, None)
    if descriptors is None or len(keypoints) < config.channel.stabilization_min_matches:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(
        reference.descriptors, descriptors, k=2
    )
    matches = [
        pair[0]
        for pair in pairs
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance
    ]
    if len(matches) < config.channel.stabilization_min_matches:
        return None
    source = np.float32([reference.keypoints[item.queryIdx].pt for item in matches])
    target = np.float32([keypoints[item.trainIdx].pt for item in matches])
    matrix, inliers = cv2.estimateAffinePartial2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=config.channel.stabilization_ransac_threshold,
    )
    if (
        matrix is None
        or inliers is None
        or float(np.mean(inliers)) < config.channel.stabilization_min_inlier_ratio
    ):
        return None
    return matrix


def _ecc_affine(
    reference: StabilizationReference, current: np.ndarray, config: Config
) -> np.ndarray | None:
    height, width = current.shape
    reference_small = cv2.resize(
        reference.gray,
        (width, height),
        interpolation=cv2.INTER_AREA,
    )
    matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
    try:
        correlation, matrix = cv2.findTransformECC(
            current,
            reference_small,
            matrix,
            cv2.MOTION_AFFINE,
            criteria,
            None,
            3,
        )
    except cv2.error:
        return None
    if correlation < config.channel.stabilization_min_ecc_correlation:
        return None
    return matrix


def _plausible_affine(
    matrix: np.ndarray, width: int, height: int, config: Config
) -> bool:
    singular_values = np.linalg.svd(matrix[:, :2], compute_uv=False)
    rotation = abs(math.degrees(math.atan2(matrix[1, 0], matrix[0, 0])))
    translation = float(np.hypot(matrix[0, 2], matrix[1, 2]))
    diagonal = float(np.hypot(width, height))
    return bool(
        np.all(np.abs(singular_values - 1.0) <= config.channel.stabilization_max_scale_change)
        and rotation <= config.channel.stabilization_max_rotation_degrees
        and translation
        <= diagonal * config.channel.stabilization_max_translation_fraction
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


def _apply_multiframe_support(
    candidates: list[CandidateFrame],
    channel_masks: list[np.ndarray],
    config: Config,
) -> None:
    for candidate in candidates:
        candidate.multiframe_support = 0.0
        candidate.multiframe_quality = candidate.frame_quality
    if not config.channel.multiframe_enabled or len(candidates) < 2:
        return

    radius = max(0, config.channel.multiframe_dilation_pixels)
    kernel_size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = [cv2.dilate(mask, kernel) for mask in channel_masks]
    window = max(0.0, config.channel.multiframe_window_seconds)
    for index, (candidate, mask) in enumerate(zip(candidates, channel_masks, strict=True)):
        pixels = int(np.count_nonzero(mask))
        if pixels == 0:
            continue
        support = 0.0
        for neighbour_index, neighbour in enumerate(candidates):
            if neighbour_index == index or abs(neighbour.time - candidate.time) > window:
                continue
            overlap = float(np.count_nonzero((mask > 0) & (dilated[neighbour_index] > 0)))
            support = max(support, overlap / pixels)
        candidate.multiframe_support = support
        candidate.multiframe_quality = candidate.frame_quality * (
            1.0 + config.channel.multiframe_bonus_weight * support
        )


def _compact_channel_mask(mask: np.ndarray, config: Config) -> np.ndarray:
    width = min(config.channel.multiframe_width, mask.shape[1])
    if width == mask.shape[1]:
        return mask
    height = max(1, round(mask.shape[0] * width / mask.shape[1]))
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


def _compact_gray(frame: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = shape
    return cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)


def _apply_multiframe_peak_quality(
    candidates: list[CandidateFrame],
    channel_masks: list[np.ndarray],
    frame_grays: list[np.ndarray],
    config: Config,
) -> None:
    if not config.channel.multiframe_enabled:
        return
    radius = max(1, config.channel.multiframe_dilation_pixels)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (radius * 2 + 1, radius * 2 + 1),
    )
    radius_frames = max(0, config.channel.multiframe_peak_radius_frames)
    for index, candidate in enumerate(candidates):
        sampled_gray = cv2.dilate(frame_grays[index], kernel)
        best_quality = candidate.frame_quality
        best_luminance = candidate.channel_luminance
        best_support = candidate.multiframe_support
        best_template = candidate.frame_number
        for template_index, template in enumerate(candidates):
            if abs(template_index - index) > radius_frames:
                continue
            if (
                template_index != index
                and template.multiframe_support
                < config.channel.multiframe_template_min_support
            ):
                continue
            mask = channel_masks[template_index] > 0
            if not np.any(mask):
                continue
            luminance = float(np.mean(sampled_gray[mask]))
            quality = (
                template.channel_length
                * (luminance / 10.0) ** 2
                * (1.0 + min(template.branch_points, 10) * 0.15)
                / max(template.channel_thickness, 1.0)
            )
            if quality > best_quality:
                best_quality = quality
                best_luminance = luminance
                best_support = template.multiframe_support
                best_template = template.frame_number
        candidate.frame_quality = best_quality
        candidate.channel_luminance = best_luminance
        candidate.peak_multiframe_support = best_support
        candidate.channel_template_frame_number = best_template
        candidate.multiframe_quality = best_quality * (
            1.0 + config.channel.multiframe_bonus_weight * best_support
        )


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
        stabilization_reference = _prepare_stabilization_reference(background, config)
        local: list[CandidateFrame] = []
        channel_masks: list[np.ndarray] = []
        frame_grays: list[np.ndarray] = []
        for offset in range(1, count + 1):
            ok, current = capture.read()
            if not ok:
                break
            metrics = frame_channel_metrics(
                background,
                current,
                config,
                stabilization_reference=stabilization_reference,
            )
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
                    channel_luminance=metrics.channel_luminance,
                    branch_points=metrics.branch_points,
                    channel_thickness=metrics.channel_thickness,
                    frame_quality=metrics.frame_quality,
                    background_frame_number=start,
                )
            )
            channel_masks.append(
                _compact_channel_mask(
                    metrics.channel_mask
                    if metrics.channel_mask is not None
                    else np.zeros_like(stabilization_reference.gray),
                    config,
                )
            )
            frame_grays.append(_compact_gray(current, channel_masks[-1].shape))
        _apply_multiframe_support(local, channel_masks, config)
        _apply_multiframe_peak_quality(local, channel_masks, frame_grays, config)
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
    candidates.sort(
        key=lambda item: item.multiframe_quality or item.frame_quality,
        reverse=True,
    )
    for rank, candidate in enumerate(candidates, 1):
        candidate.rank = rank
    return candidates
