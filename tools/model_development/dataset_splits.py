from __future__ import annotations

import math

SPLITS = ("train", "validation", "test")


def validate_split_ratios(train: float, validation: float, test: float) -> dict[str, float]:
    ratios = {"train": train, "validation": validation, "test": test}
    if any(not math.isfinite(value) or value < 0 for value in ratios.values()):
        raise ValueError("Split ratios must be finite and non-negative")
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0")
    return ratios


def assign_sources_to_splits(
    image_counts: dict[str, int], ratios: dict[str, float]
) -> dict[str, str]:
    active = [name for name in SPLITS if ratios[name] > 0]
    if not active:
        raise ValueError("At least one split ratio must be positive")
    total = sum(image_counts.values())
    targets = {name: total * ratios[name] for name in active}
    assigned_counts = {name: 0 for name in active}
    assignments: dict[str, str] = {}
    ordered = sorted(image_counts, key=lambda source: (-image_counts[source], source))
    for index, source in enumerate(ordered):
        empty = [name for name in active if assigned_counts[name] == 0]
        remaining = len(ordered) - index
        eligible = empty if empty and remaining <= len(empty) else active
        split = max(
            eligible,
            key=lambda name: (
                targets[name] - assigned_counts[name],
                ratios[name],
                -active.index(name),
            ),
        )
        assignments[source] = split
        assigned_counts[split] += image_counts[source]
    return assignments
