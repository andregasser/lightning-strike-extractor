from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2

from lightning_extractor.config import Config
from lightning_extractor.models import CandidateFrame
from lightning_extractor.pipeline import export_stills, select_export_candidates


def candidate(rank: int, event: str, score: float) -> CandidateFrame:
    return CandidateFrame(rank, event, rank, rank / 100.0, score, 1, 10.0)


class ExportSelectionTests(unittest.TestCase):
    def test_contact_sheet_places_context_around_peak(self) -> None:
        config = Config()
        config.export.minimum_geometry_score = 0.0
        config.export.contact_sheet_context_frames = 2
        fixture = (
            Path(__file__).parent
            / "fixtures"
            / "reference"
            / "positive"
            / "gx020425-night-channel.avi"
        )
        row = CandidateFrame(
            1,
            "event-a",
            50,
            0.5,
            100.0,
            1,
            10.0,
            frame_quality=100.0,
        )

        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "exports" / "stills"
            exported = export_stills(fixture, [row], output, config, progress_mode="never")
            sheet = cv2.imread(str(output.parent / "contact-sheet.jpg"))

        self.assertEqual(exported, 1)
        self.assertIsNotNone(sheet)
        assert sheet is not None
        self.assertEqual(sheet.shape[1], 5 * 640)

    def test_selects_one_qualified_frame_per_event(self) -> None:
        config = Config()
        config.export.minimum_geometry_score = 500.0
        rows = [
            candidate(1, "event-a", 900.0),
            candidate(2, "event-a", 800.0),
            candidate(3, "event-b", 700.0),
            candidate(4, "event-c", 499.0),
        ]

        selected = select_export_candidates(rows, config)

        self.assertEqual([row.rank for row in selected], [1, 3])

    def test_top_is_an_upper_limit(self) -> None:
        config = Config()
        config.export.top = 1
        rows = [candidate(1, "event-a", 900.0), candidate(2, "event-b", 800.0)]

        selected = select_export_candidates(rows, config)

        self.assertEqual([row.rank for row in selected], [1])

    def test_winner_is_longest_clear_frame_within_plausible_geometry(self) -> None:
        config = Config()
        rows = [
            candidate(1, "event-a", 100.0),
            candidate(2, "event-a", 60.0),
            candidate(3, "event-a", 10.0),
        ]
        rows[0].frame_quality = 100.0
        rows[1].frame_quality = 500.0
        rows[2].frame_quality = 10_000.0

        selected = select_export_candidates(rows, config)

        self.assertEqual([row.rank for row in selected], [2])

    def test_strongest_frame_wins_despite_multiframe_bonus(self) -> None:
        config = Config()
        rows = [candidate(1, "event-a", 100.0), candidate(2, "event-a", 90.0)]
        rows[0].frame_quality = 100.0
        rows[0].multiframe_quality = 100.0
        rows[1].frame_quality = 95.0
        rows[1].multiframe_quality = 110.0

        selected = select_export_candidates(rows, config)

        self.assertEqual([row.rank for row in selected], [1])

    def test_multiframe_support_breaks_equal_strength_tie(self) -> None:
        config = Config()
        rows = [candidate(1, "event-a", 100.0), candidate(2, "event-a", 90.0)]
        rows[0].frame_quality = 100.0
        rows[0].multiframe_support = 0.2
        rows[1].frame_quality = 100.0
        rows[1].multiframe_support = 0.9

        selected = select_export_candidates(rows, config)

        self.assertEqual([row.rank for row in selected], [2])


if __name__ == "__main__":
    unittest.main()
