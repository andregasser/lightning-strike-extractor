from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lse.config import Config
from lse.models import CandidateFrame
from lse.pipeline import analyze, select_export_candidates

FIXTURES = Path(__file__).parent / "fixtures" / "reference"


class ReferenceVideoTests(unittest.TestCase):
    def test_detects_and_selects_confirmed_lightning_channels(self) -> None:
        truth = json.loads((FIXTURES / "ground-truth.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            runs = Path(directory) / "runs"
            for fixture in truth["fixtures"]:
                with self.subTest(video=fixture["file"]):
                    config = Config()
                    config.export.top = 1
                    run = analyze(
                        FIXTURES / fixture["file"],
                        runs,
                        config,
                        progress_mode="quiet",
                    )
                    candidates = [
                        CandidateFrame(**row)
                        for row in json.loads((run / "results" / "candidates.json").read_text())
                    ]
                    winners = select_export_candidates(candidates, config)

                    self.assertEqual(len(winners), 1)
                    winner = winners[0]
                    self.assertAlmostEqual(
                        winner.time,
                        fixture["expected_winner_seconds"],
                        delta=fixture["winner_tolerance_seconds"],
                    )
                    self.assertGreaterEqual(
                        winner.channel_length, fixture["minimum_channel_length"]
                    )
                    self.assertGreaterEqual(
                        winner.channel_strength, fixture["minimum_channel_strength"]
                    )


if __name__ == "__main__":
    unittest.main()
