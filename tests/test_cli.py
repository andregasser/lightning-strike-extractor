from __future__ import annotations

import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from lse.cli import main


class CliTests(unittest.TestCase):
    def test_dry_run_accepts_multiple_inputs_and_deduplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "storm.mp4"
            video.write_bytes(b"video")
            output = StringIO()
            with redirect_stdout(output):
                exit_code = main(["analyze", str(video), str(video), "--dry-run"])
            self.assertEqual(exit_code, 0)
            self.assertIn("videos: 1", output.getvalue())
            self.assertIn("duplicates: 1", output.getvalue())

    def test_dry_run_returns_three_when_no_video_is_found(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "notes.txt").write_text("notes")
            with redirect_stdout(StringIO()):
                exit_code = main(["analyze", directory, "--dry-run"])
            self.assertEqual(exit_code, 3)

    def test_review_command_reports_label_counts(self) -> None:
        counts = {
            "lightning": 2,
            "not-lightning": 1,
            "uncertain": 0,
            "pending": 3,
        }
        output = StringIO()
        with (
            patch("lse.cli.review_candidates", return_value=(Path("labels.json"), counts)),
            redirect_stdout(output),
        ):
            exit_code = main(["review", "runs", "--no-open"])

        self.assertEqual(exit_code, 0)
        self.assertIn("lightning: 2", output.getvalue())
        self.assertIn("pending: 3", output.getvalue())


if __name__ == "__main__":
    unittest.main()
