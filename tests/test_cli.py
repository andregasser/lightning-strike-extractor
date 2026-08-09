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

    def test_dataset_export_command_uses_cli_subcommand(self) -> None:
        manifest = {"sources": [{"source_video": "storm.mp4"}], "frames": [{"file_name": "frame.jpg"}]}
        with (
            patch("lse.cli.export_frame_handoff", return_value=manifest) as export,
            redirect_stdout(StringIO()) as output,
        ):
            exit_code = main(
                [
                    "dataset-export",
                    "runs",
                    "--output",
                    "handoffs/example",
                    "--max-events-per-video",
                    "25",
                    "--context-frames",
                    "3",
                ]
            )

        self.assertEqual(exit_code, 0)
        export.assert_called_once()
        self.assertEqual(export.call_args.args[0], Path("runs"))
        self.assertEqual(export.call_args.args[1], Path("handoffs/example"))
        self.assertEqual(export.call_args.kwargs["max_events_per_video"], 25)
        self.assertEqual(export.call_args.kwargs["context_frames"], 3)
        self.assertIn("sources: 1", output.getvalue())


if __name__ == "__main__":
    unittest.main()
