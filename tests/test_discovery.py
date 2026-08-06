from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from lightning_extractor.discovery import discover_inputs


class DiscoveryTests(unittest.TestCase):
    def test_recursive_discovery_deduplicates_and_ignores_output_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.mp4"
            first.write_bytes(b"video")
            (root / "second.mov").write_bytes(b"video-2")
            os.link(first, root / "z-duplicate.mp4")
            (root / "notes.txt").write_text("not a video")
            (root / "nested").mkdir()
            (root / "nested" / "third.mkv").write_bytes(b"video-3")
            (root / "runs").mkdir()
            (root / "runs" / "generated.mp4").write_bytes(b"generated")

            result = discover_inputs([root], recursive=True)

            self.assertEqual(
                [path.name for path in result.videos], ["first.mp4", "third.mkv", "second.mov"]
            )
            self.assertEqual(len(result.duplicates), 1)
            self.assertNotIn("generated.mp4", [path.name for path in result.videos])

    def test_include_and_exclude_patterns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "storm-main.mp4").write_bytes(b"main")
            (root / "storm-preview.mp4").write_bytes(b"preview")
            (root / "sunset.mp4").write_bytes(b"sunset")
            result = discover_inputs([root], includes=["storm-*.mp4"], excludes=["*preview*"])
            self.assertEqual([path.name for path in result.videos], ["storm-main.mp4"])


if __name__ == "__main__":
    unittest.main()
