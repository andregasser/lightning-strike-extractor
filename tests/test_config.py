from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from lightning_extractor.config import load_config


class ConfigTests(unittest.TestCase):
    def test_load_config_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.toml"
            path.write_text("[export]\ntop = 12\n")
            self.assertEqual(load_config(path).export.top, 12)

    def test_unknown_setting_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.toml"
            path.write_text("[analysis]\nmagic = 12\n")
            with self.assertRaisesRegex(ValueError, "analysis.magic"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
