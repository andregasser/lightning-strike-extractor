from __future__ import annotations

import io
import unittest

from tools.model_development.serve_label_studio import CorsRequestHandler


class LabelStudioServerTests(unittest.TestCase):
    def test_response_includes_open_cors_headers(self) -> None:
        handler = CorsRequestHandler.__new__(CorsRequestHandler)
        handler._headers_buffer = []
        handler.request_version = "HTTP/1.1"
        handler.wfile = io.BytesIO()

        handler.end_headers()

        headers = handler.wfile.getvalue().decode("latin-1")
        self.assertIn("Access-Control-Allow-Origin: *", headers)
        self.assertIn("Access-Control-Allow-Methods: GET, HEAD, OPTIONS", headers)


if __name__ == "__main__":
    unittest.main()
