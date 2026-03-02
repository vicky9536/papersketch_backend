from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fastapi import FastAPI
from flask import Flask

from papersketch_backend.api.app import create_app as create_api_app
from papersketch_backend.ui.app import create_app as create_ui_app


class EntrypointSmokeTests(unittest.TestCase):
    def test_api_app_factory_returns_fastapi(self) -> None:
        app = create_api_app()

        self.assertIsInstance(app, FastAPI)

    def test_api_healthz_endpoint(self) -> None:
        app = create_api_app()

        from fastapi.testclient import TestClient

        with TestClient(app) as client:
            response = client.get("/healthz")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"ok": True})

    def test_ui_app_factory_returns_flask(self) -> None:
        app = create_ui_app()

        self.assertIsInstance(app, Flask)

    def test_ui_index_page_renders(self) -> None:
        app = create_ui_app()

        with app.test_client() as client:
            response = client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"openai:gpt-4o-mini", response.data)


if __name__ == "__main__":
    unittest.main()
