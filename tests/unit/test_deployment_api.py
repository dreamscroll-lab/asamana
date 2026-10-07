"""``/api/deployment`` reports the gates the app was built with."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from world import WorldCatalog

from interaction.api import create_app


@pytest.mark.parametrize("enabled", [True, False])
def test_reports_the_dev_tools_gate(container, test_config, enabled: bool) -> None:
    test_config.web.dev_tools_enabled = enabled
    app = create_app(container, test_config, catalog=WorldCatalog(None))

    client = TestClient(app)
    assert client.get("/api/deployment").json() == {"dev_tools": enabled, "model_keys": True}
    # Ask over HTTP: FastAPI may keep an included router as a single entry in app.routes.
    assert (client.get("/api/templates/metro/map").status_code == 200) is enabled
