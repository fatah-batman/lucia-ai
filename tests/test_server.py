"""tests/test_server.py - Tests for FastAPI web dashboard and WebSocket endpoints."""

import pytest
from fastapi.testclient import TestClient

from server import app, manager


def test_index_route_serves_html():
    """Verify GET / returns 200 OK and serves HTML content."""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "LUCIA" in response.text
    assert "orbCanvas" in response.text


def test_static_files_accessible():
    """Verify static CSS and JS files are served correctly."""
    client = TestClient(app)
    css_resp = client.get("/static/style.css")
    assert css_resp.status_code == 200
    assert "dashboard-container" in css_resp.text

    js_resp = client.get("/static/app.js")
    assert js_resp.status_code == 200
    assert "ParticlesOrb" in js_resp.text

    orb_resp = client.get("/static/particles-orb.js")
    assert orb_resp.status_code == 200
    assert "buildSphere" in orb_resp.text




def test_websocket_connect_and_state():
    """Verify WebSocket connection connects and receives initial state."""
    client = TestClient(app)
    with client.websocket_connect("/ws") as websocket:
        data = websocket.receive_json()
        assert data.get("type") == "state"
        assert data.get("value") in ("idle", "listening", "processing", "speaking")

        # Test sending interrupt action from web client
        websocket.send_json({"action": "interrupt"})


def test_event_manager_state_tracking():
    """Verify EventManager updates state accurately."""
    manager.set_state("listening")
    assert manager.current_state == "listening"
    manager.set_state("idle")
    assert manager.current_state == "idle"
