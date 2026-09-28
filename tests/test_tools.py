"""tests/test_tools.py - Unit tests for LUCIA tools without calling the LLM."""

import platform
from unittest.mock import MagicMock, patch
import pytest

from tools import get_current_time, get_weather, web_search, open_app, APP_WHITELIST


def test_get_current_time():
    """Verify get_current_time returns a non-empty formatted date and time string."""
    result = get_current_time()
    assert isinstance(result, str)
    assert len(result) > 0
    # Should include time indicator (AM or PM)
    assert "AM" in result or "PM" in result


@patch("tools.requests.get")
def test_get_weather_success(mock_get):
    """Verify get_weather successfully parses geocoding and forecast responses."""
    # Mock responses for geocoding and forecast APIs
    geo_mock = MagicMock()
    geo_mock.json.return_value = {
        "results": [{"name": "Bengaluru", "country": "India", "latitude": 12.97, "longitude": 77.59}]
    }

    forecast_mock = MagicMock()
    forecast_mock.json.return_value = {
        "current": {
            "temperature_2m": 24.5,
            "apparent_temperature": 25.0,
            "relative_humidity_2m": 60,
            "wind_speed_10m": 12.3,
            "precipitation": 0.0,
        }
    }

    mock_get.side_effect = [geo_mock, forecast_mock]

    result = get_weather("Bengaluru")
    assert "Weather in Bengaluru, India:" in result
    assert "24.5°C" in result
    assert "feels like 25.0°C" in result
    assert "humidity 60%" in result
    assert "wind 12.3 km/h" in result
    assert "precipitation 0.0 mm" in result


@patch("tools.requests.get")
def test_get_weather_city_not_found(mock_get):
    """Verify get_weather handles unknown cities gracefully."""
    geo_mock = MagicMock()
    geo_mock.json.return_value = {"results": []}
    mock_get.return_value = geo_mock

    result = get_weather("NonExistentCityXYZ")
    assert "Could not find a city called 'NonExistentCityXYZ'." in result


@patch("tools.requests.get")
def test_get_weather_network_error(mock_get):
    """Verify get_weather catches request timeouts or connection failures."""
    mock_get.side_effect = Exception("Connection timeout")

    result = get_weather("Bengaluru")
    assert "Weather lookup failed: Connection timeout" in result


@patch("tools.DDGS")
def test_web_search_success(mock_ddgs_class):
    """Verify web_search formats results as expected."""
    mock_instance = MagicMock()
    mock_instance.text.return_value = [
        {
            "title": "SpaceX News",
            "body": "SpaceX completes Starship flight test successfully.",
            "href": "https://example.com/spacex",
        }
    ]
    # Support both DDGS().text and with DDGS() as ddgs:
    mock_instance.__enter__.return_value = mock_instance
    mock_ddgs_class.return_value = mock_instance

    result = web_search("SpaceX")
    assert "- SpaceX News: SpaceX completes Starship flight test successfully. (https://example.com/spacex)" in result


@patch("tools.DDGS")
def test_web_search_empty(mock_ddgs_class):
    """Verify web_search handles empty search results."""
    mock_instance = MagicMock()
    mock_instance.text.return_value = []
    mock_instance.__enter__.return_value = mock_instance
    mock_ddgs_class.return_value = mock_instance

    result = web_search("random unsearchable query 99999")
    assert result == "No results found."


@patch("tools.DDGS")
def test_web_search_error(mock_ddgs_class):
    """Verify web_search handles network exceptions gracefully."""
    mock_instance = MagicMock()
    mock_instance.text.side_effect = Exception("Rate limit exceeded")
    mock_instance.__enter__.return_value = mock_instance
    mock_ddgs_class.return_value = mock_instance

    result = web_search("query")
    assert "Web search failed: Rate limit exceeded" in result


@patch("tools.subprocess.Popen")
def test_open_app_allowed(mock_popen):
    """Verify open_app allows whitelisted apps and calls subprocess."""
    system = platform.system()
    apps = APP_WHITELIST.get(system, {})
    if not apps:
        pytest.skip(f"No apps configured in whitelist for system {system}")

    test_app = next(iter(apps.keys()))
    result = open_app(test_app)
    assert result == f"Opened {test_app}."
    assert mock_popen.called


@patch("tools.subprocess.Popen")
def test_open_app_disallowed(mock_popen):
    """Verify open_app rejects unauthorized applications and prevents execution."""
    result = open_app("unauthorized_shell_script")
    assert "I'm not allowed to open 'unauthorized_shell_script'." in result
    assert not mock_popen.called
