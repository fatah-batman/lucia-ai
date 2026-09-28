"""
tools.py - the "actions" LUCIA can perform.

Each function has a type-hinted signature and a docstring. Ollama reads these
to build the tool schema automatically, so KEEP THE DOCSTRINGS CLEAR - the
model uses them to decide when to call each tool.

All tools here are free and need no API keys.
"""

import datetime
import platform
import subprocess

import requests

try:
    from ddgs import DDGS  # newer package name
except ImportError:  # fallback for older installs
    from duckduckgo_search import DDGS


def get_current_time() -> str:
    """Get the current local date and time."""
    now = datetime.datetime.now()
    return now.strftime("%A, %d %B %Y, %I:%M %p")


def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: Name of the city, e.g. "Bengaluru" or "London".
    """
    try:
        geo_resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=10,
        )
        geo_resp.raise_for_status()
        geo = geo_resp.json()
        if not geo.get("results"):
            return f"Could not find a city called '{city}'."
        place = geo["results"][0]

        wx_resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m,precipitation",
            },
            timeout=10,
        )
        wx_resp.raise_for_status()
        wx_data = wx_resp.json()
        if "current" not in wx_data:
            return f"Weather data currently unavailable for '{place.get('name', city)}'."
        wx = wx_data["current"]

        return (
            f"Weather in {place['name']}, {place.get('country', '')}: "
            f"{wx['temperature_2m']}°C (feels like {wx['apparent_temperature']}°C), "
            f"humidity {wx['relative_humidity_2m']}%, "
            f"wind {wx['wind_speed_10m']} km/h, "
            f"precipitation {wx['precipitation']} mm."
        )
    except Exception as e:
        return f"Weather lookup failed: {e}"


def web_search(query: str) -> str:
    """Search the web for current information, news, or facts.

    Args:
        query: What to search for.
    """
    try:
        with DDGS(timeout=10) as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return "No results found."
        lines = []
        for r in results:
            lines.append(f"- {r['title']}: {r['body']} ({r['href']})")
        return "\n".join(lines)
    except Exception as e:
        return f"Web search failed: {e}"


from config import APP_WHITELIST


def open_app(app_name: str) -> str:
    """Open an application on the user's computer.

    Args:
        app_name: Name of the app, e.g. "calculator", "chrome", "vscode".
    """
    system = platform.system()
    apps = APP_WHITELIST.get(system, {})
    key = app_name.lower().strip()

    if key not in apps:
        return f"I'm not allowed to open '{app_name}'. Available: {', '.join(apps) or 'none'}."

    target = apps[key]
    try:
        if system == "Windows":
            subprocess.Popen(f'start "" "{target}"', shell=True)
        elif system == "Darwin":
            subprocess.Popen(["open", "-a", target])
        else:
            subprocess.Popen([target])
        return f"Opened {app_name}."
    except Exception as e:
        return f"Failed to open {app_name}: {e}"


# Registry used by main.py: name -> function
TOOLS = {
    "get_current_time": get_current_time,
    "get_weather": get_weather,
    "web_search": web_search,
    "open_app": open_app,
}
