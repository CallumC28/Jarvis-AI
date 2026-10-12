from __future__ import annotations

import pytest

from home_agent.plugins.screenshot import _resolve_monitor


# Fake monitor list: left=0 primary, left=1920 right monitor
MONITORS = [
    {"left": 0,    "top": 0, "width": 1920, "height": 1080},
    {"left": 1920, "top": 0, "width": 2560, "height": 1440},
]


def test_resolve_by_index_screen_1():
    assert _resolve_monitor("screen 1", MONITORS) == MONITORS[0]


def test_resolve_by_index_screen_2():
    assert _resolve_monitor("screen 2", MONITORS) == MONITORS[1]


def test_resolve_primary():
    assert _resolve_monitor("main", MONITORS) == MONITORS[0]


def test_resolve_primary_keyword():
    assert _resolve_monitor("primary monitor", MONITORS) == MONITORS[0]


def test_resolve_left():
    assert _resolve_monitor("left monitor", MONITORS) == MONITORS[0]


def test_resolve_right():
    assert _resolve_monitor("right screen", MONITORS) == MONITORS[1]


def test_resolve_ambiguous_defaults_to_primary():
    assert _resolve_monitor("my screen", MONITORS) == MONITORS[0]


def test_resolve_empty_reference_defaults_to_primary():
    assert _resolve_monitor("", MONITORS) == MONITORS[0]


def test_resolve_raises_when_no_monitors():
    with pytest.raises(ValueError, match="No monitors"):
        _resolve_monitor("screen 1", [])
