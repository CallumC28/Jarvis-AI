from __future__ import annotations

from unittest.mock import MagicMock, patch

from home_agent.plugins.pc import PCPlugin, _best_match, _parse_volume, foreground_title

APPS = ["Google Chrome", "Spotify", "Visual Studio Code", "Microsoft Word", "WordPad", "Steam"]


def test_spoken_names_match_start_menu_apps():
    def pick(q: str) -> str | None:
        i = _best_match(q, APPS)
        return APPS[i] if i is not None else None

    assert pick("spotify") == "Spotify"
    assert pick("chrome") == "Google Chrome"          # contains
    assert pick("visual studio") == "Visual Studio Code"  # prefix
    assert pick("word") == "Microsoft Word"            # whole word beats prefix ("WordPad")
    assert pick("wordp") == "WordPad"
    assert pick("spotfy") == "Spotify"                 # mis-transcribed
    assert pick("photoshop") is None


def test_foreground_title_never_raises():
    assert isinstance(foreground_title(), str)


def test_volume_requests_are_absolute_or_relative_and_clamped():
    assert _parse_volume("30", 99) == 30
    assert _parse_volume("30%", 99) == 30
    assert _parse_volume("+10", 50) == 60
    assert _parse_volume("-10", 5) == 0
    assert _parse_volume("+20", 95) == 100


def test_open_uri_only_opens_allowlisted_app_links():
    with patch("home_agent.plugins.pc.os.startfile") as startfile:
        assert "Can't open" in PCPlugin.open_uri(MagicMock(), "ms-msdt:/id PCWDiagnostic")
        assert "Can't open" in PCPlugin.open_uri(MagicMock(), "file:///C:/Windows/System32/cmd.exe")
        startfile.assert_not_called()
        assert PCPlugin.open_uri(MagicMock(), "steam://store").startswith("Opened")
        startfile.assert_called_once_with("steam://store")


def test_steam_openurl_only_reaches_steams_own_pages():
    plugin = MagicMock()
    with patch("home_agent.plugins.pc.os.startfile") as startfile:
        assert "only" in PCPlugin.open_uri(plugin, "steam://openurl/https://evil.example/login")
        assert "only" in PCPlugin.open_uri(
            plugin, "steam://openurl/https://store.steampowered.com.evil.example/"
        )
        startfile.assert_not_called()
        PCPlugin.open_uri(plugin, "steam://openurl/https://store.steampowered.com/")
        startfile.assert_called_once()
