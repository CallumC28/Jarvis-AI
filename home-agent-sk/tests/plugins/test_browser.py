from __future__ import annotations

import pytest

from home_agent.plugins import browser
from home_agent.plugins.browser import BrowserPlugin, _resolve_url


def test_full_https_url_passthrough() -> None:
    assert _resolve_url("https://example.com") == "https://example.com"


def test_full_http_url_passthrough() -> None:
    assert _resolve_url("http://example.com/path") == "http://example.com/path"


def test_known_site_youtube() -> None:
    assert _resolve_url("youtube") == "https://www.youtube.com"


def test_known_site_github() -> None:
    assert _resolve_url("github") == "https://github.com"


def test_known_site_case_insensitive() -> None:
    assert _resolve_url("YouTube") == "https://www.youtube.com"


def test_bare_domain_gets_https_prefix() -> None:
    assert _resolve_url("bbc.co.uk") == "https://bbc.co.uk"


def test_bare_domain_with_subdomain() -> None:
    assert _resolve_url("docs.python.org") == "https://docs.python.org"


def test_unknown_phrase_falls_back_to_google_search() -> None:
    result = _resolve_url("latest python news")
    assert result.startswith("https://www.google.com/search?q=")
    assert "python" in result


def test_whitespace_is_stripped() -> None:
    assert _resolve_url("  youtube  ") == "https://www.youtube.com"


@pytest.mark.parametrize(
    "site",
    [
        'https://x.com" & calc & "',
        "https://x.com|calc",
        "https://x.com/<script>",
        "https://example.com/a b",
        'evil.com"&calc',
    ],
)
def test_unsafe_urls_become_a_search(site: str) -> None:
    assert _resolve_url(site).startswith("https://www.google.com/search?q=")


def test_query_string_ampersand_still_allowed() -> None:
    url = "https://example.com/search?a=1&b=2"
    assert _resolve_url(url) == url


def test_fallback_without_chrome_never_uses_a_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []
    monkeypatch.setattr(browser, "_find_chrome", lambda: None)
    monkeypatch.setattr(browser.webbrowser, "open", opened.append)

    def no_popen(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("subprocess must not be used without Chrome")

    monkeypatch.setattr(browser.subprocess, "Popen", no_popen)
    result = BrowserPlugin().open_website("https://example.com")
    assert opened == ["https://example.com"]
    assert "default browser" in result
