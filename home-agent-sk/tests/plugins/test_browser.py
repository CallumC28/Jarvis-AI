from __future__ import annotations

from home_agent.plugins.browser import _resolve_url


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
