from __future__ import annotations

import logging
import os
import subprocess
import webbrowser
from typing import Annotated
from urllib.parse import quote_plus

from semantic_kernel.functions import kernel_function

logger = logging.getLogger(__name__)

# The site name comes from the model, which may have read untrusted web content, so anything
# that could break out of a URL (quotes, whitespace, shell metacharacters) is never opened as one.
_UNSAFE_URL_CHARS = frozenset("\"'<>^`|\\ \t\r\n")

_CHROME_CANDIDATES: list[str] = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
]

_KNOWN_SITES: dict[str, str] = {
    "youtube": "https://www.youtube.com",
    "google": "https://www.google.com",
    "gmail": "https://mail.google.com",
    "github": "https://github.com",
    "netflix": "https://www.netflix.com",
    "reddit": "https://www.reddit.com",
    "twitter": "https://twitter.com",
    "x": "https://x.com",
    "facebook": "https://www.facebook.com",
    "instagram": "https://www.instagram.com",
    "amazon": "https://www.amazon.co.uk",
    "spotify": "https://open.spotify.com",
    "twitch": "https://www.twitch.tv",
    "discord": "https://discord.com/app",
    "linkedin": "https://www.linkedin.com",
    "stackoverflow": "https://stackoverflow.com",
    "stack overflow": "https://stackoverflow.com",
    "wikipedia": "https://www.wikipedia.org",
    "bbc": "https://www.bbc.co.uk",
    "bbc news": "https://www.bbc.co.uk/news",
    "claude": "https://claude.ai",
    "chatgpt": "https://chat.openai.com",
    "openai": "https://openai.com",
    "whatsapp": "https://web.whatsapp.com",
    "outlook": "https://outlook.live.com",
    "onedrive": "https://onedrive.live.com",
}


def _find_chrome() -> str | None:
    for path in _CHROME_CANDIDATES:
        if os.path.isfile(path):
            return path
    return None


def _resolve_url(site: str) -> str:
    s = site.strip()
    if s.startswith(("http://", "https://")) and not _UNSAFE_URL_CHARS.intersection(s):
        return s
    key = s.lower()
    if key in _KNOWN_SITES:
        return _KNOWN_SITES[key]
    # Looks like a bare domain (e.g. "bbc.co.uk", "github.com")
    if "." in s and not _UNSAFE_URL_CHARS.intersection(s):
        return f"https://{s}"
    # Fall back to Google search
    return f"https://www.google.com/search?q={quote_plus(s)}"


class BrowserPlugin:
    """Opens Google Chrome and navigates to websites."""

    @kernel_function(
        name="open_website",
        description=(
            "Open Google Chrome and navigate to a website or URL. "
            "Use this when the user asks to open a browser, open a site, or go to a URL. "
            "Accepts site names ('youtube', 'github'), bare domains ('bbc.co.uk'), "
            "or full URLs ('https://example.com')."
        ),
    )
    def open_website(
        self,
        site: Annotated[str, "Site name, domain, or full URL to open in Chrome"],
    ) -> str:
        url = _resolve_url(site)
        chrome = _find_chrome()
        try:
            if chrome:
                subprocess.Popen([chrome, url])
            else:
                # Chrome not found at known paths: use the default browser (no shell involved)
                webbrowser.open(url)
                return f"Opened {url} in your default browser."
            return f"Opened {url} in Chrome."
        except Exception as exc:
            logger.exception("Failed to open Chrome.")
            return f"Could not open Chrome: {exc}"
