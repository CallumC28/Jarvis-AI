"""Windows desktop control: apps, windows, media, volume, clipboard and typing."""

from __future__ import annotations

import ctypes
import difflib
import functools
import json
import logging
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Annotated

import pythoncom
import win32api
import win32clipboard
import win32con
import win32gui
import win32process
from pycaw.pycaw import AudioUtilities
from semantic_kernel.functions import kernel_function

from home_agent.plugins.screenshot import _get_monitors, _resolve_monitor

logger = logging.getLogger(__name__)

_DWMWA_CLOAKED = 14                 # suspended Store apps are "visible" but cloaked
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_MAX_CLIPBOARD_CHARS = 8_000

# App links the model may open. An allowlist, not a blocklist: some Windows URI handlers
# (ms-msdt:, search-ms:) have been used to run code, and tool arguments can be steered by
# text the model read (web results, clipboard, screen).
_URI_SCHEMES = frozenset({
    "steam", "spotify", "ms-settings", "discord", "mailto", "vscode", "ms-windows-store",
})
_MEDIA_KEYS = {
    "play_pause": win32con.VK_MEDIA_PLAY_PAUSE,
    "next": win32con.VK_MEDIA_NEXT_TRACK,
    "previous": win32con.VK_MEDIA_PREV_TRACK,
    "mute": win32con.VK_VOLUME_MUTE,
}


# ── Windows ───────────────────────────────────────────────────────────────────

def _is_cloaked(hwnd: int) -> bool:
    cloaked = ctypes.c_int(0)
    ctypes.windll.dwmapi.DwmGetWindowAttribute(
        hwnd, _DWMWA_CLOAKED, ctypes.byref(cloaked), ctypes.sizeof(cloaked)
    )
    return bool(cloaked.value)


def _exe_name(hwnd: int) -> str:
    _, pid = win32process.GetWindowThreadProcessId(hwnd)
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(260)
        size = ctypes.c_ulong(len(buf))
        ok = kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size))
        return Path(buf.value).stem if ok else ""
    finally:
        kernel32.CloseHandle(handle)


def _windows() -> list[int]:
    """Real app windows of other processes, front-most first (EnumWindows is z-ordered)."""
    own_pid = os.getpid()
    found: list[int] = []

    def visit(hwnd: int, _: object) -> bool:
        if (
            win32gui.IsWindowVisible(hwnd)
            and win32gui.GetWindowText(hwnd)
            and not win32gui.GetWindow(hwnd, win32con.GW_OWNER)
            and not win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE) & win32con.WS_EX_TOOLWINDOW
            and win32gui.GetClassName(hwnd) != "Progman"   # the desktop itself
            and win32process.GetWindowThreadProcessId(hwnd)[1] != own_pid
            and not _is_cloaked(hwnd)
        ):
            found.append(hwnd)
        return True

    win32gui.EnumWindows(visit, None)
    return found


def foreground_window() -> int | None:
    """The window the user is working in — skipping Jarvis's own windows."""
    windows = _windows()
    fg = win32gui.GetForegroundWindow()
    if fg in windows:
        return int(fg)
    return windows[0] if windows else None


def foreground_title() -> str:
    try:
        hwnd = foreground_window()
        return win32gui.GetWindowText(hwnd) if hwnd else ""
    except Exception:
        logger.debug("Could not read the foreground window.", exc_info=True)
        return ""


def _best_match(query: str, names: list[str]) -> int | None:
    """Index of the name a spoken app name most likely means.

    exact > whole words ("word" → Microsoft Word, not WordPad) > prefix > contains > fuzzy.
    """
    q = query.lower().strip()
    lowered = [n.lower() for n in names]
    whole_words = re.compile(rf"\b{re.escape(q)}\b")
    tests = (str.__eq__, lambda n, _: bool(whole_words.search(n)), str.startswith, str.__contains__)
    for test in tests:
        hits = [i for i, n in enumerate(lowered) if test(n, q)]
        if hits:
            return min(hits, key=lambda i: len(lowered[i]))
    close = difflib.get_close_matches(q, lowered, n=1, cutoff=0.6)
    return lowered.index(close[0]) if close else None


def _find_window(app: str) -> int | None:
    if not app.strip():
        return foreground_window()
    keys: list[str] = []
    owners: list[int] = []
    for hwnd in _windows():
        for key in (win32gui.GetWindowText(hwnd), _exe_name(hwnd)):
            keys.append(key)
            owners.append(hwnd)
    i = _best_match(app, keys)
    return owners[i] if i is not None else None


def _hotkey(*keys: int) -> None:
    for k in keys:
        win32api.keybd_event(k, 0, 0, 0)
    for k in reversed(keys):
        win32api.keybd_event(k, 0, win32con.KEYEVENTF_KEYUP, 0)


def _focus(hwnd: int) -> None:
    if win32gui.IsIconic(hwnd):
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    _hotkey(win32con.VK_MENU)   # Windows only lets the process that sent the last input steal focus
    win32gui.SetForegroundWindow(hwnd)


# ── Apps ──────────────────────────────────────────────────────────────────────

# ponytail: read once per run — apps installed while Jarvis is running need a restart.
@functools.cache
def _start_apps() -> dict[str, str]:
    """Start-menu apps (desktop and Store) → AppID, as listed by Get-StartApps."""
    command = (
        "[Console]::OutputEncoding=[Text.Encoding]::UTF8;"
        "Get-StartApps | ConvertTo-Json -Compress"
    )
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True, encoding="utf-8", timeout=30,
        creationflags=subprocess.CREATE_NO_WINDOW,
    ).stdout
    apps = json.loads(out or "[]")
    if isinstance(apps, dict):
        apps = [apps]
    return {a["Name"]: a["AppID"] for a in apps}


# ── Clipboard / typing ────────────────────────────────────────────────────────

def _get_clipboard() -> str:
    win32clipboard.OpenClipboard()
    try:
        if win32clipboard.IsClipboardFormatAvailable(win32con.CF_UNICODETEXT):
            return str(win32clipboard.GetClipboardData(win32con.CF_UNICODETEXT))
        return ""
    finally:
        win32clipboard.CloseClipboard()


def _set_clipboard(text: str) -> None:
    win32clipboard.OpenClipboard()
    try:
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardText(text, win32con.CF_UNICODETEXT)
    finally:
        win32clipboard.CloseClipboard()


def paste_text(text: str) -> str:
    """Type text into the window the user is working in (via the clipboard — handles any script)."""
    _set_clipboard(text)
    hwnd = foreground_window()
    if hwnd is None:
        return "No window to paste into; the text is on the clipboard."
    if win32gui.GetForegroundWindow() != hwnd:
        _focus(hwnd)
        time.sleep(0.15)   # let the focus change land before the keystroke
    _hotkey(win32con.VK_CONTROL, ord("V"))
    return f"Pasted into {win32gui.GetWindowText(hwnd)}."


def _parse_volume(request: str, current: int) -> int:
    """'30' → 30, '+10' → current + 10, '-10' → current - 10; clamped to 0–100."""
    text = request.strip().rstrip("%")
    value = int(text)
    target = current + value if text[0] in "+-" else value
    return max(0, min(100, target))


class PCPlugin:
    """Open, close, switch and arrange apps; read and write the clipboard; type into windows."""

    def __init__(self) -> None:
        # Get-StartApps takes a second or two — warm it up so the first "open X" is instant.
        threading.Thread(target=_start_apps, daemon=True).start()

    @kernel_function(
        name="open_app",
        description="Launch any installed app by name (Start menu), e.g. 'spotify', 'word'.",
    )
    def open_app(self, name: Annotated[str, "App name as the user said it"]) -> str:
        apps = _start_apps()
        names = list(apps)
        i = _best_match(name, names)
        if i is None:
            return f"No app called '{name}' in the Start menu."
        os.startfile(f"shell:AppsFolder\\{apps[names[i]]}")
        return f"Opened {names[i]}."

    @kernel_function(
        name="close_app",
        description="Close an app window by name. Empty name closes the window the user is in.",
    )
    def close_app(self, name: Annotated[str, "App or window name"] = "") -> str:
        hwnd = _find_window(name)
        if hwnd is None:
            return f"No open window matches '{name}'."
        title = win32gui.GetWindowText(hwnd)
        win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)   # polite: the app may ask to save
        return f"Closed {title}."

    @kernel_function(
        name="switch_to_app",
        description="Bring an already-open app window to the front.",
    )
    def switch_to_app(self, name: Annotated[str, "App or window name"]) -> str:
        hwnd = _find_window(name)
        if hwnd is None:
            return f"No open window matches '{name}'. Use open_app to launch it."
        _focus(hwnd)
        return f"Switched to {win32gui.GetWindowText(hwnd)}."

    @kernel_function(
        name="move_window",
        description="Move an app window to another monitor ('screen 2', 'left', 'main').",
    )
    def move_window(
        self,
        name: Annotated[str, "App or window name; empty for the window the user is in"],
        monitor_reference: Annotated[str, "Target monitor, e.g. 'screen 2', 'left', 'main'"],
    ) -> str:
        hwnd = _find_window(name)
        if hwnd is None:
            return f"No open window matches '{name}'."
        monitor = _resolve_monitor(monitor_reference, _get_monitors())
        was_maximised = win32gui.IsZoomed(hwnd)
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        w, h = min(right - left, monitor["width"]), min(bottom - top, monitor["height"])
        x = monitor["left"] + (monitor["width"] - w) // 2
        y = monitor["top"] + (monitor["height"] - h) // 2
        win32gui.SetWindowPos(hwnd, 0, x, y, w, h, win32con.SWP_NOZORDER)
        if was_maximised:
            win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)   # re-maximises on the new monitor
        return f"Moved {win32gui.GetWindowText(hwnd)}."

    @kernel_function(
        name="arrange_window",
        description=(
            "Resize an app window: 'maximise', 'minimise', 'restore', 'left' (snap to left "
            "half of its screen) or 'right'."
        ),
    )
    def arrange_window(
        self,
        name: Annotated[str, "App or window name; empty for the window the user is in"],
        layout: Annotated[str, "maximise | minimise | restore | left | right"],
    ) -> str:
        hwnd = _find_window(name)
        if hwnd is None:
            return f"No open window matches '{name}'."
        layout = layout.lower()
        if "max" in layout:
            win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
        elif "min" in layout:
            win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
        elif "left" in layout or "right" in layout:
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            monitor = win32api.MonitorFromWindow(hwnd, win32con.MONITOR_DEFAULTTONEAREST)
            left, top, right, bottom = win32api.GetMonitorInfo(monitor)["Work"]
            half = (right - left) // 2
            x = left if "left" in layout else left + half
            # ponytail: ignores DWM's invisible borders, so snapped halves have a few-px gap.
            win32gui.SetWindowPos(hwnd, 0, x, top, half, bottom - top, win32con.SWP_NOZORDER)
        elif "restore" in layout or "normal" in layout:
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        else:
            return f"Unknown layout '{layout}'."
        return f"Done: {win32gui.GetWindowText(hwnd)} → {layout}."

    @kernel_function(
        name="open_uri",
        description=(
            "Open a page inside an app via its link. Steam: store front = "
            "steam://openurl/https://store.steampowered.com/ , find a game = "
            "steam://openurl/https://store.steampowered.com/search/?term=<words> , library = "
            "steam://open/games, friends = steam://open/friends. Also spotify:search:<words>, "
            "ms-settings:bluetooth "
            "(also :display, :sound, :network-wifi, :windowsupdate, :apps), discord://, "
            "mailto:<address>, vscode://. Not for websites — use browser.open_website."
        ),
    )
    def open_uri(self, uri: Annotated[str, "The app link, e.g. 'steam://open/games'"]) -> str:
        scheme = uri.split(":", 1)[0].strip().lower()
        if scheme not in _URI_SCHEMES:
            return f"Can't open '{scheme}:' links. Allowed: {', '.join(sorted(_URI_SCHEMES))}."
        if uri.strip().lower().startswith("steam://openurl/") and not re.match(
            r"steam://openurl/https://(store\.steampowered\.com|steamcommunity\.com)(/|$)",
            uri.strip(), re.IGNORECASE,
        ):
            return "Steam can only be sent to its own store and community pages."
        os.startfile(uri.strip())
        return f"Opened {uri}."

    @kernel_function(
        name="media",
        description="Control what's playing (Spotify, YouTube…): play_pause, next, previous, mute.",
    )
    def media(self, action: Annotated[str, "play_pause | next | previous | mute"]) -> str:
        key = _MEDIA_KEYS.get(action.strip().lower())
        if key is None:
            return f"Unknown media action '{action}'."
        _hotkey(key)
        return f"Sent {action}."

    @kernel_function(
        name="set_volume",
        description="Set the PC's master volume: '30' (percent), '+10' or '-10' (relative).",
    )
    def set_volume(self, level: Annotated[str, "Percent like '30', or '+10' / '-10'"]) -> str:
        pythoncom.CoInitialize()   # COM is per-thread; tools run off the main thread
        endpoint = AudioUtilities.GetSpeakers().EndpointVolume
        before = round(endpoint.GetMasterVolumeLevelScalar() * 100)
        try:
            target = _parse_volume(level, before)
        except (ValueError, IndexError):
            return f"Didn't understand volume '{level}'."
        endpoint.SetMute(0, None)
        endpoint.SetMasterVolumeLevelScalar(target / 100, None)
        return f"Volume {target}% (was {before}%)."

    @kernel_function(name="minimise_all", description="Minimise every window to show the desktop.")
    def minimise_all(self) -> str:
        _hotkey(win32con.VK_LWIN, ord("M"))
        return "All windows minimised."

    @kernel_function(
        name="read_clipboard",
        description="Read the text the user has copied, to read out, summarise or rewrite it.",
    )
    def read_clipboard(self) -> str:
        text = _get_clipboard()
        if not text:
            return "The clipboard has no text."
        return text[:_MAX_CLIPBOARD_CHARS]

    @kernel_function(name="copy_to_clipboard", description="Put text on the user's clipboard.")
    def copy_to_clipboard(self, text: Annotated[str, "Text to copy"]) -> str:
        _set_clipboard(text)
        return "Copied."

    @kernel_function(
        name="paste_text",
        description=(
            "Type text into the window the user is working in (e.g. a rewritten version of "
            "what they copied, or anything they ask you to write there)."
        ),
    )
    def paste_text(self, text: Annotated[str, "The exact text to insert"]) -> str:
        return paste_text(text)
