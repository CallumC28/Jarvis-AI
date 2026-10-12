from __future__ import annotations

import logging
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import mss
from PIL import Image
from semantic_kernel.functions import kernel_function

logger = logging.getLogger(__name__)

_MAX_SIDE = 1920   # a 4K screen would cost the vision model ~4x the tokens for little gain


def _get_monitors() -> list[dict]:
    """Return individual monitors (mss index 0 is the combined virtual screen — skip it)."""
    with mss.mss() as sct:
        return list(sct.monitors[1:])


def _resolve_monitor(reference: str, monitors: list[dict]) -> dict:
    """Map a natural language monitor reference to a monitor dict.

    Raises ValueError if the monitors list is empty.
    """
    if not monitors:
        raise ValueError("No monitors found.")

    ref = reference.lower().strip()

    # Numeric: "screen 1", "monitor 2", "1", "2" → 1-based index
    for word in ref.split():
        if word.isdigit():
            idx = int(word) - 1
            if 0 <= idx < len(monitors):
                return monitors[idx]

    # Named positions
    if any(k in ref for k in ("main", "primary", "first")):
        return monitors[0]

    sorted_by_x = sorted(monitors, key=lambda m: m["left"])
    if any(k in ref for k in ("left", "leftmost")):
        return sorted_by_x[0]
    if any(k in ref for k in ("right", "rightmost")):
        return sorted_by_x[-1]

    sorted_by_y = sorted(monitors, key=lambda m: m["top"])
    if any(k in ref for k in ("top", "upper")):
        return sorted_by_y[0]
    if any(k in ref for k in ("bottom", "lower")):
        return sorted_by_y[-1]

    # Default: primary (first monitor)
    return monitors[0]


class ScreenshotPlugin:
    """Screen capture tools for giving the agent visual context."""

    def __init__(self, on_image: Callable[[Path], None] | None = None) -> None:
        self._monitors = _get_monitors()
        self._on_image = on_image
        self.captures: list[Path] = []   # HomeAgent attaches new ones to the chat as images

    @kernel_function(
        name="capture_screen",
        description=(
            "Capture a screenshot of one of the user's monitors. The image is then shown to you "
            "so you can see and describe what is on screen. "
            "Call this when the user asks you to look at their screen, see what is on display, "
            "or reference a monitor by name or number. "
            "Accepted references: 'screen 1', 'screen 2', 'left', 'right', 'main', 'primary'."
        ),
    )
    def capture_screen(
        self,
        monitor_reference: Annotated[
            str,
            "Which monitor to capture. Examples: 'screen 1', 'left monitor', 'main'. Defaults to primary.",
        ] = "main",
    ) -> str:
        try:
            monitor = _resolve_monitor(monitor_reference, self._monitors)
        except ValueError as exc:
            return f"Screenshot failed: {exc}"

        out_dir = Path(tempfile.gettempdir()) / "home-agent-screenshots"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"screen-{uuid.uuid4().hex}.png"

        try:
            with mss.mss() as sct:
                shot = sct.grab(monitor)
            img = Image.frombytes("RGB", shot.size, shot.rgb)
            img.thumbnail((_MAX_SIDE, _MAX_SIDE))
            img.save(out_path)
            self.captures.append(out_path)
            if self._on_image is not None:
                self._on_image(out_path)
            return str(out_path)
        except Exception as exc:
            logger.exception("Screenshot capture failed.")
            return f"Screenshot failed: {exc}"
