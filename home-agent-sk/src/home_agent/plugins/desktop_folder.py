from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated

from semantic_kernel.functions import kernel_function

_INVALID_CHARS = r'<>:"/\\|?*\x00-\x1F'


def _sanitize_folder_name(name: str) -> str:
    """Windows-safe folder name sanitizer."""
    name = name.strip()
    name = re.sub(f"[{_INVALID_CHARS}]", "_", name)
    name = re.sub(r"\s+", " ", name)
    reserved = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
    if name.upper() in reserved:
        name = f"{name}_"
    if not name:
        raise ValueError("Folder name cannot be empty.")
    return name


class DesktopFolderPlugin:
    """File-system tools limited to creating folders on Desktop."""

    @kernel_function(
        name="create_desktop_folder",
        description="Create a folder on the user's Desktop. Returns the created folder path.",
    )
    def create_desktop_folder(
        self,
        folder_name: Annotated[str, "The name of the folder to create on the Desktop"],
        overwrite: Annotated[bool, "If true, do not error if the folder already exists"] = False,
    ) -> str:
        safe_name = _sanitize_folder_name(folder_name)
        desktop = Path.home() / "Desktop"
        target = desktop / safe_name

        if target.exists():
            if overwrite and target.is_dir():
                return f"Folder already exists: {target}"
            return f"Cannot create folder. Already exists: {target}"

        target.mkdir(parents=True, exist_ok=False)
        return f"Created folder: {target}"
