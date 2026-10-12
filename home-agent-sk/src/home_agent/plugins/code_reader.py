from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path
from typing import Annotated

from semantic_kernel.functions import kernel_function

_SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", "env",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".next", "dist", "build", ".idea", ".vs",
}
# Never hand secrets to the model (it may be a cloud provider).
_SECRET_PATTERNS = (".env", ".env.*", "*.pem", "*.key", "*.pfx", "*.p12", "id_rsa*", "id_ed25519*")
_MAX_FILE_BYTES = 1_000_000
_MAX_LIST = 200
_MAX_READ_LINES = 400
_MAX_MATCHES = 50


def _is_secret(path: Path) -> bool:
    return any(fnmatch.fnmatch(path.name.lower(), pat) for pat in _SECRET_PATTERNS)


class CodeReaderPlugin:
    """Read-only access to source code inside the folders listed in CODE_FOLDERS."""

    def __init__(self, roots: list[Path]) -> None:
        self._roots = [r.expanduser().resolve() for r in roots]

    def _resolve(self, path: str) -> Path:
        """Map a model-supplied path to a real path, refusing anything outside the roots."""
        raw = Path(path.strip() or ".").expanduser()
        candidates = [raw] if raw.is_absolute() else [root / raw for root in self._roots]
        for candidate in candidates:
            resolved = candidate.resolve()
            inside = any(resolved.is_relative_to(root) for root in self._roots)
            if inside and resolved.exists():
                if _is_secret(resolved):
                    raise PermissionError(f"'{resolved.name}' may contain secrets; not reading it.")
                return resolved
        raise FileNotFoundError(
            f"'{path}' was not found inside the allowed code folders: "
            + ", ".join(str(r) for r in self._roots)
        )

    def _walk(self, base: Path):
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
            for name in sorted(filenames):
                path = Path(dirpath) / name
                if not _is_secret(path):
                    yield path

    @kernel_function(
        name="list_files",
        description=(
            "List files in the user's code folders. Leave path empty to see the allowed root "
            "folders. Use a glob pattern such as '*.py' to filter."
        ),
    )
    def list_files(
        self,
        path: Annotated[str, "Folder to list, absolute or relative to a code folder."] = "",
        pattern: Annotated[str, "Filename glob, e.g. '*.py'."] = "*",
    ) -> str:
        if not path.strip():
            return "Code folders:\n" + "\n".join(str(r) for r in self._roots)
        try:
            base = self._resolve(path)
        except (FileNotFoundError, PermissionError) as exc:
            return str(exc)
        if base.is_file():
            return str(base)
        matches = [p for p in self._walk(base) if fnmatch.fnmatch(p.name, pattern)]
        lines = [str(p.relative_to(base)) for p in matches[:_MAX_LIST]]
        if len(matches) > _MAX_LIST:
            lines.append(f"... {len(matches) - _MAX_LIST} more; narrow the path or pattern.")
        return f"{base}:\n" + ("\n".join(lines) or "(no matching files)")

    @kernel_function(
        name="read_file",
        description=(
            "Read a source file from the user's code folders. Returns numbered lines. "
            f"Reads at most {_MAX_READ_LINES} lines per call; use start_line to page through."
        ),
    )
    def read_file(
        self,
        path: Annotated[str, "File path, absolute or relative to a code folder."],
        start_line: Annotated[int, "1-based line to start from."] = 1,
    ) -> str:
        try:
            file = self._resolve(path)
        except (FileNotFoundError, PermissionError) as exc:
            return str(exc)
        if not file.is_file():
            return f"'{path}' is a folder; use list_files."
        if file.stat().st_size > _MAX_FILE_BYTES:
            return f"'{file.name}' is too large to read ({file.stat().st_size:,} bytes)."
        lines = file.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(start_line, 1)
        chunk = lines[start - 1 : start - 1 + _MAX_READ_LINES]
        body = "\n".join(f"{n}: {text}" for n, text in enumerate(chunk, start))
        end = start + len(chunk) - 1
        more = ""
        if end < len(lines):
            more = f"\n... {len(lines) - end} more lines (start_line={end + 1})"
        return f"{file} (lines {start}-{end} of {len(lines)}):\n{body}{more}"

    @kernel_function(
        name="search_code",
        description=(
            "Search the user's code for text (case-insensitive; regular expressions allowed). "
            "Returns matching lines as path:line: text."
        ),
    )
    def search_code(
        self,
        query: Annotated[str, "Text or regex to search for, e.g. a function name."],
        path: Annotated[str, "Limit the search to this folder; empty searches all of them."] = "",
    ) -> str:
        try:
            regex = re.compile(query, re.IGNORECASE)
        except re.error:
            regex = re.compile(re.escape(query), re.IGNORECASE)
        try:
            bases = [self._resolve(path)] if path.strip() else self._roots
        except (FileNotFoundError, PermissionError) as exc:
            return str(exc)

        hits: list[str] = []
        for base in bases:
            files = [base] if base.is_file() else self._walk(base)
            for file in files:
                if file.stat().st_size > _MAX_FILE_BYTES:
                    continue
                try:
                    text = file.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    continue   # binary or unreadable
                for n, line in enumerate(text.splitlines(), 1):
                    if regex.search(line):
                        hits.append(f"{file}:{n}: {line.strip()[:200]}")
                        if len(hits) >= _MAX_MATCHES:
                            return "\n".join(hits) + "\n... more matches; narrow the search."
        return "\n".join(hits) or f"No matches for '{query}'."
