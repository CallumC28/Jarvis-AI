from __future__ import annotations

from pathlib import Path

import pytest

from home_agent.plugins.code_reader import CodeReaderPlugin


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)
    (root / "src" / "app.py").write_text("def main():\n    return 42\n", encoding="utf-8")
    (root / ".env").write_text("OPENAI_API_KEY=sk-secret\n", encoding="utf-8")
    (root / "node_modules" / "x").mkdir(parents=True)
    (root / "node_modules" / "x" / "index.js").write_text("def main", encoding="utf-8")
    (tmp_path / "outside.txt").write_text("private", encoding="utf-8")
    return root


def test_read_file_relative_to_root(repo: Path):
    out = CodeReaderPlugin([repo]).read_file("src/app.py")
    assert "1: def main():" in out and "2:     return 42" in out


def test_paths_outside_roots_are_refused(repo: Path):
    plugin = CodeReaderPlugin([repo])
    assert "not found inside the allowed" in plugin.read_file("../outside.txt")
    assert "not found inside the allowed" in plugin.read_file(str(repo.parent / "outside.txt"))
    assert "private" not in plugin.search_code("private", path="..")


def test_secrets_are_never_read_listed_or_searched(repo: Path):
    plugin = CodeReaderPlugin([repo])
    assert "secrets" in plugin.read_file(".env")
    assert ".env" not in plugin.list_files(str(repo))
    assert ".env:" not in plugin.search_code("sk-secret")


def test_search_skips_vendored_dirs_and_finds_code(repo: Path):
    out = CodeReaderPlugin([repo]).search_code("def main")
    assert "app.py:1: def main():" in out
    assert "node_modules" not in out


def test_list_files_with_pattern(repo: Path):
    out = CodeReaderPlugin([repo]).list_files(str(repo), pattern="*.py")
    assert "app.py" in out and "index.js" not in out
