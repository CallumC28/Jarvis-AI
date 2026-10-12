from __future__ import annotations

import urllib.error
from pathlib import Path

import pytest

from home_agent.plugins import git_status
from home_agent.plugins.git_status import GitStatusPlugin

REPOS = [
    {"name": "Python/lan-proxy", "branch": "master", "upstream": None, "ahead": 0, "behind": 0,
     "files": [{"status": "M", "path": "run.py"}, {"status": "?", "path": "notes.txt"}]},
    {"name": "ReactJS-UI/Leaf-Lounge", "branch": "main", "upstream": "origin/main", "ahead": 2,
     "behind": 0, "files": []},
    {"name": "C#/Eleven", "branch": "main", "upstream": "origin/main", "ahead": 0, "behind": 0,
     "files": []},
]


@pytest.fixture
def app_dir(tmp_path: Path) -> Path:
    (tmp_path / ".next").mkdir()
    (tmp_path / ".next" / "BUILD_ID").write_text("x", encoding="utf-8")
    return tmp_path


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> dict:
    """A fake gitstat server: starts down if state['up'] is False, comes up when started."""
    state = {"up": True, "starts": 0, "repos": REPOS}

    def fetch() -> list[dict]:
        if not state["up"]:
            raise urllib.error.URLError(ConnectionRefusedError())
        return state["repos"]

    def start(_: Path) -> None:
        state["starts"] += 1
        state["up"] = True

    monkeypatch.setattr(git_status, "_fetch_repos", fetch)
    monkeypatch.setattr(git_status, "_start_server", start)
    monkeypatch.setattr(git_status.time, "sleep", lambda _: None)
    return state


def test_summary_names_only_repos_needing_attention(app_dir: Path, server: dict):
    out = GitStatusPlugin(app_dir).repo_status()
    assert out.startswith("2 of 3 repos need attention")
    assert "Python/lan-proxy: 2 uncommitted files, no remote branch" in out
    assert "ReactJS-UI/Leaf-Lounge: 2 unpushed commits" in out
    assert "Eleven" not in out
    assert "other 1 is clean" in out


def test_all_clean(app_dir: Path, server: dict):
    server["repos"] = [REPOS[2]]
    assert GitStatusPlugin(app_dir).repo_status() == "All 1 repos are clean and pushed."


def test_spoken_repo_name_gives_detail(app_dir: Path, server: dict):
    out = GitStatusPlugin(app_dir).repo_status("lan proxy")
    assert "Python/lan-proxy on branch master" in out
    assert "modified run.py" in out and "untracked notes.txt" in out


def test_unknown_repo_lists_what_exists(app_dir: Path, server: dict):
    out = GitStatusPlugin(app_dir).repo_status("zzzz")
    assert "No repo matching 'zzzz'" in out and "Leaf-Lounge" in out


def test_starts_server_when_down(app_dir: Path, server: dict):
    server["up"] = False
    out = GitStatusPlugin(app_dir).repo_status()
    assert server["starts"] == 1
    assert "need attention" in out


def test_unbuilt_app_is_not_started(tmp_path: Path, server: dict):
    server["up"] = False
    out = GitStatusPlugin(tmp_path).repo_status()
    assert server["starts"] == 0
    assert "npm run build" in out


def test_http_error_is_reported_not_restarted(app_dir: Path, server: dict, monkeypatch):
    def broken() -> list[dict]:
        raise urllib.error.HTTPError("u", 500, "Internal Server Error", None, None)  # type: ignore[arg-type]

    monkeypatch.setattr(git_status, "_fetch_repos", broken)
    out = GitStatusPlugin(app_dir).repo_status()
    assert server["starts"] == 0
    assert "500" in out
