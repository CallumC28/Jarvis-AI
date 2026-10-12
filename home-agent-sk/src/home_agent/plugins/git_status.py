from __future__ import annotations

import json
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Annotated, Any

from semantic_kernel.functions import kernel_function

from home_agent.plugins.browser import BrowserPlugin
from home_agent.plugins.pc import _best_match

# The gitstat dashboard (ReactJS-UI/gitstat-ui) — its API is the single source of git status.
URL = "http://127.0.0.1:8765"
_START_TIMEOUT = 30.0   # next start is usually up in ~2 s; slack for a cold disk
_MAX_FILES = 30         # changed files listed per repo before "and N more"

_FILE_STATUS = {
    "M": "modified", "A": "added", "D": "deleted", "R": "renamed", "C": "copied",
    "T": "type changed", "U": "conflicted", "?": "untracked",
}

Repo = dict[str, Any]


def _fetch_repos() -> list[Repo]:
    with urllib.request.urlopen(f"{URL}/api/repos", timeout=60) as resp:
        repos: list[Repo] = json.load(resp)
    return repos


def _start_server(app_dir: Path) -> None:
    """Start the dashboard hidden. It outlives Jarvis, so an open dashboard tab keeps working."""
    node = shutil.which("node") or "node"
    # ponytail: runs next directly (not `npm start`) so no console window flashes; keep the
    # host/port in step with gitstat-ui's package.json "start" script.
    subprocess.Popen(
        [node, "node_modules/next/dist/bin/next", "start", "-H", "127.0.0.1", "-p", "8765"],
        cwd=app_dir,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP,
    )


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _problems(repo: Repo) -> list[str]:
    if "error" in repo:
        return ["git couldn't read it"]
    problems = []
    if repo["files"]:
        problems.append(_plural(len(repo["files"]), "uncommitted file"))
    if repo["ahead"]:
        problems.append(_plural(repo["ahead"], "unpushed commit"))
    if repo["behind"]:
        problems.append(f"{_plural(repo['behind'], 'commit')} behind")
    if not repo["upstream"]:
        problems.append("no remote branch")
    return problems


def _summary(repos: list[Repo]) -> str:
    if not repos:
        return "No git repos found."
    dirty = [(r, p) for r in repos if (p := _problems(r))]
    if not dirty:
        return f"All {len(repos)} repos are clean and pushed."
    lines = [f"{len(dirty)} of {len(repos)} repos need attention:"]
    lines += [f"- {r['name']}: {', '.join(p)}" for r, p in dirty]
    clean = len(repos) - len(dirty)
    if clean:
        lines.append(f"The other {clean} {'is' if clean == 1 else 'are'} clean and pushed.")
    return "\n".join(lines)


def _spoken(name: str) -> str:
    """'ReactJS-UI/Leaf-Lounge' → 'leaf lounge', the way it would be said."""
    return name.rsplit("/", 1)[-1].replace("-", " ").replace("_", " ").lower()


def _detail(repo: Repo) -> str:
    if "error" in repo:
        return f"{repo['name']}: git couldn't read it ({repo['error']})."
    tracking = f"tracking {repo['upstream']}" if repo["upstream"] else "no remote branch"
    head = f"{repo['name']} on branch {repo['branch']} ({tracking})"
    problems = _problems(repo)
    lines = [f"{head}: {', '.join(problems) if problems else 'clean and pushed'}."]
    files = repo["files"]
    if files:
        shown = [
            f"{_FILE_STATUS.get(f['status'], 'changed')} {f['path']}" for f in files[:_MAX_FILES]
        ]
        more = f", and {len(files) - _MAX_FILES} more" if len(files) > _MAX_FILES else ""
        lines.append("Changed files: " + ", ".join(shown) + more)
    return "\n".join(lines)


class GitStatusPlugin:
    """Git status of the user's repos, via the gitstat dashboard (started on demand)."""

    def __init__(self, app_dir: Path) -> None:
        self._app_dir = app_dir

    def _repos(self) -> list[Repo] | str:
        """Repo statuses, starting the dashboard first if it's down; a string means failure."""
        try:
            return _fetch_repos()
        except urllib.error.HTTPError as exc:
            return f"The git dashboard answered with an error: {exc.code} {exc.reason}."
        except urllib.error.URLError:
            pass   # not running

        if not (self._app_dir / ".next" / "BUILD_ID").is_file():
            return f"The git dashboard isn't built yet; run `npm run build` in {self._app_dir}."
        _start_server(self._app_dir)
        deadline = time.monotonic() + _START_TIMEOUT
        while time.monotonic() < deadline:
            time.sleep(0.5)
            try:
                return _fetch_repos()
            except urllib.error.URLError:
                continue
        return f"Started the git dashboard but it didn't answer within {_START_TIMEOUT:.0f} s."

    @kernel_function(
        name="repo_status",
        description=(
            "Git status of the user's code repositories: uncommitted files, unpushed commits, "
            "commits behind, repos with no remote. Leave repo empty for a summary of all repos; "
            "name one repo to list its changed files."
        ),
    )
    def repo_status(
        self,
        repo: Annotated[str, "One repo to detail, as spoken, e.g. 'leaf lounge'; empty for all."]
        = "",
    ) -> str:
        repos = self._repos()
        if isinstance(repos, str):
            return repos
        if not repo.strip():
            return _summary(repos)
        i = _best_match(_spoken(repo), [_spoken(r["name"]) for r in repos])
        if i is None:
            return f"No repo matching '{repo}'. Repos: " + ", ".join(r["name"] for r in repos)
        return _detail(repos[i])

    @kernel_function(
        name="open_dashboard",
        description=(
            "Open the git status dashboard (all repos, changed files, commit and push) in Chrome."
        ),
    )
    def open_dashboard(self) -> str:
        repos = self._repos()   # makes sure the server is up before the page opens
        if isinstance(repos, str):
            return repos
        return str(BrowserPlugin().open_website(URL))
