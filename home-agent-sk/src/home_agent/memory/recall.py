"""Every conversation turn is logged, and can be searched later by meaning rather than keywords."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np
from openai import OpenAI

logger = logging.getLogger(__name__)

EMBED_MODEL = "nomic-embed-text"   # `ollama pull nomic-embed-text`
_BATCH = 64

Embedder = Callable[[list[str]], list[list[float]]]


def ollama_embedder(base_url: str) -> Embedder:
    client = OpenAI(base_url=base_url, api_key="ollama")

    def embed(texts: list[str]) -> list[list[float]]:
        return [d.embedding for d in client.embeddings.create(model=EMBED_MODEL, input=texts).data]

    return embed


def append_turn(log_path: Path, user: str, reply: str) -> None:
    entry = {"ts": datetime.now().isoformat(timespec="minutes"), "user": user, "jarvis": reply}
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("Failed to log conversation turn: %s", exc)


def _load_turns(log_path: Path) -> list[str]:
    if not log_path.exists():
        return []
    turns = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        try:
            t = json.loads(line)
            turns.append(f"{t['ts'][:10]} — User: {t['user']}\nJarvis: {t['jarvis']}")
        except (ValueError, KeyError):
            continue
    return turns


class Recall:
    """Semantic search over the conversation log plus any extra texts (notes).

    ponytail: brute-force cosine over a JSON vector cache — fine for tens of thousands of
    turns; move to sqlite-vec if the log ever gets bigger than that.
    """

    def __init__(self, log_path: Path, cache_path: Path, embed: Embedder) -> None:
        self._log_path = log_path
        self._cache_path = cache_path
        self._embed = embed

    def search(self, query: str, extra: list[str] | None = None, k: int = 5) -> list[str]:
        docs = _load_turns(self._log_path) + (extra or [])
        if not docs:
            return []
        cache = self._load_cache()
        keys = [hashlib.sha1(d.encode()).hexdigest() for d in docs]
        missing = [(key, d) for key, d in zip(keys, docs, strict=True) if key not in cache]
        for i in range(0, len(missing), _BATCH):
            batch = missing[i : i + _BATCH]
            vectors = self._embed([f"search_document: {d}" for _, d in batch])
            cache.update({key: v for (key, _), v in zip(batch, vectors, strict=True)})
        if missing:
            self._save_cache(cache)

        matrix = np.array([cache[key] for key in keys], dtype=np.float32)
        q = np.array(self._embed([f"search_query: {query}"])[0], dtype=np.float32)
        scores = matrix @ q / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(q) + 1e-9)
        return [docs[i] for i in np.argsort(scores)[::-1][:k]]

    def _load_cache(self) -> dict[str, list[float]]:
        try:
            return json.loads(self._cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save_cache(self, cache: dict[str, list[float]]) -> None:
        tmp = self._cache_path.with_suffix(".tmp")
        try:
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(cache), encoding="utf-8")
            tmp.replace(self._cache_path)
        except OSError as exc:
            logger.warning("Failed to save memory index: %s", exc)
