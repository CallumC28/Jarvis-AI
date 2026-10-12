from __future__ import annotations

from collections.abc import Callable
from typing import Annotated

import httpx
from semantic_kernel.functions import kernel_function


class TavilySearchPlugin:
    """Web search tools powered by Tavily API."""

    def __init__(
        self,
        api_key: str,
        timeout_seconds: int = 20,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._progress = progress
        self._call_count = 0
        self._last_result_text = ""

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def last_result_text(self) -> str:
        return self._last_result_text

    @kernel_function(
        name="web_search",
        description="Search the web for recent information and return concise results with links.",
    )
    async def web_search(
        self,
        query: Annotated[str, "The web search query"],
        max_results: Annotated[int, "Maximum number of search results to return (5-10, default 10)"] = 10,
    ) -> str:
        query = query.strip()
        if not query:
            return "Search query cannot be empty."

        max_results = max(5, min(max_results, 10))

        if self._progress:
            self._progress("Calling web_search...")

        payload = {
            "api_key": self._api_key,
            "query": query,
            "search_depth": "basic",  # advanced, basic, fast, ultra-fast
            "max_results": max_results,
            "include_answer": True,
            "include_raw_content": False,
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                response = await client.post("https://api.tavily.com/search", json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            details = exc.response.text[:300]
            return f"Tavily request failed ({exc.response.status_code}): {details}"
        except httpx.RequestError as exc:
            return f"Tavily request failed: {exc}"
        except ValueError:
            return "Tavily request failed: invalid JSON response."

        if self._progress:
            self._progress("Formatting results...")

        lines: list[str] = []
        answer = (data.get("answer") or "").strip()
        if answer:
            lines.append(f"Summary: {answer}")

        results = data.get("results") or []
        if not results:
            output = "\n".join(lines) if lines else "No web results found."
            self._call_count += 1
            self._last_result_text = output
            return output

        lines.append("Results:")
        for idx, item in enumerate(results, start=1):
            title = (item.get("title") or "Untitled").strip()
            url = (item.get("url") or "").strip()
            content = (item.get("content") or "").replace("\n", " ").strip()
            snippet = content[:240]
            lines.append(f"{idx}. {title}")
            if url:
                lines.append(f"   {url}")
            if snippet:
                lines.append(f"   {snippet}")

        if self._progress:
            self._progress("Done.")

        output = "\n".join(lines)
        self._call_count += 1
        self._last_result_text = output
        return output
