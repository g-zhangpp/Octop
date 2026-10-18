"""LangChain tools for the built-in DeepSeek Search / Fetch provider.

Mounted through ``HarnessAgentConfig.tools`` (same mechanism as cron / knowledge
/ mobile host tools). Both tools are always mounted together when an API key is
configured; each one can still be switched off per agent in the tool-settings
denylist (``tools_disabled``), which the harness filters by tool name.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from langchain_core.tools import StructuredTool

from octop.infra.websearch.client import (
    DeepSeekSearchClient,
    DeepSeekSearchError,
    format_fetch_result,
    format_search_results,
)
from octop.infra.websearch.config import DeepSeekSearchConfig

WEB_SEARCH_TOOL = "web_search"
WEB_FETCH_TOOL = "web_fetch"

DEEPSEEK_WEB_TOOLS: tuple[str, ...] = (WEB_SEARCH_TOOL, WEB_FETCH_TOOL)

WEB_SEARCH_DESCRIPTION = (
    "Search the web through DeepSeek's search engine. Use it when the answer "
    "needs current, external, or verifiable information such as news, prices, "
    "releases, documentation, or facts beyond the model's knowledge. "
    "`query` supports search operators that are passed through to the engine: "
    '`site:example.com`, `-site:example.com`, `-word`, `-"exact phrase"`, '
    '`"exact phrase"`, `lang:zh,en`, `after:2025-01-01`, `before:2026-06-01`. '
    "Returns ranked results (strongest first) with title, URL, site, publish "
    "date, rerank_score (0-3; >=1.5 is high quality), authority (0-4; >=3 is "
    "very authoritative) and a summary. Cite the URLs you use."
)

WEB_FETCH_DESCRIPTION = (
    "Fetch one web page and return its title and Markdown content. Use it to "
    "read the full page behind a search result before quoting or summarising. "
    "`url` must be an absolute http(s) URL. On failure a readable `Error: ...` "
    "string is returned so you can pick another URL."
)


@dataclass(frozen=True)
class DeepSeekWebTools:
    """Both DeepSeek web tools plus the resolved config they were built from."""

    tools: list[StructuredTool]
    config: DeepSeekSearchConfig


def build_deepseek_web_tools(env: Mapping[str, str]) -> DeepSeekWebTools | None:
    """Build the DeepSeek ``web_search`` / ``web_fetch`` tools from env vars.

    Returns ``None`` when no DeepSeek search API key is configured.
    """
    config = DeepSeekSearchConfig.from_env(env)
    if config is None:
        return None

    client = DeepSeekSearchClient(config)

    async def web_search(
        query: str,
        count: int | None = None,
    ) -> str:
        try:
            results = await client.search(query, count=count)
        except DeepSeekSearchError as exc:
            return f"Error: {exc.readable()}"
        except Exception as exc:  # noqa: BLE001 — never abort the agent turn
            return f"Error: DeepSeek web search failed: {exc}"
        return format_search_results(
            query.strip(),
            results,
            include_content=config.need_content,
        )

    async def web_fetch(url: str) -> str:
        try:
            page = await client.fetch(url)
        except DeepSeekSearchError as exc:
            return f"Error: {exc.readable()}"
        except Exception as exc:  # noqa: BLE001 — never abort the agent turn
            return f"Error: DeepSeek web fetch failed: {exc}"
        return format_fetch_result(page)

    tools = [
        StructuredTool.from_function(
            coroutine=web_search,
            name=WEB_SEARCH_TOOL,
            description=WEB_SEARCH_DESCRIPTION,
        ),
        StructuredTool.from_function(
            coroutine=web_fetch,
            name=WEB_FETCH_TOOL,
            description=WEB_FETCH_DESCRIPTION,
        ),
    ]
    return DeepSeekWebTools(tools=tools, config=config)


__all__ = [
    "DEEPSEEK_WEB_TOOLS",
    "WEB_FETCH_DESCRIPTION",
    "WEB_FETCH_TOOL",
    "WEB_SEARCH_DESCRIPTION",
    "WEB_SEARCH_TOOL",
    "DeepSeekWebTools",
    "build_deepseek_web_tools",
]
