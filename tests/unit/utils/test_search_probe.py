"""Unit tests for search-provider HTTP probes."""

from __future__ import annotations

import httpx
import pytest

from octop.infra.utils import search_probe


def _patch_client(monkeypatch: pytest.MonkeyPatch, handler: httpx.MockTransport) -> None:
    real = httpx.AsyncClient

    def factory(*args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs = dict(kwargs)
        kwargs["transport"] = handler
        return real(*args, **kwargs)

    monkeypatch.setattr(search_probe.httpx, "AsyncClient", factory)


@pytest.mark.asyncio
async def test_unknown_provider() -> None:
    result = await search_probe.probe_search_provider("nope", {})
    assert result["success"] is False
    assert result["error_type"] == "invalid_config"
    assert result["provider_id"] == "nope"


@pytest.mark.asyncio
async def test_tavily_missing_key() -> None:
    result = await search_probe.probe_search_provider("tavily", {})
    assert result["success"] is False
    assert result["error_type"] == "invalid_config"
    assert "TAVILY_API_KEY" in (result.get("error") or "")


@pytest.mark.asyncio
async def test_google_missing_cse() -> None:
    result = await search_probe.probe_search_provider("google", {"GOOGLE_API_KEY": "gk"})
    assert result["success"] is False
    assert "GOOGLE_CSE_ID" in (result.get("error") or "")


@pytest.mark.asyncio
async def test_tavily_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "api.tavily.com"
        assert request.headers.get("Authorization") == "Bearer tvly-test"
        return httpx.Response(200, json={"results": [{"url": "https://example.com"}]})

    _patch_client(monkeypatch, httpx.MockTransport(handler))
    result = await search_probe.probe_search_provider(
        "tavily",
        {"TAVILY_API_KEY": "tvly-test"},
    )
    assert result["success"] is True
    assert result["result_count"] == 1
    assert result["provider_id"] == "tavily"


@pytest.mark.asyncio
async def test_tavily_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="invalid api key")

    _patch_client(monkeypatch, httpx.MockTransport(handler))
    result = await search_probe.probe_search_provider(
        "tavily",
        {"TAVILY_API_KEY": "bad"},
    )
    assert result["success"] is False
    assert result["error_type"] == "auth_error"


@pytest.mark.asyncio
async def test_brave_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "api.search.brave.com"
        assert request.headers.get("X-Subscription-Token") == "brave-key"
        return httpx.Response(
            200,
            json={"web": {"results": [{"url": "https://example.com"}]}},
        )

    _patch_client(monkeypatch, httpx.MockTransport(handler))
    result = await search_probe.probe_search_provider(
        "brave",
        {"BRAVE_API_KEY": "brave-key"},
    )
    assert result["success"] is True
    assert result["result_count"] == 1


@pytest.mark.asyncio
async def test_deepseek_missing_key() -> None:
    result = await search_probe.probe_search_provider("deepseek", {})
    assert result["success"] is False
    assert result["error_type"] == "invalid_config"
    assert "DEEPSEEK_API_KEY" in (result.get("error") or "")


@pytest.mark.asyncio
async def test_deepseek_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.deepseek.com/v1/tools/search"
        assert request.headers.get("Authorization") == "Bearer sk-ds"
        return httpx.Response(200, json={"results": []})

    _patch_client(monkeypatch, httpx.MockTransport(handler))
    result = await search_probe.probe_search_provider(
        "deepseek",
        {"DEEPSEEK_API_KEY": "sk-ds"},
    )
    assert result["success"] is True
    assert result["provider_id"] == "deepseek"
    assert result["result_count"] == 0


@pytest.mark.asyncio
async def test_deepseek_accepts_api_key_alias_and_custom_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://proxy.test/v1/tools/search"
        assert request.headers.get("Authorization") == "Bearer sk-shared"
        return httpx.Response(200, json={"results": [{"url": "https://example.com"}]})

    _patch_client(monkeypatch, httpx.MockTransport(handler))
    result = await search_probe.probe_search_provider(
        "deepseek",
        {
            "DEEPSEEK_SEARCH_API_KEY": "sk-shared",
            "DEEPSEEK_SEARCH_BASE_URL": "https://proxy.test",
        },
    )
    assert result["success"] is True
    assert result["result_count"] == 1


@pytest.mark.asyncio
async def test_deepseek_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="invalid api key")

    _patch_client(monkeypatch, httpx.MockTransport(handler))
    result = await search_probe.probe_search_provider(
        "deepseek",
        {"DEEPSEEK_API_KEY": "bad"},
    )
    assert result["success"] is False
    assert result["error_type"] == "auth_error"


@pytest.mark.asyncio
async def test_rate_limited_provider_reports_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="slow down")

    _patch_client(monkeypatch, httpx.MockTransport(handler))
    result = await search_probe.probe_search_provider(
        "deepseek",
        {"DEEPSEEK_API_KEY": "sk-ds"},
    )
    assert result["success"] is False
    assert result["error_type"] == "rate_limited"


@pytest.mark.asyncio
async def test_deepseek_prefers_explicit_search_only_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("Authorization") == "Bearer sk-search"
        return httpx.Response(200, json={"results": []})

    _patch_client(monkeypatch, httpx.MockTransport(handler))
    result = await search_probe.probe_search_provider(
        "deepseek",
        {
            "DEEPSEEK_API_KEY": "sk-generic",
            "DEEPSEEK_SEARCH_API_KEY": "sk-search",
        },
    )
    assert result["success"] is True


@pytest.mark.asyncio
async def test_deepseek_probe_retries_transient_429(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "0"})
        return httpx.Response(200, json={"results": []})

    _patch_client(monkeypatch, httpx.MockTransport(handler))
    result = await search_probe.probe_search_provider(
        "deepseek",
        {"DEEPSEEK_API_KEY": "sk-ds"},
    )
    assert result["success"] is True
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_probe_reports_http_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    _patch_client(monkeypatch, httpx.MockTransport(handler))
    result = await search_probe.probe_search_provider(
        "deepseek",
        {"DEEPSEEK_API_KEY": "sk-ds"},
    )
    assert result["success"] is False
    assert result["error_type"] == "timeout"
