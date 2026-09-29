"""Tests for the provider backends, retry logic, and cost accounting."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from rider_sim.agent.llm import (
    AnthropicBackend,
    LLMClient,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    OpenAIBackend,
    RetryableProviderError,
    ToolCall,
    ToolSpec,
    estimate_tokens,
    price_per_1m,
    resolve_backend,
)


class FakeAnthropicClient:
    def __init__(self, content_blocks: list[object]) -> None:
        self._blocks = content_blocks
        self.create_calls: list[dict[str, object]] = []

    @property
    def messages(self) -> FakeAnthropicClient:
        return self

    async def create(self, **kwargs: object) -> object:
        self.create_calls.append(kwargs)
        return FakeAnthropicMessage(self._blocks)


class FakeAnthropicMessage:
    def __init__(self, blocks: list[object]) -> None:
        self.content = blocks


class FakeTextBlock:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class FakeToolUseBlock:
    type = "tool_use"

    def __init__(self, id: str, name: str, input: dict[str, object]) -> None:
        self.id = id
        self.name = name
        self.input = input


class FakeOpenAIClient:
    def __init__(self, message: object) -> None:
        self._message = message
        self.create_calls: list[dict[str, object]] = []

    @property
    def chat(self) -> FakeOpenAIClient:
        return self

    @property
    def completions(self) -> FakeOpenAIClient:
        return self

    async def create(self, **kwargs: object) -> object:
        self.create_calls.append(kwargs)
        return FakeOpenAICompletion(self._message)


class FakeOpenAICompletion:
    def __init__(self, message: object) -> None:
        self.choices = [FakeOpenAIChoice(message)]


class FakeOpenAIChoice:
    def __init__(self, message: object) -> None:
        self.message = message


class FakeOpenAIMessage:
    def __init__(self, content: str, tool_calls: list[object] | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls


class FakeOpenAIToolCall:
    def __init__(self, id: str, name: str, arguments: str) -> None:
        self.id = id
        self.function = FakeOpenAIFunction(name, arguments)


class FakeOpenAIFunction:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


def _request() -> LLMRequest:
    return LLMRequest(
        model="test-model",
        system="sys",
        messages=(LLMMessage(role="user", content="hello"),),
        tools=(ToolSpec(name="t", description="d", input_schema={"type": "object"}),),
        temperature=0.0,
        seed=1,
    )


def test_anthropic_backend_text_and_tools() -> None:
    backend = AnthropicBackend(api_key="sk-test")
    backend._client = FakeAnthropicClient(  # type: ignore[assignment]
        [
            FakeTextBlock("decided"),
            FakeToolUseBlock("tc1", "t", {"a": 1}),
        ]
    )
    response = asyncio.run(backend.complete(_request()))
    assert response.content == "decided"
    assert response.tool_calls == (ToolCall(id="tc1", name="t", arguments={"a": 1}),)


def test_openai_backend_text_and_tools() -> None:
    backend = OpenAIBackend(api_key="sk-test")
    backend._client = FakeOpenAIClient(  # type: ignore[assignment]
        FakeOpenAIMessage(
            "decided",
            [FakeOpenAIToolCall("tc1", "t", json.dumps({"a": 1}))],
        )
    )
    response = asyncio.run(backend.complete(_request()))
    assert response.content == "decided"
    assert response.tool_calls == (ToolCall(id="tc1", name="t", arguments={"a": 1}),)


class FlakyBackend:
    name = "flaky"

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        if self.calls <= self.failures:
            raise RetryableProviderError(429, "rate limited")
        return LLMResponse(content="ok")


def test_client_retries_429_then_succeeds(tmp_path: Path) -> None:
    backend = FlakyBackend(failures=2)
    client = LLMClient(backend=backend, cache_db=None, backoff_base_seconds=0.001, max_retries=3)
    response = asyncio.run(client.complete(_request()))
    assert response.content == "ok"
    assert backend.calls == 3


def test_client_gives_up_after_max_retries(tmp_path: Path) -> None:
    backend = FlakyBackend(failures=99)
    client = LLMClient(backend=backend, cache_db=None, backoff_base_seconds=0.001, max_retries=2)
    with pytest.raises(RetryableProviderError):
        asyncio.run(client.complete(_request()))
    assert backend.calls == 3


def test_resolve_backend_and_prices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="auto"):
        resolve_backend("auto")
    with pytest.raises(ValueError, match="unknown provider"):
        resolve_backend("bogus")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert resolve_backend("anthropic").name == "anthropic"
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    assert resolve_backend("openai").name == "openai"
    assert resolve_backend("auto").name == "anthropic"

    monkeypatch.setenv("ANTHROPIC_INPUT_PRICE_PER_1M", "2.5")
    monkeypatch.setenv("ANTHROPIC_OUTPUT_PRICE_PER_1M", "7.5")
    assert price_per_1m("anthropic") == (2.5, 7.5)
    assert price_per_1m("offline") == (0.0, 0.0)
    assert estimate_tokens("12345678") == 2


def test_client_cost_tracking(tmp_path: Path) -> None:
    class CostBackend:
        name = "anthropic"

        async def complete(self, request: LLMRequest) -> LLMResponse:
            return LLMResponse(content="x" * 4000)  # ~1000 estimated tokens

    monkeypatch_env = pytest.MonkeyPatch()
    monkeypatch_env.setenv("ANTHROPIC_INPUT_PRICE_PER_1M", "3.0")
    monkeypatch_env.setenv("ANTHROPIC_OUTPUT_PRICE_PER_1M", "15.0")
    client = LLMClient(backend=CostBackend(), cache_db=None)
    asyncio.run(client.complete(_request()))
    monkeypatch_env.undo()
    assert client.cost_usd > 0
