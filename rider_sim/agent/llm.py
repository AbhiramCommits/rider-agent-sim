"""Provider-agnostic LLM client with native tool calling.

Two backends implement the same protocol:

- ``AnthropicBackend`` (default) -- Anthropic Messages API. Note: the current
  anthropic SDK no longer exposes ``temperature`` or ``seed`` request params;
  they are still part of the cache key for reproducibility but are not sent.
- ``OpenAIBackend`` -- OpenAI Chat Completions API with ``tools``,
  ``tool_choice``, ``temperature``, ``seed`` and JSON response format.

Shared behavior of :class:`LLMClient`:

- ``asyncio`` batching with a bounded semaphore (configurable max concurrency).
- Exponential-backoff retry on 429 / 5xx responses.
- An on-disk response cache (DuckDB) keyed by the sha256 of
  ``(provider, model, prompt, tools, temperature, seed)``, so reruns are free
  and deterministic. The cache hit rate is reported at the end of every
  :meth:`LLMClient.run_batch` call.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import duckdb
from rich.console import Console

from rider_sim.config import LLM_CACHE_DB

console = Console()

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_MAX_TOKENS = 1024

# Price estimates (USD per 1M tokens) used for the cost-per-decision metric.
# Defaults are snapshot estimates from provider pricing pages; override via
# environment variables. Cached reads never hit the API and cost nothing.
_INPUT_PRICE_ENV = {
    "anthropic": ("ANTHROPIC_INPUT_PRICE_PER_1M", "ANTHROPIC_OUTPUT_PRICE_PER_1M"),
    "openai": ("OPENAI_INPUT_PRICE_PER_1M", "OPENAI_OUTPUT_PRICE_PER_1M"),
}
_DEFAULT_PRICES_PER_1M = {"anthropic": (3.0, 15.0), "openai": (0.15, 0.6)}


def estimate_tokens(text: str) -> int:
    """Coarse token estimate from character count (4 chars/token)."""
    return max(1, len(text) // 4)


def price_per_1m(provider: str) -> tuple[float, float]:
    input_env, output_env = _INPUT_PRICE_ENV.get(provider, ("", ""))
    input_price, output_price = _DEFAULT_PRICES_PER_1M.get(provider, (0.0, 0.0))
    raw_input = os.getenv(input_env) if input_env else None
    if raw_input is not None:
        with contextlib.suppress(ValueError):
            input_price = float(raw_input)
    raw_output = os.getenv(output_env) if output_env else None
    if raw_output is not None:
        with contextlib.suppress(ValueError):
            output_price = float(raw_output)
    return input_price, output_price


def _request_token_count(request: LLMRequest) -> int:
    parts = [request.system]
    for message in request.messages:
        parts.append(message.content)
        for call in message.tool_calls:
            parts.append(json.dumps(call.arguments))
    for tool in request.tools:
        parts.append(json.dumps(tool.input_schema))
    return sum(estimate_tokens(part) for part in parts)


def _response_token_count(response: LLMResponse) -> int:
    parts = [response.content]
    for call in response.tool_calls:
        parts.append(json.dumps(call.arguments))
    return sum(estimate_tokens(part) for part in parts)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class LLMMessage:
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None


@dataclass(frozen=True)
class LLMRequest:
    model: str
    system: str
    messages: tuple[LLMMessage, ...]
    tools: tuple[ToolSpec, ...] = ()
    temperature: float = 0.0
    seed: int | None = None


@dataclass(frozen=True)
class LLMResponse:
    content: str
    tool_calls: tuple[ToolCall, ...] = ()


class RetryableProviderError(Exception):
    """Raised by backends for 429 / 5xx responses; triggers client backoff."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"provider error {status_code}: {message}")
        self.status_code = status_code


class LLMBackend(Protocol):
    name: str

    async def complete(self, request: LLMRequest) -> LLMResponse: ...


class AnthropicBackend:
    """Anthropic Messages API backend with native tool calling."""

    name = "anthropic"

    def __init__(self, api_key: str | None = None) -> None:
        import anthropic

        self._client = anthropic.AsyncAnthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def _to_native_messages(self, messages: tuple[LLMMessage, ...]) -> list[dict[str, Any]]:
        native: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "tool":
                native.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": message.tool_call_id,
                                "content": message.content,
                            }
                        ],
                    }
                )
            elif message.role == "assistant" and message.tool_calls:
                native.append(
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": call.id,
                                "name": call.name,
                                "input": call.arguments,
                            }
                            for call in message.tool_calls
                        ],
                    }
                )
            elif message.role == "assistant":
                native.append({"role": "assistant", "content": message.content})
            else:
                native.append({"role": message.role, "content": message.content})
        return native

    async def complete(self, request: LLMRequest) -> LLMResponse:
        import anthropic

        kwargs: dict[str, Any] = {
            "model": request.model,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "system": request.system,
            "messages": self._to_native_messages(request.messages),
        }
        if request.tools:
            kwargs["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                }
                for tool in request.tools
            ]
            kwargs["tool_choice"] = {"type": "auto"}
        try:
            message = await self._client.messages.create(**kwargs)
        except anthropic.APIStatusError as exc:
            if exc.status_code == 429 or exc.status_code >= 500:
                raise RetryableProviderError(int(exc.status_code), str(exc)) from exc
            raise
        tool_calls: list[ToolCall] = []
        content_parts: list[str] = []
        for block in message.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                content_parts.append(getattr(block, "text", ""))
            elif block_type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=str(getattr(block, "id", "")),
                        name=str(getattr(block, "name", "")),
                        arguments=dict(getattr(block, "input", {})),
                    )
                )
        return LLMResponse(content="\n".join(content_parts), tool_calls=tuple(tool_calls))


class OpenAIBackend:
    """OpenAI Chat Completions backend with native function calling."""

    name = "openai"

    def __init__(self, api_key: str | None = None) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def _to_native_messages(self, messages: tuple[LLMMessage, ...]) -> list[dict[str, Any]]:
        native: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "tool":
                native.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.tool_call_id,
                        "content": message.content,
                    }
                )
            elif message.role == "assistant" and message.tool_calls:
                native.append(
                    {
                        "role": "assistant",
                        "content": message.content or None,
                        "tool_calls": [
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": json.dumps(call.arguments),
                                },
                            }
                            for call in message.tool_calls
                        ],
                    }
                )
            else:
                native.append({"role": message.role, "content": message.content})
        return native

    async def complete(self, request: LLMRequest) -> LLMResponse:
        from openai import APIStatusError

        kwargs: dict[str, Any] = {
            "model": request.model,
            "temperature": request.temperature,
            "max_completion_tokens": DEFAULT_MAX_TOKENS,
            "messages": self._to_native_messages(request.messages),
        }
        if request.seed is not None:
            kwargs["seed"] = request.seed
        if request.tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    },
                }
                for tool in request.tools
            ]
            kwargs["tool_choice"] = "auto"
        try:
            completion = await self._client.chat.completions.create(**kwargs)
        except APIStatusError as exc:
            if exc.status_code == 429 or exc.status_code >= 500:
                raise RetryableProviderError(int(exc.status_code), str(exc)) from exc
            raise
        message = completion.choices[0].message
        tool_calls: list[ToolCall] = []
        for call in message.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            tool_calls.append(
                ToolCall(id=str(call.id), name=call.function.name, arguments=arguments)
            )
        return LLMResponse(content=message.content or "", tool_calls=tuple(tool_calls))


@dataclass
class LLMClient:
    """Batched, cached, retrying wrapper over a provider backend."""

    backend: LLMBackend
    cache_db: Path | None = LLM_CACHE_DB
    max_concurrency: int = 8
    max_retries: int = 3
    backoff_base_seconds: float = 1.0
    _semaphore: asyncio.BoundedSemaphore | None = field(default=None, init=False)
    _hits: int = field(default=0, init=False)
    _requests: int = field(default=0, init=False)
    _cost_usd: float = field(default=0.0, init=False)
    _con: duckdb.DuckDBPyConnection | None = field(default=None, init=False)

    @property
    def cost_usd(self) -> float:
        return self._cost_usd

    def _get_semaphore(self) -> asyncio.BoundedSemaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.BoundedSemaphore(self.max_concurrency)
        return self._semaphore

    def _connection(self) -> duckdb.DuckDBPyConnection:
        assert self.cache_db is not None
        if self._con is None:
            self.cache_db.parent.mkdir(parents=True, exist_ok=True)
            self._con = duckdb.connect(str(self.cache_db))
            self._con.execute(
                "CREATE TABLE IF NOT EXISTS llm_cache ("
                "key VARCHAR PRIMARY KEY, response_json VARCHAR, "
                "created_at TIMESTAMP DEFAULT now())"
            )
        return self._con

    def cache_key(self, request: LLMRequest) -> str:
        payload: dict[str, Any] = {
            "provider": self.backend.name,
            "model": request.model,
            "system": request.system,
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "tool_calls": [
                        {"id": c.id, "name": c.name, "arguments": c.arguments} for c in m.tool_calls
                    ],
                    "tool_call_id": m.tool_call_id,
                }
                for m in request.messages
            ],
            "tools": [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in request.tools
            ],
            "temperature": request.temperature,
            "seed": request.seed,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def _cache_get(self, key: str) -> LLMResponse | None:
        if self.cache_db is None:
            return None
        row = (
            self._connection()
            .execute("SELECT response_json FROM llm_cache WHERE key = ?", [key])
            .fetchone()
        )
        if row is None:
            return None
        payload = json.loads(row[0])
        return LLMResponse(
            content=str(payload["content"]),
            tool_calls=tuple(
                ToolCall(id=c["id"], name=c["name"], arguments=c["arguments"])
                for c in payload["tool_calls"]
            ),
        )

    def _cache_put(self, key: str, response: LLMResponse) -> None:
        if self.cache_db is None:
            return
        payload = json.dumps(
            {
                "content": response.content,
                "tool_calls": [
                    {"id": c.id, "name": c.name, "arguments": c.arguments}
                    for c in response.tool_calls
                ],
            },
            sort_keys=True,
        )
        self._connection().execute(
            "INSERT OR REPLACE INTO llm_cache (key, response_json) VALUES (?, ?)",
            [key, payload],
        )

    def _record_cost(self, request: LLMRequest, response: LLMResponse) -> None:
        input_price, output_price = price_per_1m(self.backend.name)
        input_tokens = _request_token_count(request)
        output_tokens = _response_token_count(response)
        self._cost_usd += (input_tokens * input_price + output_tokens * output_price) / 1_000_000.0

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """One request with caching, concurrency limit, and retry backoff."""
        key = self.cache_key(request)
        self._requests += 1
        cached = self._cache_get(key)
        if cached is not None:
            self._hits += 1
            return cached
        sem = self._get_semaphore()
        response: LLMResponse | None = None
        async with sem:
            for attempt in range(self.max_retries + 1):
                try:
                    response = await self.backend.complete(request)
                    break
                except RetryableProviderError:
                    if attempt == self.max_retries:
                        raise
                    await asyncio.sleep(self.backoff_base_seconds * (2**attempt))
        assert response is not None
        self._record_cost(request, response)
        self._cache_put(key, response)
        return response

    async def run_batch(self, requests: list[LLMRequest]) -> list[LLMResponse]:
        """Run requests concurrently; always report cache hit rate afterwards."""
        responses = await asyncio.gather(*(self.complete(r) for r in requests))
        self.report_cache_stats()
        return list(responses)

    def cache_stats(self) -> tuple[int, int]:
        return self._hits, self._requests

    def report_cache_stats(self) -> None:
        hits, total = self.cache_stats()
        rate = (hits / total) if total else 0.0
        console.log(f"[bold]llm cache[/bold]: {hits}/{total} hits ({rate:.1%})")


def resolve_backend(provider: str, api_key: str | None = None) -> LLMBackend:
    """Build the backend for ``provider`` (anthropic | openai | auto)."""
    if provider == "anthropic":
        return AnthropicBackend(api_key=api_key)
    if provider == "openai":
        return OpenAIBackend(api_key=api_key)
    if provider == "auto":
        if os.getenv("ANTHROPIC_API_KEY"):
            return AnthropicBackend(api_key=api_key)
        if os.getenv("OPENAI_API_KEY"):
            return OpenAIBackend(api_key=api_key)
        raise ValueError("provider='auto' requires ANTHROPIC_API_KEY or OPENAI_API_KEY")
    raise ValueError(f"unknown provider {provider!r}; expected anthropic | openai | auto")
