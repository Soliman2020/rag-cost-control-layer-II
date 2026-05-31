"""
openai_client/client.py
-----------------------
OpenAI client wrapper that integrates SemanticCache, QueryRouter, and BudgetEnforcer
for production-ready cost-optimized LLM calls.

Usage:
    from openai_client import CostAwareClient

    client = CostAwareClient(
        api_key="sk-...",
        cache_threshold=0.92,      # 0.92+ for OpenAI embeddings
        simple_threshold=0.25,
        complex_threshold=0.65,
        hourly_limit_usd=10.0,
        daily_limit_usd=100.0,
    )

    # Make a chat completion call
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is RAG?"},
        ],
    )

    # Check cost savings
    print(client.get_stats())
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from openai import AsyncOpenAI, OpenAI

from query_router.router import ModelTier, QueryRouter
from semantic_cache.cache import SemanticCache
from token_budget.budget import BudgetEnforcer

logger = logging.getLogger(__name__)


# Default cost mapping for OpenAI models (input + output blended)
DEFAULT_MODEL_COSTS: dict[str, float] = {
    # GPT-4o family
    "gpt-4o": 0.005,
    "gpt-4o-mini": 0.000165,
    "gpt-4o-2024-05-13": 0.005,
    "gpt-4o-mini-2024-07-18": 0.000165,
    # GPT-4 Turbo
    "gpt-4-turbo": 0.01,
    "gpt-4-turbo-2024-04-09": 0.01,
    # GPT-4
    "gpt-4": 0.03,
    "gpt-4-0613": 0.03,
    # GPT-3.5 Turbo
    "gpt-3.5-turbo": 0.002,
    "gpt-3.5-turbo-0125": 0.0005,
    # o1-preview (reasoning models)
    "o1-preview": 0.015,
    "o1-mini": 0.003,
    # Legacy
    "gpt-3.5-turbo-1106": 0.001,
}


# Model tier mapping
MODEL_TIER_MAP: dict[str, ModelTier] = {
    "gpt-4o-mini": ModelTier.SIMPLE,
    "gpt-4o-mini-2024-07-18": ModelTier.SIMPLE,
    "gpt-4o": ModelTier.STANDARD,
    "gpt-4o-2024-05-13": ModelTier.STANDARD,
    "gpt-4-turbo": ModelTier.STANDARD,
    "gpt-4-turbo-2024-04-09": ModelTier.STANDARD,
    "gpt-4": ModelTier.COMPLEX,
    "gpt-4-0613": ModelTier.COMPLEX,
    "o1-preview": ModelTier.COMPLEX,
    "gpt-3.5-turbo": ModelTier.SIMPLE,
    "gpt-3.5-turbo-0125": ModelTier.SIMPLE,
    "gpt-3.5-turbo-1106": ModelTier.SIMPLE,
    "o1-mini": ModelTier.SIMPLE,
}


@dataclass
class ClientStats:
    """Stats for the CostAwareClient."""

    total_requests: int = 0
    cache_hits: int = 0
    llm_calls: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    cost_saved_usd: float = 0.0

    @property
    def cache_hit_rate(self) -> float:
        return self.cache_hits / self.total_requests if self.total_requests else 0.0

    def summary(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "llm_calls": self.llm_calls,
            "cache_hit_rate_pct": round(self.cache_hit_rate * 100, 1),
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "cost_saved_usd": round(self.cost_saved_usd, 6),
        }


class CostAwareClient:
    """
    OpenAI client with integrated cost control.

    Combines SemanticCache, QueryRouter, and BudgetEnforcer into a single
    drop-in replacement for the standard OpenAI client.

    Parameters
    ----------
    api_key : str | None
        OpenAI API key. Defaults to OPENAI_API_KEY env var.
    base_url : str
        Base URL for OpenAI API. Defaults to "https://api.openai.com/v1".
        For proxy/gateway use, e.g., "https://openai.vocareum.com/v1".
    cache_threshold : float
        Cosine similarity threshold for cache hits.
        Use 0.92-0.95 for OpenAI embeddings.
    cache_max_size : int
        Maximum cache entries before LRU eviction.
    cache_ttl_seconds : float | None
        Cache entry TTL. None = no expiry.
    simple_threshold : float
        Query complexity threshold for SIMPLE tier routing.
    complex_threshold : float
        Query complexity threshold for COMPLEX tier routing.
    hourly_limit_usd : float
        Hourly spend limit that trips circuit breaker.
    daily_limit_usd : float
        Daily spend limit that trips circuit breaker.
    per_request_limit_usd : float
        Per-request cost limit.
    downgrade_on_breach : bool
        True = downgrade to cheap model on breach, False = block.
    model_cost_override : dict[str, float]
        Override default model costs (per 1K tokens).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        cache_threshold: float = 0.92,
        cache_max_size: int = 1000,
        cache_ttl_seconds: Optional[float] = 3600,
        simple_threshold: float = 0.25,
        complex_threshold: float = 0.65,
        hourly_limit_usd: float = 10.0,
        daily_limit_usd: float = 100.0,
        per_request_limit_usd: float = 0.25,
        downgrade_on_breach: bool = True,
        model_cost_override: Optional[dict[str, float]] = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url
        if not self.api_key:
            logger.warning("No OpenAI API key provided. Set OPENAI_API_KEY env var.")

        # Initialize components
        from semantic_cache.embedders import OpenAIEmbedder

        self.cache = SemanticCache(
            threshold=cache_threshold,
            max_size=cache_max_size,
            ttl_seconds=cache_ttl_seconds,
            embedder=OpenAIEmbedder(api_key=self.api_key, base_url=self.base_url),
        )

        self.router = QueryRouter(
            simple_threshold=simple_threshold,
            complex_threshold=complex_threshold,
        )

        self.enforcer = BudgetEnforcer(
            hourly_limit_usd=hourly_limit_usd,
            daily_limit_usd=daily_limit_usd,
            per_request_limit_usd=per_request_limit_usd,
            downgrade_on_breach=downgrade_on_breach,
        )

        # Costs and model mapping
        self.model_costs = {**DEFAULT_MODEL_COSTS, **(model_cost_override or {})}

        # Stats
        self.stats = ClientStats()
        self._lock = __import__("threading").RLock()

    @property
    def chat(self) -> _ChatCompletions:
        """Return chat completions interface."""
        return _ChatCompletions(self)

    @property
    def embeddings(self) -> _Embeddings:
        """Return embeddings interface."""
        return _Embeddings(self)

    def get_stats(self) -> dict:
        """Return combined stats from all components."""
        return {
            "client": self.stats.summary(),
            "cache": self.cache.get_stats(),
            "router": self.router.get_stats(),
            "enforcer": self.enforcer.status(),
        }

    def _get_tier_and_cost(
        self,
        model: str,
        estimated_tokens: int = 500,
    ) -> tuple[ModelTier, str, float]:
        """Determine model tier and cost from model name."""
        # Check explicit mapping first
        if model in MODEL_TIER_MAP:
            tier = MODEL_TIER_MAP[model]
        else:
            # Fall back to routing decision based on model name
            if "mini" in model.lower() or "3.5" in model:
                tier = ModelTier.SIMPLE
            elif "o1" in model.lower():
                tier = ModelTier.COMPLEX
            else:
                # Default routing via QueryRouter
                tier = self.router.route("").tier  # empty query = SIMPLE

        # Get model ID from router's model map
        model_id = self.router.model_map.get(tier, model)

        # Get cost
        cost_per_1k = self.model_costs.get(model, 0.005)
        estimated_cost = cost_per_1k * estimated_tokens / 1000

        return tier, model_id, estimated_cost


class _ChatCompletions:
    """Chat completions interface for CostAwareClient."""

    def __init__(self, client: CostAwareClient) -> None:
        self._client = client

    @property
    def completions(self) -> "_Completions":
        """Return completions interface for OpenAI-compatible API."""
        return _Completions(self._client)

    def create(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        **kwargs,
    ) -> "OpenAIResponse":
        """
        Create a chat completion with cost control.

        This methods checks cache, routes to appropriate model tier,
        enforces budget limits, and tracks costs.
        """
        # Build query from messages (user message is the query for caching)
        query = self._extract_query(messages)

        # Step 1: Check cache
        cached_response = self._client.cache.get(query)
        if cached_response:
            with self._client._lock:
                self._client.stats.cache_hits += 1
                self._client.stats.total_requests += 1
            logger.debug("Cache hit for: %s", query[:50])
            return _CachedResponse(cached_response, query)

        # Step 2: Route to model tier
        routing_decision = self._client.router.route(query)

        # Step 3: Estimate tokens and enforce budget
        estimated_tokens = self._estimate_tokens(messages, max_tokens)

        with self._client._lock:
            effective_tier = routing_decision.tier.value
            effective_model = routing_decision.model_id

        with self._client.enforcer.request(
            model_tier=effective_tier,
            estimated_tokens=estimated_tokens,
        ) as ctx:
            if not ctx.allowed:
                # Circuit breaker or budget breach
                with self._client._lock:
                    self._client.stats.total_requests += 1
                return _FallbackResponse(ctx.fallback_response)

            # Check if downgraded by circuit breaker
            if ctx.downgraded:
                effective_tier = "simple"
                effective_model = self._client.router.model_map.get(
                    ModelTier.SIMPLE, "gpt-4o-mini"
                )
                logger.info("Request downgraded to %s", effective_model)

            # Step 4: Call OpenAI
            openai_client = OpenAI(api_key=self._client.api_key, base_url=self._client.base_url)

            # Extract content from messages for context
            context = self._build_context(messages)

            try:
                response = openai_client.chat.completions.create(
                    model=effective_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    **kwargs,
                )
            except Exception as e:
                logger.error("OpenAI API call failed: %s", e)
                return _ErrorResponse(str(e))

            # Extract response content
            content = response.choices[0].message.content or ""
            usage = response.usage

            # Calculate actual cost
            input_tokens = usage.prompt_tokens if usage else estimated_tokens // 2
            output_tokens = usage.completion_tokens if usage else estimated_tokens // 2
            total_tokens = input_tokens + output_tokens

            input_cost = (
                self._client.model_costs.get(effective_model, 0.005)
                * input_tokens
                / 1000
            )
            output_cost = (
                self._client.model_costs.get(effective_model, 0.005)
                * output_tokens
                / 1000
            )
            total_cost = input_cost + output_cost

            # Record actual usage
            ctx.record_actual(actual_tokens=total_tokens, cost_usd=total_cost)

            # Cache the response
            self._client.cache.set(query, content)

            # Update stats
            with self._client._lock:
                self._client.stats.total_requests += 1
                self._client.stats.llm_calls += 1
                self._client.stats.total_tokens += total_tokens
                self._client.stats.total_cost_usd += total_cost

                # Calculate savings from routing to cheaper model
                original_cost = (
                    DEFAULT_MODEL_COSTS.get("gpt-4", 0.03) * total_tokens / 1000
                )
                self._client.stats.cost_saved_usd += max(0, original_cost - total_cost)

            return _OpenAIResponse(
                content=content,
                model=effective_model,
                usage=usage,
                cost_usd=total_cost,
                cached=False,
            )

    def _extract_query(self, messages: list[dict[str, str]]) -> str:
        """Extract the user query from messages for caching."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        # Fallback: concatenate all user messages
        return " ".join(
            m.get("content", "") for m in messages if m.get("role") == "user"
        )

    def _build_context(self, messages: list[dict[str, str]]) -> str:
        """Build context string from messages."""
        return "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages
        )

    def _estimate_tokens(
        self, messages: list[dict[str, str]], max_tokens: Optional[int]
    ) -> int:
        """Estimate total tokens for the request."""
        # Rough estimate: 4 chars per token
        context_len = sum(len(m.get("content", "")) for m in messages)
        estimated = context_len // 4
        if max_tokens:
            estimated += max_tokens
        return max(estimated, 100)


class _Completions:
    """Completions interface for OpenAI-compatible API (client.chat.completions.create)."""

    def __init__(self, client: CostAwareClient) -> None:
        self._client = client
        self._chat = _ChatCompletions(client)

    def create(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        **kwargs,
    ) -> "OpenAIResponse":
        """Delegate to _ChatCompletions.create."""
        return self._chat.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            **kwargs,
        )


class _Embeddings:
    """Embeddings interface for CostAwareClient."""

    def __init__(self, client: CostAwareClient) -> None:
        self._client = client

    def create(
        self,
        model: str = "text-embedding-3-small",
        input: Union[str, list[str]] = "",
        dimensions: Optional[int] = None,
        **kwargs,
    ) -> "OpenAIEmbeddingResponse":
        """Create embeddings (passthrough - no cost control needed)."""
        openai_client = OpenAI(api_key=self._client.api_key, base_url=self._client.base_url)

        # Normalize input to list
        texts = [input] if isinstance(input, str) else input

        response = openai_client.embeddings.create(
            model=model,
            input=texts,
            dimensions=dimensions,
            **kwargs,
        )

        return _OpenAIEmbeddingResponse(response)


# Response wrapper classes
class OpenAIResponse:
    """Base class for responses."""

    def __init__(self, content: str, cached: bool = False) -> None:
        self.content = content
        self.cached = cached


class _CachedResponse(OpenAIResponse):
    """Cached response wrapper."""

    def __init__(self, content: str, query: str) -> None:
        super().__init__(content, cached=True)
        self.query = query
        self.model = "cached"
        self.usage = None
        self.cost_usd = 0.0

    def __repr__(self) -> str:
        return f"<CachedResponse: {self.content[:50]}...>"


class _OpenAIResponse(OpenAIResponse):
    """Real OpenAI response wrapper."""

    def __init__(
        self,
        content: str,
        model: str,
        usage: Any,
        cost_usd: float,
        cached: bool = False,
    ) -> None:
        super().__init__(content, cached=cached)
        self.model = model
        self.usage = usage
        self.cost_usd = cost_usd

    def __repr__(self) -> str:
        return f"<OpenAIResponse model={self.model} cost=${self.cost_usd:.6f}>"


class _FallbackResponse(OpenAIResponse):
    """Fallback response when circuit breaker is open."""

    def __init__(self, message: str) -> None:
        super().__init__(message, cached=False)
        self.model = "fallback"
        self.usage = None
        self.cost_usd = 0.0

    def __repr__(self) -> str:
        return f"<FallbackResponse: {self.content[:50]}>"


class _ErrorResponse(OpenAIResponse):
    """Error response wrapper."""

    def __init__(self, error: str) -> None:
        super().__init__(f"Error: {error}", cached=False)
        self.model = "error"
        self.usage = None
        self.cost_usd = 0.0

    def __repr__(self) -> str:
        return f"<ErrorResponse: {self.content[:50]}>"


class OpenAIEmbeddingResponse:
    """Wrapper for embedding responses."""

    def __init__(self, response: Any) -> None:
        self.response = response

    @property
    def data(self) -> list:
        return self.response.data

    def __getitem__(self, key: int):
        return self.response.data[key]


# Convenience function for quick setup
def create_client(api_key: Optional[str] = None, **kwargs) -> CostAwareClient:
    """Create a CostAwareClient with sensible defaults."""
    return CostAwareClient(api_key=api_key, **kwargs)
