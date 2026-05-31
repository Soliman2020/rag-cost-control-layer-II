"""
semantic_cache/embedders.py
----------------------------
Pre-built embedder implementations for SemanticCache.
Swap these in to use API-backed semantic embeddings instead of TF-IDF.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class EmbedderBase(ABC):
    """Base class for embedder implementations."""

    @abstractmethod
    def fit(self, texts: list[str]) -> None:
        """Build vocabulary from texts (no-op for API embedders)."""
        pass

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Convert text to embedding vector."""
        pass


class OpenAIEmbedder(EmbedderBase):
    """
    OpenAI text-embedding-3-small embedder.

    Usage:
        from semantic_cache import SemanticCache
        from semantic_cache.embedders import OpenAIEmbedder

        embedder = OpenAIEmbedder(api_key="sk-...")
        cache = SemanticCache(threshold=0.92, embedder=embedder)

    Or set OPENAI_API_KEY env var and use:
        embedder = OpenAIEmbedder()  # reads from env
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "text-embedding-3-small",
        dimensions: Optional[int] = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package required. Install with: pip install openai>=1.30.0"
            )

        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.dimensions = dimensions
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> "OpenAI":
        """Lazy-load OpenAI client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def fit(self, texts: list[str]) -> None:
        """No-op for API embedders."""
        pass

    def embed(self, text: str) -> list[float]:
        """Generate embedding via OpenAI API."""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=self.dimensions,
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error("OpenAI embed failed: %s", e)
            raise


class SentenceTransformerEmbedder(EmbedderBase):
    """
    Local sentence-transformers embedder (requires sentence-transformers package).

    Usage:
        from semantic_cache import SemanticCache
        from semantic_cache.embedders import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
        cache = SemanticCache(threshold=0.92, embedder=embedder)
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers package required. "
                "Install with: pip install sentence-transformers"
            )

        self.model_name = model_name
        self.device = device
        self._model = None

    @property
    def model(self) -> "SentenceTransformer":
        """Lazy-load model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def fit(self, texts: list[str]) -> None:
        """No-op for pre-trained models."""
        pass

    def embed(self, text: str) -> list[float]:
        """Generate embedding via local model."""
        return self.model.encode(text, normalize_embeddings=True).tolist()


# Registry for convenient access
EMBEDDERS = {
    "openai": OpenAIEmbedder,
    "sentence-transformers": SentenceTransformerEmbedder,
}


def create_embedder(
    provider: str,
    **kwargs
) -> EmbedderBase:
    """
    Factory function to create embedders by name.

    Usage:
        embedder = create_embedder("openai", api_key="sk-...")
        embedder = create_embedder("sentence-transformers", model_name="all-MiniLM-L6-v2")
    """
    if provider not in EMBEDDERS:
        raise ValueError(
            f"Unknown embedder: {provider}. Available: {list(EMBEDDERS.keys())}"
        )
    return EMBEDDERS[provider](**kwargs)