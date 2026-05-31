"""
rag_pipeline/pipeline.py
-----------------------
Complete RAG pipeline with cost control.

Components:
- DocumentLoader: Load documents from files/URLs
- TextSplitter: Split documents into chunks
- VectorStore: Store and retrieve embeddings (in-memory for now)
- RAGPipeline: End-to-end RAG with cost control

Usage:
    from rag_pipeline import RAGPipeline

    pipeline = RAGPipeline(
        api_key="sk-...",
        base_url="https://openai.vocareum.com/v1",
    )

    # Add documents
    pipeline.add_documents([
        {"content": "RAG stands for Retrieval Augmented Generation...", "source": "doc1"},
        {"content": "Vector databases store embeddings...", "source": "doc2"},
    ])

    # Query
    response = pipeline.query("What is RAG?")
    print(response.content)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


# Default system prompt for RAG
DEFAULT_RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
Always base your answer on the context provided. If the context doesn't contain enough information to answer the question, say so."""


@dataclass
class Document:
    """A document chunk for RAG."""
    content: str
    source: str
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievedDocument:
    """A document retrieved from the vector store."""
    content: str
    source: str
    score: float
    metadata: dict = field(default_factory=dict)


@dataclass
class RAGResponse:
    """Response from RAG pipeline."""
    content: str
    sources: list[str]
    retrieved_docs: list[RetrievedDocument]
    model: str
    cost_usd: float
    tokens_used: int
    cached: bool = False


class TextSplitter:
    """Split text into overlapping chunks."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        """Split text into chunks."""
        if not text:
            return []

        words = text.split()
        if len(words) <= self.chunk_size:
            return [text]

        chunks = []
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = " ".join(words[i : i + self.chunk_size])
            if chunk:
                chunks.append(chunk)
            if i + self.chunk_size >= len(words):
                break

        return chunks

    def split_documents(self, documents: list[dict]) -> list[Document]:
        """Split documents into chunks."""
        docs = []
        for doc in documents:
            content = doc.get("content", "")
            source = doc.get("source", "unknown")

            chunks = self.split_text(content)
            for i, chunk in enumerate(chunks):
                docs.append(Document(
                    content=chunk,
                    source=f"{source}_chunk_{i}",
                    metadata={"chunk_index": i, "total_chunks": len(chunks)},
                ))

        return docs


class InMemoryVectorStore:
    """Simple in-memory vector store using OpenAI embeddings."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url
        self._documents: list[Document] = []
        self._embeddings: list[list[float]] = []

    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to the store."""
        if not documents:
            return

        # Get embeddings for all documents
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        texts = [doc.content for doc in documents]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )

        for doc, emb in zip(documents, response.data):
            self._documents.append(doc)
            self._embeddings.append(emb.embedding)

        logger.info(f"Added {len(documents)} documents to vector store")

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        threshold: float = 0.7,
    ) -> list[RetrievedDocument]:
        """Search for similar documents."""
        if not self._documents:
            return []

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        # Get query embedding
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=query,
        )
        query_embedding = response.data[0].embedding

        # Calculate similarities
        from semantic_cache.cache import _cosine

        similarities = []
        for i, doc_embedding in enumerate(self._embeddings):
            sim = _cosine(query_embedding, doc_embedding)
            if sim >= threshold:
                similarities.append((i, sim))

        # Sort by similarity and take top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = []
        for i, score in similarities[:k]:
            doc = self._documents[i]
            results.append(RetrievedDocument(
                content=doc.content,
                source=doc.source,
                score=score,
                metadata=doc.metadata,
            ))

        return results

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
    ) -> list[tuple[RetrievedDocument, float]]:
        """Search and return all results with scores (no threshold)."""
        if not self._documents:
            return []

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=query,
        )
        query_embedding = response.data[0].embedding

        from semantic_cache.cache import _cosine

        similarities = []
        for i, doc_embedding in enumerate(self._embeddings):
            sim = _cosine(query_embedding, doc_embedding)
            doc = self._documents[i]
            similarities.append((RetrievedDocument(
                content=doc.content,
                source=doc.source,
                score=sim,
                metadata=doc.metadata,
            ), sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    def __len__(self) -> int:
        return len(self._documents)


class RAGPipeline:
    """
    Complete RAG pipeline with integrated cost control.

    Combines:
    - Vector store for document retrieval
    - Semantic cache for query caching
    - Query router for model selection
    - Budget enforcer for cost control

    Usage:
        pipeline = RAGPipeline(api_key="sk-...")
        pipeline.add_documents([{"content": "...", "source": "doc1"}])
        response = pipeline.query("What is RAG?")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        cache_threshold: float = 0.92,
        simple_threshold: float = 0.25,
        complex_threshold: float = 0.65,
        hourly_limit_usd: float = 10.0,
        daily_limit_usd: float = 100.0,
        per_request_limit_usd: float = 0.25,
        k: int = 4,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        system_prompt: str = DEFAULT_RAG_SYSTEM_PROMPT,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url
        self.k = k
        self.system_prompt = system_prompt

        # Components
        from openai_client import CostAwareClient
        from semantic_cache.cache import SemanticCache
        from semantic_cache.embedders import OpenAIEmbedder

        # OpenAI client with cost control
        self.client = CostAwareClient(
            api_key=api_key,
            base_url=base_url,
            cache_threshold=cache_threshold,
            simple_threshold=simple_threshold,
            complex_threshold=complex_threshold,
            hourly_limit_usd=hourly_limit_usd,
            daily_limit_usd=daily_limit_usd,
            per_request_limit_usd=per_request_limit_usd,
        )

        # Vector store
        self.vector_store = InMemoryVectorStore(
            api_key=api_key,
            base_url=base_url,
        )

        # Text splitter
        self.splitter = TextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def add_documents(self, documents: list[dict]) -> None:
        """Add documents to the RAG pipeline."""
        # Split into chunks
        chunked_docs = self.splitter.split_documents(documents)

        # Add to vector store
        self.vector_store.add_documents(chunked_docs)

        logger.info(f"Added {len(chunked_docs)} document chunks")

    def query(
        self,
        query: str,
        k: Optional[int] = None,
        include_sources: bool = True,
    ) -> RAGResponse:
        """
        Query the RAG pipeline.

        Parameters
        ----------
        query : str
            The question to answer.
        k : int | None
            Number of documents to retrieve. Defaults to self.k.
        include_sources : bool
            Whether to include source documents in response.

        Returns
        -------
        RAGResponse
            The answer and metadata.
        """
        k = k or self.k

        # Step 1: Retrieve relevant documents
        retrieved = self.vector_store.similarity_search_with_score(query, k=k)

        if not retrieved:
            return RAGResponse(
                content="No relevant documents found.",
                sources=[],
                retrieved_docs=[],
                model="none",
                cost_usd=0.0,
                tokens_used=0,
            )

        # Step 2: Build context from retrieved documents
        context_parts = []
        sources = []
        for doc, score in retrieved:
            context_parts.append(doc.content)
            sources.append(doc.source)

        context = "\n\n---\n\n".join(context_parts)

        # Step 3: Build messages with context
        full_query = f"""Context:
{context}

Question: {query}

Answer:"""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": full_query},
        ]

        # Step 4: Call LLM with cost control
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=500,
        )

        # Step 5: Return response
        retrieved_docs = [doc for doc, _ in retrieved]

        return RAGResponse(
            content=response.content,
            sources=sources,
            retrieved_docs=retrieved_docs,
            model=response.model,
            cost_usd=response.cost_usd,
            tokens_used=response.usage.total_tokens if response.usage else 0,
            cached=response.cached,
        )

    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        return {
            "vector_store": {
                "document_count": len(self.vector_store),
            },
            "client": self.client.get_stats(),
        }


def create_pipeline(
    api_key: Optional[str] = None,
    base_url: str = "https://api.openai.com/v1",
    **kwargs,
) -> RAGPipeline:
    """Factory function to create a RAG pipeline."""
    return RAGPipeline(api_key=api_key, base_url=base_url, **kwargs)