"""
demo/rag_demo.py
----------------
Demo showing the complete RAG pipeline with cost control.

Usage:
    export OPENAI_API_KEY="sk-..."
    python demo/rag_demo.py
"""

import os
import sys

import dotenv

dotenv.load_dotenv()

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    from rag_pipeline import RAGPipeline

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        print("Run: export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    print("=" * 60)
    print("RAG Pipeline with Cost Control — Demo")
    print("=" * 60)
    print()

    # Create pipeline
    pipeline = RAGPipeline(
        api_key=api_key,
        base_url="https://openai.vocareum.com/v1",
        k=3,
        chunk_size=200,
        chunk_overlap=20,
    )

    # Sample documents
    documents = [
        {
            "content": """RAG stands for Retrieval Augmented Generation. It is a technique that enhances
LLM outputs by retrieving relevant context from a knowledge base before generating responses.
RAG helps reduce hallucinations and provides up-to-date information.""",
            "source": "rag_intro",
        },
        {
            "content": """Vector databases are specialized databases that store embeddings - numerical
representations of text, images, or other data. They enable semantic search by finding
similar items based on their vector distance in high-dimensional space.""",
            "source": "vector_db",
        },
        {
            "content": """Semantic search goes beyond keyword matching to understand the meaning behind
queries. It uses embeddings to find documents that are conceptually related, even if they
don't share exact words. This makes search more robust and flexible.""",
            "source": "semantic_search",
        },
        {
            "content": """Embedding models convert text into numerical vectors. Popular options include
OpenAI's text-embedding-3-small, sentence-transformers like all-MiniLM-L6-v2, and open-source
alternatives. The choice depends on quality vs. speed requirements.""",
            "source": "embeddings",
        },
    ]

    print("Adding documents to pipeline...")
    pipeline.add_documents(documents)
    print(f"Added {len(pipeline.vector_store)} document chunks")
    print()

    # Test queries
    queries = [
        "What is RAG?",
        "How do vector databases work?",
        "What is RAG?",  # Repeat to test caching
    ]

    print("Running queries...")
    print("-" * 60)

    for i, query in enumerate(queries, 1):
        print(f"\n[Query {i}]: {query}")
        print("-" * 40)

        response = pipeline.query(query)

        if response.cached:
            print(f"  Source:  CACHE HIT")
        else:
            print(f"  Source:  LLM CALL")
            print(f"  Model:   {response.model}")
            print(f"  Cost:    ${response.cost_usd:.6f}")
            print(f"  Tokens:  {response.tokens_used}")

        print(f"  Answer: {response.content[:100]}...")
        print(f"  Sources: {response.sources}")

    # Print summary
    print()
    print("=" * 60)
    print("Run Summary")
    print("=" * 60)

    stats = pipeline.get_stats()
    client_stats = stats["client"]["client"]

    print(f"  Total requests:    {client_stats['total_requests']}")
    print(f"  Cache hits:       {client_stats['cache_hits']}")
    print(f"  Cache hit rate:   {client_stats['cache_hit_rate_pct']}%")
    print(f"  Total cost:       ${client_stats['total_cost_usd']:.6f}")
    print(f"  Cost saved:       ${client_stats['cost_saved_usd']:.6f}")
    print()


if __name__ == "__main__":
    main()