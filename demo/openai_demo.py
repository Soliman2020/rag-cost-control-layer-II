"""
demo/openai_demo.py
-------------------
Production demo showing CostAwareClient with OpenAI integration.

Usage:
    # Set your API key
    export OPENAI_API_KEY="sk-..."

    # Run the demo
    python demo/openai_demo.py
"""

import os
import sys

import dotenv

dotenv.load_dotenv()

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    from openai_client import CostAwareClient

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        print("Run: export OPENAI_API_KEY='sk-...'")
        print("Or:  set OPENAI_API_KEY=sk-...  (Windows)")
        sys.exit(1)

    print("=" * 60)
    print("RAG Cost Control Layer — OpenAI Integration Demo")
    print("=" * 60)
    print()

    # Initialize the cost-aware client
    client = CostAwareClient(
        api_key=api_key,
        base_url="https://openai.vocareum.com/v1",
        cache_threshold=0.92,  # Higher threshold for OpenAI embeddings
        simple_threshold=0.25,
        complex_threshold=0.65,
        hourly_limit_usd=10.0,
        daily_limit_usd=100.0,
        per_request_limit_usd=0.25,
        downgrade_on_breach=True,
    )

    # Test queries
    queries = [
        "What is RAG?",
        "What is a vector database?",
        "How does hybrid retrieval work?",
        "Compare the trade-offs of agentic RAG vs standard RAG",
        "What is RAG?",  # Repeated - should hit cache
    ]

    print(f"Running {len(queries)} queries...\n")

    for i, query in enumerate(queries, 1):
        print(f"[Query {i:02d}] {query}")
        print("-" * 40)

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query},
            ],
            max_tokens=200,
        )

        if response.cached:
            print(f"  Source:  CACHE HIT")
            print(f"  Saved:   Full LLM call avoided")
        else:
            print(f"  Source:  LLM CALL")
            print(f"  Model:   {response.model}")
            print(f"  Cost:    ${response.cost_usd:.6f}")
            if response.usage:
                print(
                    f"  Tokens:  in={response.usage.prompt_tokens} out={response.usage.completion_tokens}"
                )

        print(f"  Response: {response.content[:80]}...")
        print()

    # Print summary
    print("=" * 60)
    print("Run Summary")
    print("=" * 60)

    stats = client.get_stats()

    print(f"  Cache hit rate:    {stats['client']['cache_hit_rate_pct']}%")
    print(f"  Total requests:   {stats['client']['total_requests']}")
    print(f"  LLM calls:         {stats['client']['llm_calls']}")
    print(f"  Cache hits:        {stats['client']['cache_hits']}")
    print(f"  Total cost:        ${stats['client']['total_cost_usd']:.6f}")
    print(f"  Cost saved:        ${stats['client']['cost_saved_usd']:.6f}")
    print(f"  Total tokens:      {stats['client']['total_tokens']}")
    print()
    print(f"  Router distribution:")
    router_stats = stats["router"]
    print(f"    Simple:    {router_stats['simple_pct']}%")
    print(f"    Standard:  {router_stats['standard_pct']}%")
    print(f"    Complex:   {router_stats['complex_pct']}%")
    print()
    print(f"  Circuit breaker:   {stats['enforcer']['circuit_breaker']['state']}")
    print(
        f"  Hourly spend:     ${stats['enforcer']['ledger']['hourly_spend_usd']:.4f} / ${stats['enforcer']['ledger']['hourly_limit_usd']}"
    )
    print(
        f"  Daily spend:      ${stats['enforcer']['ledger']['daily_spend_usd']:.4f} / ${stats['enforcer']['ledger']['daily_limit_usd']}"
    )
    print()


if __name__ == "__main__":
    main()
