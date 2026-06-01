# rag-cost-control-layer

A production-ready Python library for cost-controlled RAG (Retrieval Augmented Generation) pipelines. Provides semantic caching, intelligent query routing, token budget enforcement, circuit breaking, and a complete RAG pipeline — reducing LLM costs by up to 85% without degrading answer quality.

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Dependencies](https://img.shields.io/badge/dependencies-optional-green)](requirements.txt)

---

## The Problem

RAG systems are optimized for relevance, not cost. This creates three silent failures in production:

| Issue | Impact |
|-------|--------|
| **Context over-fetching** | Retrieving top-10 chunks when 2–3 answer the query. The rest is noise you pay for. |
| **No caching** | Two users ask the same question ten minutes apart. You pay the full LLM cost twice. |
| **No model routing** | Every query hits your most expensive model, even simple factoids that don't need it. |

At 10,000 requests/day, naive RAG costs **$120/day**. With this cost control layer: **$17/day**. That's **$3,090 saved every month** without changing the model or degrading answer quality.

---

## What It Does

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           RAG COST CONTROL LAYER                             │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Incoming Query                                                              │
│        │                                                                     │
│        ▼                                                                    │
│  ┌───────────────────┐                                                       │
│  │  Semantic Cache   │ ◄── Check for cached responses                        │
│  │  (TF-IDF / OpenAI)│     Returns in ~4ms at $0 cost                        │
│  └─────────┬─────────┘                                                       │
│            │ HIT                                                             │
│            ▼                                                                 │
│         $0 RESPONSE                                                          │
│            │ MISS                                                            │
│            ▼                                                                 │
│  ┌───────────────────┐                                                       │
│  │  Query Router     │ ◄── Classify query complexity                         │
│  │  3-signal scoring│     Route to cheapest model tier                       │
│  └─────────┬─────────┘                                                       │
│            │                                                                 │
│            ▼                                                                 │
│  ┌───────────────────┐   SIMPLE ──► gpt-4o-mini ($0.000165/1K)               │
│  │  Model Selection │   STANDARD ► gpt-4o       ($0.005/1K)                  │
│  │  Tier Routing    │   COMPLEX ──► gpt-4       ($0.03/1K)                   │
│  └─────────┬─────────┘                                                        │
│            │                                                                  │
│            ▼                                                                 │
│  ┌───────────────────┐                                                       │
│  │  Token Budget    │ ◄── Allocate tokens by priority                        │
│  │  + CostLedger   │     Track hourly/daily spend                            │
│  │  + CircuitBreaker│ ◄── Prevent runaway costs                              │
│  └─────────┬─────────┘                                                       │
│            │                                                                 │
│            ▼                                                                 │
│       LLM CALL                                                               │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Core Components

| Component | Purpose | Key Feature |
|-----------|---------|-------------|
| **SemanticCache** | Cache query responses | TF-IDF or OpenAI embeddings, ~4ms lookup |
| **QueryRouter** | Route to cheapest model | 3-signal complexity scoring, routes 81% to cheap tier |
| **TokenBudget** | Allocate tokens | Slot-based priority allocation |
| **CostLedger** | Track spending | Sliding window hourly/daily limits |
| **CircuitBreaker** | Prevent overspend | CLOSED → OPEN → HALF_OPEN state machine |
| **CostAwareClient** | OpenAI wrapper | Drop-in replacement with full cost control |
| **RAGPipeline** | Complete RAG | Document ingestion → retrieval → generation |

---

## Installation

### Basic (No Dependencies)
```bash
git clone https://github.com/Soliman2020/rag-cost-control-layer-II.git
cd rag-cost-control-layer/rag-cost-control-layer
python demo/demo.py  # Works immediately with standard library only
```

### With OpenAI Support
```bash
pip install -r requirements.txt
```

**Requirements:**
- `openai>=1.30.0` — For OpenAI API calls and embeddings
- `tiktoken>=0.7.0` — For accurate token counting
- `httpx>=0.27.0` — HTTP client
- `python-dotenv>=1.0.0` — For .env file support

**Python 3.10+ required.**

**Setup Environment:**
```bash
# Copy the template environment file and fill in your values
cp .copy_env .env
# Edit .env with your actual API key
```

**.copy_env template:**
```env
OPENAI_API_KEY=your_api_key_here
```

---

## Quick Start

### Option 1: Complete RAG Pipeline (Recommended)

The easiest way to get started — includes document storage, retrieval, and cost-controlled generation:

```python
from rag_pipeline import RAGPipeline

# Initialize pipeline with your API credentials
pipeline = RAGPipeline(
    api_key="sk-your-key-here",
    base_url="https://api.openai.com/v1",  # or proxy URL
    k=3,                    # Retrieve top 3 documents
    chunk_size=500,         # Split into 500-char chunks
)

# Add documents to your knowledge base
pipeline.add_documents([
    {
        "content": "RAG stands for Retrieval Augmented Generation. "
                    "It enhances LLM outputs by retrieving relevant "
                    "context from a knowledge base.",
        "source": "rag_intro"
    },
    {
        "content": "Vector databases store embeddings - numerical "
                    "representations of text that enable semantic search.",
        "source": "vector_db"
    },
])

# Query your RAG system
response = pipeline.query("What is RAG?")

print(f"Answer: {response.content}")
print(f"Sources: {response.sources}")
print(f"Cost: ${response.cost_usd:.6f}")
print(f"Cached: {response.cached}")
```

**What happens under the hood:**

1. `add_documents()` → Splits text into chunks → Creates embeddings → Stores in vector DB
2. `query()` → Embeds question → Similarity search → Retrieves top-k chunks
3. Builds context from retrieved chunks
4. Calls LLM with cost control (cache check → model routing → budget enforcement)
5. Returns answer with sources, cost, and cache status

---

### Option 2: CostAwareClient (OpenAI Wrapper)

Use this if you already have a RAG system and just want to add cost control:

```python
from openai_client import CostAwareClient

# Create cost-aware OpenAI client
client = CostAwareClient(
    api_key="sk-your-key-here",
    base_url="https://api.openai.com/v1",
    cache_threshold=0.92,    # 0.92+ for OpenAI embeddings
    simple_threshold=0.25,    # Queries below this → cheap model
    complex_threshold=0.65,   # Queries above this → expensive model
    hourly_limit_usd=10.0,    # Circuit breaker triggers at $10/hour
    daily_limit_usd=100.0,
)

# Use exactly like standard OpenAI client
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is RAG?"},
    ],
)

print(response.content)
print(client.get_stats())  # Shows cache hits, costs, savings
```

---

### Option 3: Individual Components

For fine-grained control, use components individually:

```python
from semantic_cache.cache import SemanticCache
from query_router.router import QueryRouter, ModelTier
from token_budget.budget import BudgetEnforcer

# 1. Semantic Cache — $0 cost on cache hit
cache = SemanticCache(
    threshold=0.85,      # Cosine similarity threshold
    max_size=1000,        # Max cached entries
    ttl_seconds=3600,     # 1-hour TTL per entry
)

# 2. Query Router — Route to cheapest model
router = QueryRouter(
    simple_threshold=0.25,   # Score < 0.25 → SIMPLE tier
    complex_threshold=0.65,  # Score > 0.65 → COMPLEX tier
)

# 3. Budget Enforcer — Track costs and prevent overspend
enforcer = BudgetEnforcer(
    hourly_limit_usd=5.0,
    daily_limit_usd=50.0,
    per_request_limit_usd=0.10,
    downgrade_on_breach=True,  # Auto-downgrade on cost breach
)

query = "What is RAG?"

# Step 1: Check cache
cached = cache.get(query)
if cached:
    print(f"Cache hit: {cached}")
else:
    # Step 2: Route to appropriate model tier
    decision = router.route(query)
    print(f"Routed to: {decision.model_id} (tier: {decision.tier.value})")

    # Step 3: Enforce budget and call LLM
    with enforcer.request(model_tier=decision.tier.value, estimated_tokens=500) as ctx:
        if ctx.allowed:
            # Your LLM call here
            response = "RAG stands for Retrieval Augmented Generation..."
            ctx.record_actual(actual_tokens=180, cost_usd=0.0009)
        else:
            response = ctx.fallback_response

    # Step 4: Cache the response
    cache.set(query, response)
    print(f"Response: {response}")
```

---

## Code Examples Explained

### 1. Semantic Cache — How It Works

```python
from semantic_cache.cache import SemanticCache

cache = SemanticCache(threshold=0.85)

# on cache HIT: Returns immediately, $0 cost
result = cache.get("What is RAG?")  # str or None

# on cache MISS: Store response for future requests
cache.set("What is RAG?", "RAG stands for...")
```

**How similarity works:**
- Converts query to vector (TF-IDF or OpenAI embeddings)
- Compares with all cached queries using cosine similarity
- Returns cached response if similarity ≥ threshold
- Example: "What is RAG?" has 0.92 similarity with "Explain RAG to me" → cache hit

**Why threshold matters:**
| Threshold | Use Case | Risk |
|-----------|----------|------|
| 0.75 | Demo only | False positives (similar words) |
| 0.85 | TF-IDF production | Balanced |
| 0.92+ | OpenAI embeddings | Minimal false hits |

---

### 2. Query Router — How It Works

```python
from query_router.router import QueryRouter, ModelTier

router = QueryRouter(simple_threshold=0.25, complex_threshold=0.65)
decision = router.route("What is RAG?")
```

**3-Signal Complexity Scoring:**

| Signal | Weight | What It Detects |
|--------|--------|-----------------|
| **Length** | 20% | Normalized token count (saturates at 80) |
| **Entities** | 30% | Capitalized/numeric/technical terms |
| **Reasoning** | 50% | Keywords: compare, analyze, trade-off, why, how... |

**Example scores:**
```
"What is RAG?"           → 0.10 (SIMPLE)  "What is X" = factoid
"How does RAG work?"     → 0.35 (STANDARD) Short but has "how"
"Compare RAG vs fine-tuning" → 0.75 (COMPLEX) "compare" keyword + entities
```

**Model tier mapping:**
```python
ModelTier.SIMPLE   → gpt-4o-mini   ($0.000165/1K tokens)
ModelTier.STANDARD → gpt-4o       ($0.005/1K tokens)
ModelTier.COMPLEX  → gpt-4        ($0.03/1K tokens)
```

---

### 3. Budget Enforcer — How It Works

```python
from token_budget.budget import BudgetEnforcer

enforcer = BudgetEnforcer(
    hourly_limit_usd=10.0,
    daily_limit_usd=100.0,
    downgrade_on_breach=True,
)

with enforcer.request(model_tier="standard", estimated_tokens=500) as ctx:
    if not ctx.allowed:
        return ctx.fallback_response  # Circuit breaker open
    
    # Reserve tokens in priority order
    ctx.budget.reserve("system", 200)
    ctx.budget.reserve_text("context", retrieved_docs)
    ctx.budget.reserve("output", min(512, ctx.budget.remaining()))
    
    # Call LLM...
    ctx.record_actual(actual_tokens=620, cost_usd=0.0031)
```

**Circuit Breaker States:**

```
CLOSED    → Normal operation, requests allowed
   │
   ▼ (hourly/daily limit breached)
OPEN      → Requests blocked or downgraded to SIMPLE tier
   │
   ▼ (cooldown seconds elapsed)
HALF_OPEN → One probe request allowed
   │
   ▼ (probe succeeds/fails)
CLOSED or OPEN
```

---

## Running the Demos

### Option 1: Jupyter Notebook (All-in-One)
```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .copy_env .env

# Launch Jupyter
jupyter notebook demo/all_three_demos.ipynb
```

The notebook includes all three demos in one file:
- Demo 1: Core Components (no dependencies)
- Demo 2: CostAwareClient (requires OpenAI)
- Demo 3: Complete RAG Pipeline (requires OpenAI)

### Option 2: Run Individual Demos

#### Original Demo (No Dependencies)
```bash
python demo/demo.py
```

#### OpenAI Client Demo
```bash
# Set your API key
export OPENAI_API_KEY="sk-..."

python demo/openai_demo.py
```

#### RAG Pipeline Demo
```bash
# Set your API key
export OPENAI_API_KEY="sk-..."

python demo/rag_demo.py
```

---

## Configuration Guide

### Production Setup

| Parameter | Recommended | Description |
|-----------|--------------|-------------|
| `cache_threshold` | 0.92+ | OpenAI embeddings. Higher = fewer false hits |
| `simple_threshold` | 0.20-0.25 | Route more to cheap model |
| `complex_threshold` | 0.65-0.70 | Protect analytical queries |
| `hourly_limit_usd` | `peak_hourly_requests × avg_cost × 2.5` | Start conservative |
| `downgrade_on_breach` | `True` | Graceful degradation |

### Setting Limits

```python
# Example: 1000 requests/hour peak, $0.01 avg cost
# Hourly limit = 1000 × 0.01 × 2.5 = $25
enforcer = BudgetEnforcer(hourly_limit_usd=25.0)
```

### OpenAI API Proxy

Many educational platforms use proxy gateways. Set `base_url`:

```python
from rag_pipeline import RAGPipeline

pipeline = RAGPipeline(
    api_key="sk-...",
    base_url="https://openai.vocareum.com/v1",  # Vocareum proxy
)
```

---

## Performance

Measured on Python 3.12, Windows 11, CPU-only.

| Operation | Latency | Notes |
|-----------|---------|-------|
| Cache lookup (hit) | ~4 ms | Embed + cosine similarity |
| Cache lookup (miss) | ~4-5 ms | Same as hit, no response returned |
| Query routing | <0.025 ms | Keyword scoring |
| Budget check | <0.1 ms | In-memory allocation |
| LLM API call | 200-800 ms | Network dependent |

**Total overhead on cache miss: ~4ms**
**Total overhead on cache hit: ~4ms (no LLM call)**

---

## Project Structure

```
rag-cost-control-layer/
├── semantic_cache/
│   ├── cache.py              # SemanticCache (TF-IDF + custom embedder support)
│   └── embedders.py         # OpenAI, SentenceTransformer embedders
├── query_router/
│   └── router.py             # QueryRouter (complexity-based routing)
├── token_budget/
│   └── budget.py            # TokenBudget, CostLedger, CircuitBreaker
├── openai_client/
│   └── client.py            # CostAwareClient (OpenAI wrapper)
├── rag_pipeline/
│   └── pipeline.py          # Complete RAG pipeline
├── token_counter/
│   └── counter.py           # Accurate token counting (tiktoken)
├── demo/
│   ├── all_three_demos.ipynb # Jupyter notebook (all demos in one)
│   ├── demo.py               # Original demo (no deps)
│   ├── openai_demo.py       # OpenAI client demo
│   └── rag_demo.py           # RAG pipeline demo
├── benchmarks/
│   └── run_benchmarks.py    # Performance benchmarks
└── requirements.txt          # Dependencies
```

---

## Important Notes

### 1. Cache Persistence
Cache is in-memory only. Entries are lost on restart. For production:
- **Option A**: Add Redis backend (same `get()/set()` interface)
- **Option B**: Serialize cache to disk periodically

### 2. CostLedger Persistence
Spend history resets on restart. For multi-worker deployments:
- Use Redis to share spend data across workers
- Or accept reset on container restarts

### 3. Token Counting
Default uses `1 token ≈ 4 characters` (~15% accuracy). For production:
- Use `token_counter/counter.py` with `tiktoken` for exact counts

### 4. Routing Thresholds Are Empirical
Calibrated on RAG-domain queries. After one week of production traffic:
- If costs are high but answers good → raise `complex_threshold`
- If analytical queries get degraded → lower `complex_threshold`

### 5. Vector Store
`InMemoryVectorStore` is for development. For production with large document sets:
- Swap for Pinecone, Weaviate, Milvus, Qdrant
- Interface: `add_documents()`, `similarity_search()` — easy to swap

---

## When to Use This

### ✓ Worth It When:
- RAG system in production with measurable LLM spend
- Repeated or similar queries from users
- Mix of simple factoids + complex analytical queries
- Agentic systems where retry loops run unattended

### ✗ Skip When:
- Single-turn queries on tiny fixed dataset
- Hard latency requirements < 5ms total
- Fewer than 50 requests/day

---

## License

MIT

---

## References

- [**RAG Is Burning Money — I Built a Cost Control Layer to Fix It**](https://towardsdatascience.com/rag-is-burning-money-i-built-a-cost-control-layer-to-fix-it/)
