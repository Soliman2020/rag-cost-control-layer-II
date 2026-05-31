"""
token_counter/counter.py
------------------------
Accurate token counting using tiktoken.
Drop-in replacement for the char/4 heuristic in budget.py.
"""

from __future__ import annotations

from typing import Optional

# Token counting constants
TiktokenModel = str


class TokenCounter:
    """
    Accurate token counter using OpenAI's tiktoken.

    Usage:
        counter = TokenCounter("gpt-4o")
        count = counter.count("Your text here")
        count = counter.count_messages([{"role": "user", "content": "Hello"}])
    """

    # Default encoding for each model family
    DEFAULT_ENCODINGS: dict[str, str] = {
        "gpt-4o": "cl100k_base",
        "gpt-4": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "o1": "o200k_base",
        "o1-mini": "o200k_base",
        "o1-preview": "o200k_base",
    }

    def __init__(self, model: Optional[str] = None) -> None:
        self.model = model
        self._encoder: Optional[Any] = None

    @property
    def encoder(self):
        """Lazy-load tiktoken encoder."""
        if self._encoder is None:
            try:
                import tiktoken
            except ImportError:
                raise ImportError(
                    "tiktoken required. Install with: pip install tiktoken>=0.7.0"
                )

            # Determine encoding name
            encoding_name = "cl100k_base"  # default
            if self.model:
                for model_prefix, encoding in self.DEFAULT_ENCODINGS.items():
                    if self.model.startswith(model_prefix):
                        encoding_name = encoding
                        break

            self._encoder = tiktoken.get_encoding(encoding_name)

        return self._encoder

    def count(self, text: str) -> int:
        """Count tokens in a text string."""
        if not text:
            return 0
        return len(self.encoder.encode(text))

    def count_messages(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
    ) -> int:
        """
        Count tokens for chat API messages.

        Follows OpenAI's token counting formula:
        - Base: 3 tokens per message
        - Per message: count content tokens + role token
        - Per request: 3 tokens
        """
        if not messages:
            return 0

        # Estimate based on model
        # GPT-4 family uses cl100k_base (4 tokens per message overhead)
        # o1 family uses o200k_base (different formula)
        encoding = self.model or model or "gpt-4o"

        if encoding.startswith("o1"):
            # o1 models have different token counting
            # Sum all content in messages
            total = 0
            for msg in messages:
                content = msg.get("content", "")
                total += self.count(content)
            return total

        # Standard formula for GPT-4, GPT-4o, GPT-3.5
        num_messages = len(messages)
        num_tokens = 3  # Base overhead per request

        for msg in messages:
            num_tokens += 3  # Per message overhead
            for key, value in msg.items():
                if value:
                    num_tokens += self.count(str(value))
                    if key == "name":
                        num_tokens += 1  # name field adds 1 token

        return num_tokens

    def count_tokens_remaining(
        self,
        max_tokens: int,
        messages: list[dict[str, str]],
    ) -> int:
        """Calculate available tokens for completion given message context."""
        prompt_tokens = self.count_messages(messages, self.model)
        return max(0, max_tokens - prompt_tokens)

    @staticmethod
    def get_tokenizer(model: str) -> "TokenCounter":
        """Factory method to get a tokenizer for a specific model."""
        return TokenCounter(model)


# Legacy compatibility - count function
def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens in text using specified model."""
    return TokenCounter(model).count(text)