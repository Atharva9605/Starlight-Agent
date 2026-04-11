"""
Azure OpenAI Client Manager for Starlight AI-CRM Mailer.
Replaces the Gemini key_manager for all generation and embedding tasks.

Required .env variables:
    AZURE_OPENAI_ENDPOINT        - e.g. https://your-resource.openai.azure.com/
    AZURE_OPENAI_API_KEY         - Azure OpenAI API key
    AZURE_OPENAI_API_VERSION     - e.g. 2024-02-01  (defaults provided)
    AZURE_OPENAI_CHAT_DEPLOYMENT - Your GPT-4o deployment name (e.g. gpt-4o)
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT - Your embedding deployment name
                                   (e.g. text-embedding-3-large)
"""
import os
import time
import logging
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from openai import AzureOpenAI, RateLimitError, APIError

load_dotenv()
log = logging.getLogger("azure_client")


class AzureOpenAIManager:
    """
    Manages Azure OpenAI client connections for chat completions and embeddings.
    Provides retry logic with exponential backoff for rate-limit handling.
    """

    def __init__(self) -> None:
        self.endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        self.api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
        self.api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        self.chat_deployment: str = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")
        self.embedding_deployment: str = os.getenv(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"
        )

        if not self.endpoint or not self.api_key:
            log.warning(
                "AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY not set. "
                "Set them in your .env file before running generation."
            )

        self._client: Optional[AzureOpenAI] = None

    @property
    def client(self) -> AzureOpenAI:
        if self._client is None:
            self._client = AzureOpenAI(
                azure_endpoint=self.endpoint,
                api_key=self.api_key,
                api_version=self.api_version,
            )
        return self._client

    def get_client(self) -> AzureOpenAI:
        return self.client

    def get_chat_deployment(self) -> str:
        return self.chat_deployment

    def get_embedding_deployment(self) -> str:
        return self.embedding_deployment

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        json_mode: bool = False,
        max_retries: int = 4,
    ) -> str:
        """
        Generate a chat completion with exponential-backoff retry on rate limits.

        Args:
            messages:    OpenAI messages list, e.g. [{"role":"system","content":"..."}]
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens:  Maximum tokens to generate.
            json_mode:   If True, force JSON output via response_format.
            max_retries: How many times to retry on RateLimitError.

        Returns:
            The generated text as a plain string.
        """
        kwargs: Dict[str, Any] = {
            "model": self.chat_deployment,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        wait = 2
        for attempt in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content or ""
            except RateLimitError as e:
                if attempt == max_retries:
                    log.error("Rate limit persists after %d retries.", max_retries)
                    raise
                log.warning(
                    "Rate limit hit (attempt %d/%d). Retrying in %ds...",
                    attempt + 1, max_retries, wait,
                )
                time.sleep(wait)
                wait = min(wait * 2, 60)
            except APIError as e:
                log.error("Azure OpenAI API error: %s", e)
                raise

        return ""  # unreachable, but satisfies type checker

    def embed_text(self, text: str) -> List[float]:
        """Embed a single string and return its vector."""
        response = self.client.embeddings.create(
            input=text,
            model=self.embedding_deployment,
        )
        return response.data[0].embedding

    def embed_documents(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        """
        Embed a list of documents.

        Azure OpenAI processes up to 16 inputs per request; this method
        batches automatically and preserves order.

        Args:
            texts:      List of text strings to embed.
            batch_size: Number of texts per API call (max 16 for Azure).

        Returns:
            List of embedding vectors in the same order as input.
        """
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.embedding_deployment,
            )
            # Sort by index to guarantee order when batching
            sorted_data = sorted(response.data, key=lambda x: x.index)
            all_embeddings.extend(d.embedding for d in sorted_data)
        return all_embeddings


# ---------------------------------------------------------------------------
# Module-level singleton — import and use directly:
#   from azure_client import azure_manager
# ---------------------------------------------------------------------------
azure_manager = AzureOpenAIManager()
