"""
LLM client manager for Starlight AI-CRM Mailer.
Handles all generation and embedding tasks behind one interface (`azure_manager`).

Two providers, picked by LLM_PROVIDER (defaults to "gemini" when GEMINI_API_KEYS
is set, otherwise "azure"):

Gemini (Google AI Studio, via its OpenAI-compatible endpoint):
    GEMINI_API_KEYS            - comma/newline separated keys, highest priority first
    GEMINI_QUALITY_MODELS      - fallback chain for client-facing copy
    GEMINI_FAST_MODELS         - fallback chain for extraction / retrieval helpers
    GEMINI_QUALITY_REASONING   - reasoning_effort for quality calls (default none)
    GEMINI_FAST_REASONING      - reasoning_effort for fast calls (default minimal)
    GEMINI_EMBEDDING_MODEL     - default gemini-embedding-001
    EMBEDDING_DIMENSION        - vector size requested from the embedding model (1536)

Azure OpenAI:
    AZURE_OPENAI_ENDPOINT             - https://your-resource.openai.azure.com/
    AZURE_OPENAI_API_KEY              - your Azure OpenAI key
    AZURE_OPENAI_API_VERSION          - e.g. 2024-12-01-preview
    AZURE_OPENAI_DEPLOYMENT_NAME      - chat deployment (alias: AZURE_OPENAI_CHAT_DEPLOYMENT)
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT - embedding deployment

Callers pass tier="fast" for work that never reaches a client (website analysis,
HyDE, captions); everything else runs on the quality chain.
"""
import os
import re
import base64
import time
import logging
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AzureOpenAI,
    BadRequestError,
    OpenAI,
    RateLimitError,
)

_here = Path(__file__).parent
load_dotenv(dotenv_path=_here / ".env")

log = logging.getLogger("azure_client")

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_QUALITY_MODELS = "gemini-3.8-flash,gemini-3.6-flash,gemini-3.5-flash-lite"
DEFAULT_FAST_MODELS = "gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-3.6-flash"
# gemini-embedding-001 accepts ~2048 tokens per input; longer chunks are cut to fit.
GEMINI_EMBED_MAX_CHARS = 7000

# Cooldowns after an error, in seconds.
KEY_DEAD_COOLDOWN = 15 * 60      # 401/402/403: bad key, project denied, credits depleted
KEY_MODEL_RATE_COOLDOWN = 60     # 429 on one key for one model
KEY_MODEL_MISSING_COOLDOWN = 60 * 60  # 404: model not offered to this key's project
MODEL_OVERLOAD_COOLDOWN = 30     # 5xx / timeout: model-wide demand spike

REASONING_LEVELS = ("none", "minimal", "low")
# Gemini counts thinking tokens against max_tokens, so callers' budgets (sized
# for visible output) get this much extra whenever the model is allowed to think.
THINKING_HEADROOM = {"none": 0, "minimal": 512, "low": 2048, "": 4096}


def _split_list(raw: str) -> List[str]:
    return [x.strip() for x in re.split(r"[,\n]", raw or "") if x.strip()]


def _image_message(image_bytes: bytes, text_prompt: str, system_prompt: str, image_mime: str) -> List[Dict[str, Any]]:
    data_url = f"data:{image_mime};base64,{base64.b64encode(image_bytes).decode('utf-8')}"
    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
            {"type": "text", "text": text_prompt},
        ],
    })
    return messages


def _message_to_dict(choice_message: Any) -> Dict[str, Any]:
    tool_calls = []
    for i, tc in enumerate(choice_message.tool_calls or []):
        # model_dump keeps provider extras (Gemini thought signatures) that must be echoed back.
        call = tc.model_dump(exclude_none=True)
        call["id"] = call.get("id") or f"call_{i}"
        call["type"] = "function"
        call.setdefault("function", {})
        call["function"]["arguments"] = call["function"].get("arguments") or "{}"
        tool_calls.append(call)
    return {
        "role": "assistant",
        "content": choice_message.content or "",
        "tool_calls": tool_calls,
    }


# ---------------------------------------------------------------------------
# Azure OpenAI
# ---------------------------------------------------------------------------

def _check_azure_env() -> None:
    missing = [v for v in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY") if not os.getenv(v, "").strip()]
    if missing:
        raise EnvironmentError(
            "Missing required environment variables: " + ", ".join(missing)
            + ". Set them in .env, or set GEMINI_API_KEYS to use Gemini instead."
        )
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    if not endpoint.startswith("https://"):
        raise EnvironmentError(f"AZURE_OPENAI_ENDPOINT must start with 'https://' (got '{endpoint}')")


class AzureOpenAIManager:
    """Azure OpenAI chat, vision and embeddings with backoff on rate limits."""

    provider = "azure"

    def __init__(self) -> None:
        _check_azure_env()
        self.endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/") + "/"
        self.api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
        self.api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        self.chat_deployment: str = (
            os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
            or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
            or "gpt-4o"
        )
        self.embedding_deployment: str = os.getenv(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"
        )
        log.info(
            "Azure OpenAI configured: endpoint=%s  chat=%s  embedding=%s",
            self.endpoint, self.chat_deployment, self.embedding_deployment,
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
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        json_mode: bool = False,
        max_retries: int = 4,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        tier: str = "quality",
    ) -> str:
        msg = self.chat_completion_message(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            max_retries=max_retries,
            tools=tools,
            tool_choice=tool_choice,
            tier=tier,
        )
        return msg.get("content") or ""

    def chat_completion_message(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        json_mode: bool = False,
        max_retries: int = 4,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        tier: str = "quality",
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.chat_deployment,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode and not tools:
            kwargs["response_format"] = {"type": "json_object"}
        if tools:
            kwargs["tools"] = tools
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice

        wait = 2
        for attempt in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(**kwargs)
                return _message_to_dict(response.choices[0].message)
            except RateLimitError:
                if attempt == max_retries:
                    log.error("Rate limit persists after %d retries.", max_retries)
                    raise
                log.warning("Rate limit hit (attempt %d/%d). Retrying in %ds...", attempt + 1, max_retries, wait)
                time.sleep(wait)
                wait = min(wait * 2, 60)
            except BadRequestError as e:
                err_msg = str(e)
                if "content_filter" in err_msg or "ResponsibleAIPolicyViolation" in err_msg:
                    log.error("Azure OpenAI content filter triggered: %s", err_msg)
                    raise RuntimeError(f"Content filter triggered: {err_msg}") from e
                log.error("Azure OpenAI Bad Request: %s", e)
                raise
            except APIError as e:
                log.error("Azure OpenAI API error: %s", e)
                raise
        return {"role": "assistant", "content": "", "tool_calls": []}

    def embed_text(self, text: str) -> List[float]:
        response = self.client.embeddings.create(input=text, model=self.embedding_deployment)
        return response.data[0].embedding

    def embed_documents(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            response = self.client.embeddings.create(input=texts[i: i + batch_size], model=self.embedding_deployment)
            all_embeddings.extend(d.embedding for d in sorted(response.data, key=lambda x: x.index))
        return all_embeddings

    def vision_completion(
        self,
        image_bytes: bytes,
        text_prompt: str,
        system_prompt: str = "",
        image_mime: str = "image/png",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        json_mode: bool = False,
        max_retries: int = 4,
        tier: str = "quality",
    ) -> str:
        return self.chat_completion(
            _image_message(image_bytes, text_prompt, system_prompt, image_mime),
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            max_retries=max_retries,
            tier=tier,
        )


# ---------------------------------------------------------------------------
# Gemini (OpenAI-compatible endpoint) with key rotation and model fallback
# ---------------------------------------------------------------------------

class LLMUnavailableError(RuntimeError):
    """Every key and model in the chain failed or is cooling down."""


class GeminiManager:
    """
    Gemini chat, vision and embeddings.

    Keys are tried in priority order and models in fallback order. A failing key
    or model is parked for a cooldown, so one exhausted project or one overloaded
    model does not slow every later request.
    """

    provider = "gemini"

    def __init__(self) -> None:
        self.keys = _split_list(os.getenv("GEMINI_API_KEYS", ""))
        if not self.keys:
            raise EnvironmentError("GEMINI_API_KEYS is empty — add at least one Google AI Studio key.")
        self.models = {
            "quality": _split_list(os.getenv("GEMINI_QUALITY_MODELS", DEFAULT_QUALITY_MODELS)),
            "fast": _split_list(os.getenv("GEMINI_FAST_MODELS", DEFAULT_FAST_MODELS)),
        }
        self.reasoning = {
            "quality": os.getenv("GEMINI_QUALITY_REASONING", "none").strip().lower(),
            "fast": os.getenv("GEMINI_FAST_REASONING", "minimal").strip().lower(),
        }
        self.embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001").strip()
        self.embedding_dim = int(os.getenv("EMBEDDING_DIMENSION", "1536") or 1536)
        self.timeout = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "45") or 45)
        # Give up on a call after this long, waiting out cooldowns in between.
        self.deadline = float(os.getenv("GEMINI_CALL_DEADLINE_SECONDS", "150") or 150)

        self._clients: Dict[int, OpenAI] = {}
        self._lock = threading.Lock()
        self._key_until: Dict[int, float] = {}
        self._key_model_until: Dict[tuple, float] = {}
        self._model_until: Dict[str, float] = {}
        # reasoning_effort each model accepted last; "" means send none at all.
        self._effort_ok: Dict[str, str] = {}
        log.info(
            "Gemini configured: %d keys  quality=%s  fast=%s  embedding=%s@%d",
            len(self.keys), self.models["quality"], self.models["fast"],
            self.embedding_model, self.embedding_dim,
        )

    # -- bookkeeping -------------------------------------------------------

    def _client(self, key_idx: int) -> OpenAI:
        with self._lock:
            client = self._clients.get(key_idx)
            if client is None:
                client = OpenAI(
                    api_key=self.keys[key_idx],
                    base_url=GEMINI_OPENAI_BASE_URL,
                    timeout=self.timeout,
                    max_retries=0,
                )
                self._clients[key_idx] = client
            return client

    def _park(self, table: Dict, slot: Any, seconds: float) -> None:
        with self._lock:
            table[slot] = max(table.get(slot, 0.0), time.monotonic() + seconds)

    def _ready(self, key_idx: int, model: str, now: float) -> bool:
        with self._lock:
            return (
                self._key_until.get(key_idx, 0.0) <= now
                and self._key_model_until.get((key_idx, model), 0.0) <= now
            )

    def _model_ready(self, model: str, now: float) -> bool:
        with self._lock:
            return self._model_until.get(model, 0.0) <= now

    def _next_wake(self, models: List[str]) -> float:
        with self._lock:
            times = [t for t in self._key_until.values()]
            times += list(self._key_model_until.values())
            times += [self._model_until.get(m, 0.0) for m in models]
        future = [t for t in times if t > time.monotonic()]
        return min(future) if future else time.monotonic()

    def status(self) -> Dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            return {
                "provider": "gemini",
                "keys": len(self.keys),
                "parked_keys": [f"key#{i + 1}" for i, t in self._key_until.items() if t > now],
                "parked_models": [m for m, t in self._model_until.items() if t > now],
                "models": self.models,
                "embedding_model": self.embedding_model,
            }

    # -- core rotation loop ------------------------------------------------

    def _rotate(self, models: List[str], call: Callable[[OpenAI, str], Any], *, model_wide_overload: bool) -> Any:
        """
        Run `call(client, model)` across models (outer) and keys (inner).

        model_wide_overload: treat 5xx/timeouts as a model-wide spike and move to
        the next model. Embeddings pass False — their single model cannot change
        without invalidating every stored vector, so they rotate keys instead.
        """
        start = time.monotonic()
        last_error: Optional[BaseException] = None
        while True:
            attempted = False
            for model in models:
                now = time.monotonic()
                if model_wide_overload and not self._model_ready(model, now):
                    continue
                for key_idx in range(len(self.keys)):
                    if not self._ready(key_idx, model, time.monotonic()):
                        continue
                    attempted = True
                    tag = f"key#{key_idx + 1}/{model}"
                    try:
                        return call(self._client(key_idx), model)
                    except RateLimitError as e:
                        last_error = e
                        log.warning("Gemini %s rate limited; parking %ss", tag, KEY_MODEL_RATE_COOLDOWN)
                        self._park(self._key_model_until, (key_idx, model), KEY_MODEL_RATE_COOLDOWN)
                    except (APITimeoutError, APIConnectionError) as e:
                        last_error = e
                        log.warning("Gemini %s connection/timeout: %s", tag, type(e).__name__)
                        if model_wide_overload:
                            self._park(self._model_until, model, MODEL_OVERLOAD_COOLDOWN)
                            break
                    except APIStatusError as e:
                        last_error = e
                        code = e.status_code
                        if code in (401, 402, 403):
                            log.warning("Gemini key#%d unusable (HTTP %s); parking %ss", key_idx + 1, code, KEY_DEAD_COOLDOWN)
                            self._park(self._key_until, key_idx, KEY_DEAD_COOLDOWN)
                        elif code == 404:
                            log.warning("Gemini %s not available (404); parking", tag)
                            self._park(self._key_model_until, (key_idx, model), KEY_MODEL_MISSING_COOLDOWN)
                        elif code >= 500:
                            log.warning("Gemini %s overloaded (HTTP %s)", tag, code)
                            if model_wide_overload:
                                self._park(self._model_until, model, MODEL_OVERLOAD_COOLDOWN)
                                break
                        else:
                            raise
            remaining = self.deadline - (time.monotonic() - start)
            if remaining <= 0:
                break
            if not attempted:
                time.sleep(min(max(self._next_wake(models) - time.monotonic(), 0.5), 10.0, remaining))
            else:
                time.sleep(min(2.0, remaining))
        raise LLMUnavailableError(
            f"Gemini unavailable — every key/model failed or is cooling down "
            f"(last error: {type(last_error).__name__ if last_error else 'none'}: {last_error})"
        )

    def _create_chat(self, client: OpenAI, model: str, kwargs: Dict[str, Any], tier: str) -> Any:
        """chat.completions.create, adapting reasoning_effort to what this model accepts."""
        preferred = self._effort_ok.get(model, self.reasoning.get(tier, "none"))
        efforts = [preferred] + [e for e in REASONING_LEVELS if e != preferred] + [""]
        last_bad: Optional[BadRequestError] = None
        for effort in efforts:
            body = dict(kwargs)
            if effort:
                body["reasoning_effort"] = effort
            body["max_tokens"] = kwargs["max_tokens"] + THINKING_HEADROOM.get(effort, 0)
            try:
                response = client.chat.completions.create(model=model, **body)
                self._effort_ok[model] = effort
                return response
            except BadRequestError as e:
                # Gemini models differ on supported thinking levels and often
                # reject an unsupported one with a generic INVALID_ARGUMENT.
                last_bad = e
                continue
        assert last_bad is not None
        raise last_bad

    # -- public interface (matches AzureOpenAIManager) ---------------------

    def get_client(self) -> OpenAI:
        return self._client(0)

    def get_chat_deployment(self) -> str:
        return self.models["quality"][0]

    def get_embedding_deployment(self) -> str:
        return self.embedding_model

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        json_mode: bool = False,
        max_retries: int = 4,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        tier: str = "quality",
    ) -> str:
        msg = self.chat_completion_message(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            max_retries=max_retries,
            tools=tools,
            tool_choice=tool_choice,
            tier=tier,
        )
        return msg.get("content") or ""

    def chat_completion_message(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        json_mode: bool = False,
        max_retries: int = 4,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        tier: str = "quality",
    ) -> Dict[str, Any]:
        tier = tier if tier in self.models else "quality"
        kwargs: Dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode and not tools:
            kwargs["response_format"] = {"type": "json_object"}
        if tools:
            kwargs["tools"] = tools
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice

        def call(client: OpenAI, model: str) -> Dict[str, Any]:
            response = self._create_chat(client, model, kwargs, tier)
            choice = response.choices[0]
            visible = getattr(response.usage, "completion_tokens", None) or 0
            if choice.finish_reason == "length" and not choice.message.tool_calls and visible < max_tokens // 2:
                # Thinking ate the budget despite the headroom; retry once with room to spare.
                log.info("Gemini %s ran out of budget while thinking; retrying with more room", model)
                response = self._create_chat(client, model, {**kwargs, "max_tokens": max_tokens + 8192}, tier)
                choice = response.choices[0]
            if choice.finish_reason == "content_filter" and not (choice.message.content or choice.message.tool_calls):
                raise RuntimeError(f"Content filter triggered by {model}")
            return _message_to_dict(choice.message)

        return self._rotate(self.models[tier], call, model_wide_overload=True)

    def _embed(self, texts: List[str]) -> List[List[float]]:
        inputs = [(t or " ")[:GEMINI_EMBED_MAX_CHARS] for t in texts]

        def call(client: OpenAI, model: str) -> List[List[float]]:
            response = client.embeddings.create(model=model, input=inputs, dimensions=self.embedding_dim)
            # Gemini sometimes leaves index unset; fall back to response order.
            data = sorted(enumerate(response.data), key=lambda p: p[1].index if p[1].index is not None else p[0])
            vectors = [d.embedding for _, d in data]
            if len(vectors) != len(inputs):
                raise RuntimeError(f"Gemini returned {len(vectors)} embeddings for {len(inputs)} inputs")
            return vectors

        return self._rotate([self.embedding_model], call, model_wide_overload=False)

    def embed_text(self, text: str) -> List[float]:
        return self._embed([text])[0]

    def embed_documents(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            out.extend(self._embed(texts[i: i + batch_size]))
        return out

    def vision_completion(
        self,
        image_bytes: bytes,
        text_prompt: str,
        system_prompt: str = "",
        image_mime: str = "image/png",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        json_mode: bool = False,
        max_retries: int = 4,
        tier: str = "quality",
    ) -> str:
        return self.chat_completion(
            _image_message(image_bytes, text_prompt, system_prompt, image_mime),
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            max_retries=max_retries,
            tier=tier,
        )


def active_provider() -> str:
    explicit = os.getenv("LLM_PROVIDER", "").strip().lower()
    if explicit in ("gemini", "azure"):
        return explicit
    return "gemini" if _split_list(os.getenv("GEMINI_API_KEYS", "")) else "azure"


# ---------------------------------------------------------------------------
# Module-level singleton — import and use directly:
#   from azure_client import azure_manager
# Lazy so the API process can boot for health checks before secrets are present.
# ---------------------------------------------------------------------------
class _LazyLLMManager:
    def __init__(self) -> None:
        self._inner: Optional[Any] = None
        self._lock = threading.Lock()

    def _get(self) -> Any:
        if self._inner is None:
            with self._lock:
                if self._inner is None:
                    self._inner = GeminiManager() if active_provider() == "gemini" else AzureOpenAIManager()
        return self._inner

    def __getattr__(self, name: str):
        return getattr(self._get(), name)


azure_manager = _LazyLLMManager()
