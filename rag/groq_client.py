"""
rag/groq_client.py
------------------
Lightweight HTTP client for the Groq Cloud API.
Provides a drop-in replacement for OllamaClient, HFClient, and GeminiClient.

API endpoint targeted:
- POST https://api.groq.com/openai/v1/chat/completions
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Iterator, Optional

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class GroqError(RuntimeError):
    """Base for all Groq client errors."""

class GroqConnectionError(GroqError):
    """Could not reach the Groq API."""

class GroqTimeoutError(GroqError):
    """Request exceeded the configured timeout."""

class GroqResponseError(GroqError):
    """Unexpected or malformed response from the Groq API."""

class GroqRateLimitError(GroqError):
    """429 from the Groq API, retries exhausted."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class GroqConfig(BaseModel):
    """All parameters controlling the Groq API connection."""

    api_key:      str   = Field(default=os.environ.get("GROQ_API_KEY", ""))
    model:        str   = Field(default=os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b"))
    api_base:     str   = Field(default="https://api.groq.com/openai/v1")

    # Generation
    temperature:  float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens:   int   = Field(default=1024, ge=1, le=8192)
    top_p:        float = Field(default=0.9, ge=0.0, le=1.0)

    # HTTP / retry
    timeout:      float = Field(default=60.0, gt=0)
    max_retries:  int   = Field(default=3, ge=0, le=10)
    retry_delay:  float = Field(default=1.0, gt=0)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class GroqClient:
    """
    Thin wrapper around Groq's chat completion endpoint.
    Implements the same interface as OllamaClient.
    """

    def __init__(self, config: Optional[GroqConfig] = None) -> None:
        self._cfg = config or GroqConfig()
        if not self._cfg.api_key:
            logger.warning("[GROQ] No GROQ_API_KEY set. Requests will fail.")

    @property
    def model(self) -> str:
        return self._cfg.model

    @property
    def config(self) -> GroqConfig:
        return self._cfg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        *,
        temperature:    Optional[float] = None,
        max_tokens:     Optional[int]   = None,
        top_p:          Optional[float] = None,
        repeat_penalty: Optional[float] = None,  # Not supported by Groq, ignored
    ) -> dict:
        """
        Non-streaming chat. Returns parsed dict with answer, token counts, and latency.
        """
        payload = self._build_payload(messages, False, temperature, max_tokens, top_p)
        url = f"{self._cfg.api_base}/chat/completions"

        t_start = time.perf_counter()
        body = self._post_json(url, payload)
        latency = (time.perf_counter() - t_start) * 1000

        result = self._parse_chat_response(body)
        result["latency_ms"] = round(latency, 2)
        # Groq's response doesn't break out queue time vs eval time the way
        # Ollama's does, so the whole request latency is the closest
        # available proxy. This used to be hardcoded to 0.0, which silently
        # zeroed out every tokens/sec calculation downstream
        # (rag/rag_engine.py:72-73 guards on generation_time_ms > 0).
        result["generation_time_ms"] = round(latency, 2)
        return result

    def chat_stream(
        self,
        messages: list[dict],
        *,
        temperature:    Optional[float] = None,
        max_tokens:     Optional[int]   = None,
        top_p:          Optional[float] = None,
        repeat_penalty: Optional[float] = None,
    ) -> Iterator[str]:
        """
        Streaming chat. Yields text chunk strings as they arrive.
        """
        payload = self._build_payload(messages, True, temperature, max_tokens, top_p)
        url = f"{self._cfg.api_base}/chat/completions"
        timeout = httpx.Timeout(connect=10.0, read=self._cfg.timeout, write=10.0, pool=5.0)
        headers = {
            "Authorization": f"Bearer {self._cfg.api_key}",
            "Content-Type": "application/json"
        }

        # Single retry loop covers both 429 backoff and the 404-model-fallback --
        # chat_stream previously had neither, unlike _post_json, so a live SSE
        # request just failed outright on the first transient rate limit or
        # model blip instead of recovering the way a non-streaming call would.
        # Safe to loop before yielding: the status code is checked before any
        # body line is consumed, so no partial output has reached the caller yet.
        active_payload = payload
        try:
            with httpx.Client(timeout=timeout) as client:
                for attempt in range(self._cfg.max_retries + 1):
                    with client.stream("POST", url, json=active_payload, headers=headers) as response:
                        if response.status_code == 200:
                            yield from self._iter_stream_lines(response)
                            return

                        body_text = response.read().decode("utf-8")

                        if response.status_code == 429:
                            if attempt == self._cfg.max_retries:
                                raise GroqRateLimitError(
                                    f"Groq API streaming rate limit exceeded after {self._cfg.max_retries} retries."
                                )
                            wait_s = self._parse_retry_after(response) or (self._cfg.retry_delay * (2 ** attempt))
                            logger.warning(
                                f"[GROQ] Streaming 429 rate limited, retrying in {wait_s:.1f}s "
                                f"(attempt {attempt+1}/{self._cfg.max_retries})"
                            )
                            time.sleep(wait_s)
                            continue

                        if (
                            response.status_code == 404
                            and "model_not_found" in body_text
                            and active_payload["model"] != FALLBACK_MODEL
                        ):
                            logger.warning(
                                f"[GROQ] Model '{active_payload['model']}' unavailable (404), "
                                f"retrying this stream with fallback '{FALLBACK_MODEL}'"
                            )
                            active_payload = dict(active_payload, model=FALLBACK_MODEL)
                            continue

                        raise GroqResponseError(
                            f"Groq API returned status code {response.status_code}: {body_text}"
                        )
        except httpx.ConnectError as exc:
            raise GroqConnectionError(f"Cannot connect to Groq API endpoint.") from exc
        except httpx.TimeoutException as exc:
            raise GroqTimeoutError(f"Groq streaming timed out after {self._cfg.timeout}s.") from exc

    @staticmethod
    def _iter_stream_lines(response: "httpx.Response") -> Iterator[str]:
        for line in response.iter_lines():
            line = line.strip()
            if not line or line == "data: [DONE]":
                continue
            if line.startswith("data: "):
                line = line[len("data: "):]
            try:
                obj = json.loads(line)
                choices = obj.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                text = delta.get("content", "")
                if text:
                    yield text
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

    def health_check(self) -> bool:
        """Returns True if the API is configured and responds to a simple check."""
        if not self._cfg.api_key:
            return False
        try:
            messages = [{"role": "user", "content": "ping"}]
            self.chat(messages, max_tokens=5)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_payload(
        self,
        messages: list[dict],
        stream: bool,
        temperature: Optional[float],
        max_tokens: Optional[int],
        top_p: Optional[float],
    ) -> dict:
        """Converts standard chat messages to OpenAI/Groq API format."""
        payload: dict = {
            "model": self._cfg.model,
            "messages": messages,
            "stream": stream,
            "temperature": temperature if temperature is not None else self._cfg.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self._cfg.max_tokens,
            "top_p": top_p if top_p is not None else self._cfg.top_p,
        }
        return payload

    def _post_json(self, url: str, payload: dict) -> dict:
        """Helper to post JSON data with retries.

        429 gets its own retry path, honoring the Retry-After header when
        the API sends one (it usually does, with a precise fractional-second
        wait like "20.93s") rather than guessing with exponential backoff --
        free-tier TPM limits are tight enough on the larger models that a
        wrong guess either retries too early (another 429) or wastes time
        waiting longer than necessary.
        """
        timeout = httpx.Timeout(connect=10.0, read=self._cfg.timeout, write=10.0, pool=5.0)
        headers = {
            "Authorization": f"Bearer {self._cfg.api_key}",
            "Content-Type": "application/json"
        }
        last_exc: Optional[Exception] = None
        last_status: Optional[int] = None

        for attempt in range(self._cfg.max_retries + 1):
            try:
                with httpx.Client(timeout=timeout) as client:
                    response = client.post(url, json=payload, headers=headers)
                    if response.status_code == 429:
                        last_status = 429
                        if attempt == self._cfg.max_retries:
                            break
                        wait_s = self._parse_retry_after(response) or (self._cfg.retry_delay * (2 ** attempt))
                        logger.warning(f"[GROQ] 429 rate limited, retrying in {wait_s:.1f}s (attempt {attempt+1}/{self._cfg.max_retries})")
                        time.sleep(wait_s)
                        continue
                    if response.status_code == 404 and payload["model"] != FALLBACK_MODEL and "model_not_found" in response.text:
                        logger.warning(
                            f"[GROQ] Model '{payload['model']}' unavailable (404), "
                            f"retrying this request with fallback '{FALLBACK_MODEL}'"
                        )
                        fallback_payload = dict(payload, model=FALLBACK_MODEL)
                        fb_response = client.post(url, json=fallback_payload, headers=headers)
                        if fb_response.status_code != 200:
                            raise GroqResponseError(
                                f"Groq API returned status code {fb_response.status_code} "
                                f"(fallback model also failed): {fb_response.text}"
                            )
                        return fb_response.json()
                    if response.status_code != 200:
                        raise GroqResponseError(
                            f"Groq API returned status code {response.status_code}: {response.text}"
                        )
                    return response.json()
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                last_exc = exc
                if attempt == self._cfg.max_retries:
                    break
                time.sleep(self._cfg.retry_delay * (2 ** attempt))

        if last_status == 429:
            raise GroqRateLimitError(f"Groq API rate limit exceeded after {self._cfg.max_retries} retries.")
        if isinstance(last_exc, httpx.ConnectError):
            raise GroqConnectionError("Cannot connect to Groq API endpoint.") from last_exc
        else:
            raise GroqTimeoutError("Groq API request timed out.") from last_exc

    @staticmethod
    def _parse_retry_after(response: "httpx.Response") -> Optional[float]:
        """Reads the Retry-After header (seconds, possibly fractional) if present."""
        header = response.headers.get("retry-after")
        if not header:
            return None
        try:
            return max(0.0, float(header))
        except ValueError:
            return None

    def _parse_chat_response(self, body: dict) -> dict:
        """Parses the non-streaming Groq API response."""
        try:
            choices = body.get("choices", [])
            if not choices:
                raise GroqResponseError("Groq API response has no choices.")
            
            message = choices[0].get("message", {})
            answer = message.get("content", "")
            
            finish_reason = choices[0].get("finish_reason", "stop")
            
            usage = body.get("usage", {})
            p_tokens = usage.get("prompt_tokens", 0)
            c_tokens = usage.get("completion_tokens", 0)
            
        except (KeyError, IndexError, TypeError) as exc:
            raise GroqResponseError(f"Malformed Groq API response: {exc}. Body: {body}") from exc

        return {
            "answer": answer,
            "prompt_tokens": p_tokens,
            "completion_tokens": c_tokens,
            "finish_reason": finish_reason,
            # generation_time_ms is set by chat() right after this returns.
            # Read the echoed "model" field rather than self._cfg.model -- a
            # 404 fallback in _post_json may have served this from a
            # different model than the one configured.
            "model_name": body.get("model", self._cfg.model),
        }


# ---------------------------------------------------------------------------
# Singleton Getter
# ---------------------------------------------------------------------------

_groq_client_instance: Optional[GroqClient] = None

def get_groq_client(config: Optional[GroqConfig] = None) -> GroqClient:
    global _groq_client_instance
    if _groq_client_instance is None:
        _groq_client_instance = GroqClient(config)
    return _groq_client_instance


# ---------------------------------------------------------------------------
# Startup model resolution
# ---------------------------------------------------------------------------

FALLBACK_MODEL = "openai/gpt-oss-20b"


def resolve_active_model() -> dict:
    """
    Probes GROQ_MODEL with a real (tiny) completion before any user-facing
    request depends on it, and falls back to FALLBACK_MODEL if the
    configured model can't actually serve a request -- decommissioned,
    typo'd in the env file, or not available on this account's tier. Groq
    retires free-tier model availability with some regularity, and finding
    that out from a live user's chat request is a worse failure mode than
    a slightly slower startup.

    Mutates os.environ["GROQ_MODEL"] on fallback -- this must run before
    the first GroqConfig() is constructed (GroqConfig reads the env var as
    a field default), which in practice means before get_rag_engine() or
    get_reranker() is ever called. backend/app.py's startup handler calls
    this first for exactly that reason.

    Returns {"model": str, "ok": bool, "fell_back": bool, "error": str|None}.
    Never raises -- a probe failure is a startup log line, not a crash;
    the app should still start and let individual requests fail with a
    real error if Groq is down for the fallback model too.
    """
    requested = os.environ.get("GROQ_MODEL", FALLBACK_MODEL)

    def _probe(model: str) -> Optional[str]:
        cfg = GroqConfig(model=model, timeout=15.0, max_retries=0)
        client = GroqClient(cfg)
        try:
            client.chat([{"role": "user", "content": "Reply with: OK"}], max_tokens=8)
            return None
        except Exception as exc:
            return str(exc)

    error = _probe(requested)
    if error is None:
        return {"model": requested, "ok": True, "fell_back": False, "error": None}

    logger.warning(f"[GROQ] Model '{requested}' failed startup probe: {error[:200]}")
    if requested == FALLBACK_MODEL:
        # Already on the fallback and it still failed -- nothing left to
        # fall back to. Leave GROQ_MODEL as-is; real requests will surface
        # the same error.
        return {"model": requested, "ok": False, "fell_back": False, "error": error}

    fallback_error = _probe(FALLBACK_MODEL)
    os.environ["GROQ_MODEL"] = FALLBACK_MODEL
    if fallback_error is None:
        logger.warning(f"[GROQ] Falling back to '{FALLBACK_MODEL}' for this process.")
        return {"model": FALLBACK_MODEL, "ok": True, "fell_back": True, "error": error}

    logger.error(f"[GROQ] Fallback model '{FALLBACK_MODEL}' also failed: {fallback_error[:200]}")
    return {"model": FALLBACK_MODEL, "ok": False, "fell_back": True, "error": fallback_error}
