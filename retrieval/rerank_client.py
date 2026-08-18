"""
retrieval/rerank_client.py
----------------------------
Listwise reranking via one extra Groq chat call, replacing the local
cross-encoder path. There is no torch/sentence-transformers on this box
and never will be (see retrieval/reranker.py's docstring) -- HF's
serverless Inference API was tested as an alternative and rejected: it
scored an irrelevant passage 0.94 against an unrelated query and ranked a
control pair backwards on one probe, meaning it serves bge-reranker-base
bi-encoder-style rather than as a true cross-encoder there.

The model asked to rank is whatever GROQ_MODEL resolves to for answer
generation (env var, no separate config) -- reusing the same model means
no extra credential or provider, and the accuracy bar for "which of these
5-20 passages is most relevant" is much lower than for answer generation,
so a smaller/faster model is not a hard requirement here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Optional

import httpx

if TYPE_CHECKING:
    from retrieval.hybrid_search import RawSearchResult

logger = logging.getLogger(__name__)

TIMEOUT_S = 4.0
MAX_CANDIDATES = 20
TRUNCATE_CHARS = 400
CACHE_MAX = 256
CIRCUIT_FAILURE_THRESHOLD = 3
CIRCUIT_COOLDOWN_S = 300.0

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

_SYSTEM_PROMPT = (
    "You rank passages by relevance to a query. Respond with ONLY a JSON "
    'object of the form {"ranking": [i, i, ...]} listing every passage '
    "index from most to least relevant to the query. Every index from the "
    "input must appear exactly once, with no extras and no omissions."
)


class _LRUCache:
    """Bounded cache: (query, candidate set) -> validated ranking.

    Exact-match only (same query text, same candidate chunk_ids in the same
    order) -- this is a latency/cost optimization for repeated queries, not
    a semantic cache, so no attempt is made to match near-duplicate queries.
    """

    def __init__(self, maxsize: int) -> None:
        self._maxsize = maxsize
        self._data: "OrderedDict[str, list[int]]" = OrderedDict()

    def get(self, key: str) -> Optional[list[int]]:
        if key not in self._data:
            return None
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key: str, value: list[int]) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        if len(self._data) > self._maxsize:
            self._data.popitem(last=False)


class GroqListwiseReranker:
    """
    Guardrails, all load-bearing on a 2-core/1-worker box:

      - 4s timeout: a slow rerank must not stall a request past what a
        user would wait for an answer anyway.
      - Circuit breaker: 3 consecutive failures trips it; while tripped,
        rerank() returns immediately with no network call for 5 minutes.
        A flaky endpoint must not add a guaranteed-slow failed call to
        every single query for as long as it stays down.
      - Candidates capped at 20 and content truncated to ~400 chars each:
        bounds both the token cost and the latency of the call.
      - A small LRU cache on (query, candidate set): identical repeated
        queries skip the network call entirely.
    """

    def __init__(self) -> None:
        self._cache = _LRUCache(CACHE_MAX)
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

    def rerank(
        self,
        query: str,
        candidates: "list[RawSearchResult]",
        top_k: int,
    ) -> "tuple[list[RawSearchResult], bool]":
        """
        Returns (results, reranked).

        results always has length min(top_k, len(candidates)). reranked is
        False whenever the call could not be trusted (circuit open, HTTP
        failure, malformed JSON, a ranking that isn't a clean permutation
        of the input indices) -- in every such case results is simply
        candidates in their original (fused) order, never a guess dressed
        up as a real rerank. The caller derives its mode label ("hybrid"
        vs "hybrid+rerank") directly from this boolean, so it can no
        longer report reranking that didn't happen.
        """
        if not candidates:
            return [], False

        pool = candidates[:MAX_CANDIDATES]
        tail = candidates[MAX_CANDIDATES:]

        if self._circuit_is_open():
            return candidates[:top_k], False

        cache_key = self._cache_key(query, pool)
        cached = self._cache.get(cache_key)
        if cached is not None:
            reordered = self._apply_order(pool, cached) + tail
            return reordered[:top_k], True

        try:
            order = self._call_groq(query, pool)
            if not self._is_valid_permutation(order, len(pool)):
                raise ValueError(f"ranking is not a permutation of 0..{len(pool)-1}: {order!r}")
        except Exception as e:
            logger.warning(f"[RERANK] Groq listwise rerank did not apply: {e}")
            self._record_failure()
            return candidates[:top_k], False

        self._record_success()
        self._cache.put(cache_key, order)
        reordered = self._apply_order(pool, order) + tail
        return reordered[:top_k], True

    # ------------------------------------------------------------------
    # Circuit breaker
    # ------------------------------------------------------------------

    def _circuit_is_open(self) -> bool:
        return time.monotonic() < self._circuit_open_until

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= CIRCUIT_FAILURE_THRESHOLD:
            self._circuit_open_until = time.monotonic() + CIRCUIT_COOLDOWN_S
            logger.warning(
                f"[RERANK] {self._consecutive_failures} consecutive failures -- "
                f"disabling reranking for {CIRCUIT_COOLDOWN_S:.0f}s."
            )

    def _record_success(self) -> None:
        self._consecutive_failures = 0

    # ------------------------------------------------------------------
    # Groq call
    # ------------------------------------------------------------------

    def _cache_key(self, query: str, pool: "list[RawSearchResult]") -> str:
        ids = ",".join(c.chunk_id for c in pool)
        return hashlib.sha256(f"{query}|{ids}".encode("utf-8")).hexdigest()

    def _call_groq(self, query: str, pool: "list[RawSearchResult]") -> list[int]:
        api_key = os.environ.get("GROQ_API_KEY", "")
        model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set")

        numbered = "\n\n".join(
            f"[{i}] {c.content[:TRUNCATE_CHARS]}" for i, c in enumerate(pool)
        )
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Query: {query}\n\nPassages:\n{numbered}"},
            ],
            "temperature": 0,
            "max_tokens": 300,
            # Reasoning models (gpt-oss on Groq) spend part of max_tokens on a
            # hidden reasoning trace before the JSON body -- "low" keeps that
            # trace short enough to leave room for the actual answer on a
            # 20-candidate pool. response_format must be strict json_schema,
            # not the older json_object: json_object only nudges the shape,
            # and this model reliably collapses {"ranking":[0,1,2]} into
            # {"ranking":["012"]} under it. "strict": True is what actually
            # forces separate integer array elements.
            "reasoning_effort": "low",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "reranking",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "ranking": {"type": "array", "items": {"type": "integer"}},
                        },
                        "required": ["ranking"],
                        "additionalProperties": False,
                    },
                },
            },
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        with httpx.Client(timeout=TIMEOUT_S) as client:
            resp = client.post(_GROQ_URL, json=payload, headers=headers)
            if resp.status_code != 200:
                raise RuntimeError(f"Groq rerank call returned {resp.status_code}: {resp.text[:200]}")
            body = resp.json()

        content = body["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        order = parsed["ranking"]
        if not isinstance(order, list):
            raise ValueError(f"'ranking' is not a list: {order!r}")
        return [int(i) for i in order]

    @staticmethod
    def _is_valid_permutation(order: list[int], n: int) -> bool:
        return sorted(order) == list(range(n))

    @staticmethod
    def _apply_order(pool: "list[RawSearchResult]", order: list[int]) -> "list[RawSearchResult]":
        return [pool[i] for i in order]
