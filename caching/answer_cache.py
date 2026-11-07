"""Persistent exact + semantic answer cache.

Lookup order for a question inside a *scope*:
  1. exact   - same normalised text (no embedding call at all)
  2. semantic - nearest cached question by cosine similarity, accepted only if
                similarity >= threshold AND the "signature" guard passes
                (same numbers and same meaning-flipping qualifiers such as
                in-network / out-of-network / not / family / individual).

A scope is a hash of everything that changes an answer: model, prompt, retrieval
settings, mode, filters, top-k and a content fingerprint of the indexed documents.
Entries never match across scopes, so changing documents or settings can never
serve a stale answer. Metadata (vectors) lives in memory; payloads live in SQLite.
Question text is never logged (hash prefix only).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional, Sequence

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)

SCHEMA_VERSION = 1
_MAX_PAYLOAD_BYTES = 512 * 1024
_SEMANTIC_CANDIDATES = 5

# ----------------------------------------------------------------------------
# Guard: tokens that flip the meaning of a question but barely move embeddings.
# Edit these sets to match your documents.
# ----------------------------------------------------------------------------
_PHRASES = [
    (re.compile(r"\bout[\s-]*of[\s-]*network\b|\bout[\s-]*network\b|\boon\b"), " outofnetwork "),
    (re.compile(r"\bin[\s-]*network\b"), " innetwork "),
    (re.compile(r"\bnon[\s-]*preferred\b"), " nonpreferred "),
    (re.compile(r"\bout[\s-]*of[\s-]*pocket\b"), " outofpocket "),
]
QUALIFIERS = frozenset(
    """innetwork outofnetwork outofpocket nonpreferred preferred not no non never without except excluding
    excluded family individual single employee spouse child children dependent dependents inpatient outpatient
    emergency urgent preventive generic brand specialist primary mail retail maximum minimum max min copay
    copayment coinsurance deductible premium""".split()
)
_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_WORD_RE = re.compile(r"[a-z]+")


def normalize_question(text: str) -> str:
    s = unicodedata.normalize("NFKC", text or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s.rstrip(" ?!.,;:")


def question_signature(text: str) -> frozenset[str]:
    s = normalize_question(text)
    for pattern, repl in _PHRASES:
        s = pattern.sub(repl, s)
    nums = {"#" + n.replace(",", "").rstrip(".") for n in _NUM_RE.findall(s)}
    words = {w for w in _WORD_RE.findall(s) if w in QUALIFIERS}
    return frozenset(nums | words)


def make_scope(parts: dict[str, Any]) -> str:
    raw = json.dumps(parts, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _exact_key(scope: str, question: str) -> str:
    return hashlib.sha256(f"{scope}\x00{normalize_question(question)}".encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------
@dataclass
class CacheLookup:
    status: str  # hit_exact | hit_semantic | miss | bypass
    payload: Optional[dict[str, Any]] = None
    similarity: Optional[float] = None
    reason: str = ""
    key: str = ""
    vector: Optional[np.ndarray] = None
    age_seconds: float = 0.0

    @property
    def hit(self) -> bool:
        return self.status in ("hit_exact", "hit_semantic")


@dataclass
class _Meta:
    key: str
    scope: str
    signature: frozenset
    expires_at: float
    last_hit_at: float
    created_at: float
    vector: Optional[np.ndarray]


@dataclass
class _ScopeIndex:
    ids: list = field(default_factory=list)
    matrix: Optional[np.ndarray] = None
    dirty: bool = False


def _unit(vector: Sequence[float]) -> Optional[np.ndarray]:
    v = np.asarray(vector, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 and np.isfinite(n) else None


class AnswerCache:
    def __init__(
        self,
        path: str,
        *,
        ttl_seconds: int = 86400,
        max_entries: int = 1000,
        similarity_threshold: float = 0.93,
        semantic_enabled: bool = True,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self.threshold = similarity_threshold
        self.semantic_enabled = semantic_enabled
        self._clock = clock
        self._lock = threading.RLock()
        self._stats: Counter = Counter()
        self._meta: dict[str, _Meta] = {}
        self._index: dict[str, _ScopeIndex] = {}
        self._flights: dict[str, list] = {}
        self._flight_guard = threading.Lock()
        if path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
        with self._lock, self._conn:
            if path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS answer_cache (
                    key TEXT PRIMARY KEY, scope TEXT NOT NULL, question TEXT NOT NULL, norm_question TEXT NOT NULL,
                    payload TEXT NOT NULL, meta TEXT NOT NULL, vector BLOB, dim INTEGER,
                    created_at REAL NOT NULL, expires_at REAL NOT NULL, last_hit_at REAL NOT NULL,
                    hit_count INTEGER NOT NULL DEFAULT 0, schema INTEGER NOT NULL)"""
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_ac_scope ON answer_cache(scope)")
        self._load()
        logger.info(
            "AnswerCache ready (entries=%d, ttl=%ss, max=%d, semantic=%s, threshold=%.2f)",
            len(self._meta), ttl_seconds, max_entries, semantic_enabled, similarity_threshold,
        )

    # ------------------------------------------------------------------ setup
    def _load(self) -> None:
        now = self._clock()
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM answer_cache WHERE expires_at <= ? OR schema != ?", (now, SCHEMA_VERSION))
            rows = self._conn.execute(
                "SELECT key, scope, norm_question, vector, dim, created_at, expires_at, last_hit_at FROM answer_cache"
            ).fetchall()
        for key, scope, norm_q, blob, dim, created, expires, last_hit in rows:
            vec = np.frombuffer(blob, dtype=np.float32).copy() if blob and dim else None
            self._add_meta(_Meta(key, scope, question_signature(norm_q), expires, last_hit, created, vec))

    def _add_meta(self, m: _Meta) -> None:
        self._meta[m.key] = m
        if m.vector is None:
            return
        idx = self._index.setdefault(m.scope, _ScopeIndex())
        idx.ids.append(m.key)
        if idx.dirty:
            return
        row = m.vector.reshape(1, -1)
        if idx.matrix is None:
            idx.matrix = row
        elif idx.matrix.shape[1] == row.shape[1]:
            idx.matrix = np.vstack([idx.matrix, row])
        else:
            idx.dirty = True

    def _drop(self, keys: Sequence[str]) -> None:
        """Remove from memory and SQLite. Caller holds the lock."""
        for key in keys:
            m = self._meta.pop(key, None)
            if m is not None and m.scope in self._index:
                self._index[m.scope].dirty = True
        if keys:
            with self._conn:
                self._conn.executemany("DELETE FROM answer_cache WHERE key = ?", [(k,) for k in keys])

    def _matrix(self, scope: str) -> tuple[list, Optional[np.ndarray]]:
        idx = self._index.get(scope)
        if idx is None:
            return [], None
        if idx.dirty:
            ids = [k for k in idx.ids if k in self._meta and self._meta[k].vector is not None]
            dims = {self._meta[k].vector.shape[0] for k in ids}
            if len(dims) == 1:
                idx.matrix = np.vstack([self._meta[k].vector for k in ids])
            else:
                ids, idx.matrix = [], None
            idx.ids, idx.dirty = ids, False
            if not ids:
                self._index.pop(scope, None)
                return [], None
        return idx.ids, idx.matrix

    # ----------------------------------------------------------------- lookup
    def lookup(
        self, question: str, scope: str, *, embed: Optional[Callable[[str], Sequence[float]]] = None,
        count_miss: bool = True,
    ) -> CacheLookup:
        started = time.perf_counter()
        key = _exact_key(scope, question)
        qhash = key[:10]
        if count_miss:
            self._stats["lookups"] += 1
        now = self._clock()
        try:
            with self._lock:
                m = self._meta.get(key)
                if m is not None and m.expires_at <= now:
                    self._drop([key]); self._stats["expired"] += 1; m = None
                payload = self._read_payload(key, now) if m is not None else None
            if payload is not None:
                return self._hit("hit_exact", key, qhash, payload, 1.0, started)

            vector = None
            if self.semantic_enabled and embed is not None:
                vector = _unit(embed(question))
                if vector is not None:
                    return self._semantic(question, scope, key, qhash, vector, now, started)
            if not count_miss:
                return CacheLookup("miss", None, None, "no_exact_match", key)
            return self._miss(key, qhash, "no_exact_match", None, None)
        except Exception as exc:  # noqa: BLE001 - the cache must never break answering
            self._stats["errors"] += 1
            logger.warning("Answer cache lookup failed (%s); continuing without cache", type(exc).__name__)
            return CacheLookup("bypass", reason="cache_error", key=key)

    def _semantic(self, question, scope, key, qhash, vector, now, started) -> CacheLookup:
        sig = question_signature(question)
        best = None
        with self._lock:
            ids, matrix = self._matrix(scope)
            if matrix is None or matrix.shape[1] != vector.shape[0]:
                return self._miss(key, qhash, "no_entries", None, vector)
            sims = matrix @ vector
            order = np.argsort(-sims)[:_SEMANTIC_CANDIDATES]
            best = float(sims[order[0]])
            for i in order:
                sim = float(sims[i])
                if sim < self.threshold:
                    break
                cand = self._meta.get(ids[int(i)])
                if cand is None:
                    continue
                if cand.expires_at <= now:
                    self._drop([cand.key]); self._stats["expired"] += 1
                    continue
                if cand.signature != sig:
                    self._stats["guard_rejections"] += 1
                    logger.info("Answer cache guard rejected near-match (qhash=%s, sim=%.3f)", qhash, sim)
                    continue
                payload = self._read_payload(cand.key, now)
                if payload is not None:
                    return self._hit("hit_semantic", cand.key, qhash, payload, sim, started, vector)
        reason = "below_threshold" if best is not None and best < self.threshold else "guard_mismatch"
        return self._miss(key, qhash, reason, best, vector)

    def _read_payload(self, key: str, now: float) -> Optional[dict[str, Any]]:
        row = self._conn.execute("SELECT payload, meta, created_at FROM answer_cache WHERE key = ?", (key,)).fetchone()
        if row is None:
            self._meta.pop(key, None)
            return None
        with self._conn:
            self._conn.execute(
                "UPDATE answer_cache SET hit_count = hit_count + 1, last_hit_at = ? WHERE key = ?", (now, key)
            )
        if key in self._meta:
            self._meta[key].last_hit_at = now
        payload = json.loads(row[0])
        payload["_meta"] = json.loads(row[1])
        payload["_age"] = max(0.0, now - row[2])
        return payload

    def _hit(self, status, key, qhash, payload, sim, started, vector=None) -> CacheLookup:
        meta = payload.get("_meta") or {}
        took_ms = (time.perf_counter() - started) * 1000
        self._stats["hits"] += 1
        self._stats[status + "s"] += 1
        self._stats["saved_llm_calls"] += 1
        self._stats["saved_tokens"] += int(meta.get("total_tokens") or 0)
        self._stats["saved_latency_ms"] += int(max(0.0, float(meta.get("latency_ms") or 0) - took_ms))
        logger.info(
            "Answer cache HIT type=%s sim=%.3f age_s=%d qhash=%s lookup_ms=%.1f saved_tokens=%s",
            status.split("_")[1], sim, payload.get("_age", 0), qhash, took_ms, meta.get("total_tokens", 0),
            extra={"cache_event": "hit", "cache_type": status, "cache_similarity": round(sim, 4)},
        )
        return CacheLookup(status, payload, sim, key=key, vector=vector, age_seconds=payload.get("_age", 0.0))

    def _miss(self, key, qhash, reason, best, vector) -> CacheLookup:
        self._stats["misses"] += 1
        self._stats["miss_" + reason] += 1
        logger.info(
            "Answer cache MISS reason=%s best_sim=%s qhash=%s", reason,
            "n/a" if best is None else f"{best:.3f}", qhash,
            extra={"cache_event": "miss", "cache_reason": reason},
        )
        return CacheLookup("miss", None, best, reason, key, vector)

    def record_bypass(self, reason: str) -> None:
        self._stats["bypass"] += 1
        self._stats["bypass_" + reason] += 1
        logger.info("Answer cache BYPASS reason=%s", reason, extra={"cache_event": "bypass", "cache_reason": reason})

    # ------------------------------------------------------------------ store
    def store(
        self, question: str, scope: str, payload: dict[str, Any], *,
        meta: Optional[dict[str, Any]] = None, vector: Optional[np.ndarray] = None,
    ) -> bool:
        try:
            body = json.dumps(payload, default=str, ensure_ascii=False)
            if len(body) > _MAX_PAYLOAD_BYTES:
                self._stats["store_skipped_too_large"] += 1
                return False
            key = _exact_key(scope, question)
            norm = normalize_question(question)
            now = self._clock()
            vec = vector if (self.semantic_enabled and vector is not None) else None
            with self._lock:
                if key in self._meta:
                    self._drop([key])
                with self._conn:
                    self._conn.execute(
                        "INSERT OR REPLACE INTO answer_cache VALUES (?,?,?,?,?,?,?,?,?,?,?,0,?)",
                        (key, scope, question, norm, body, json.dumps(meta or {}, default=str),
                         vec.tobytes() if vec is not None else None, int(vec.shape[0]) if vec is not None else 0,
                         now, now + self.ttl, now, SCHEMA_VERSION),
                    )
                self._add_meta(_Meta(key, scope, question_signature(norm), now + self.ttl, now, now, vec))
                self._stats["stores"] += 1
                self._evict_if_needed(now)
            logger.info("Answer cache STORE qhash=%s entries=%d", key[:10], len(self._meta),
                        extra={"cache_event": "store"})
            return True
        except Exception as exc:  # noqa: BLE001
            self._stats["errors"] += 1
            logger.warning("Answer cache store failed (%s)", type(exc).__name__)
            return False

    def _evict_if_needed(self, now: float) -> None:
        expired = [k for k, m in self._meta.items() if m.expires_at <= now]
        if expired:
            self._drop(expired); self._stats["expired"] += len(expired)
        if len(self._meta) > self.max_entries:
            target = int(self.max_entries * 0.9)
            victims = sorted(self._meta.values(), key=lambda m: m.last_hit_at)[: len(self._meta) - target]
            self._drop([m.key for m in victims])
            self._stats["evictions"] += len(victims)

    # ----------------------------------------------------------- invalidation
    def invalidate(self, reason: str = "manual") -> int:
        with self._lock:
            count = len(self._meta)
            with self._conn:
                self._conn.execute("DELETE FROM answer_cache")
            self._meta.clear(); self._index.clear()
            self._stats["invalidations"] += 1
            self._stats["invalidated_entries"] += count
        logger.info("Answer cache INVALIDATED reason=%s entries_removed=%d", reason, count,
                    extra={"cache_event": "invalidate", "cache_reason": reason})
        return count

    # ------------------------------------------------------------ single flight
    @contextlib.contextmanager
    def inflight(self, key: str, timeout: float = 120.0) -> Iterator[None]:
        """Serialise identical concurrent questions so only one calls the LLM."""
        with self._flight_guard:
            entry = self._flights.setdefault(key, [threading.Lock(), 0])
            entry[1] += 1
        acquired = entry[0].acquire(timeout=timeout)
        try:
            yield
        finally:
            if acquired:
                entry[0].release()
            with self._flight_guard:
                entry[1] -= 1
                if entry[1] <= 0:
                    self._flights.pop(key, None)

    def peek_exact(self, question: str, scope: str) -> CacheLookup:
        """Exact-only re-check used after waiting on another thread's in-flight call."""
        return self.lookup(question, scope, embed=None, count_miss=False)

    # ------------------------------------------------------------------ stats
    def stats(self) -> dict[str, Any]:
        with self._lock:
            s = dict(self._stats)
            s["entries"] = len(self._meta)
        decided = s.get("hits", 0) + s.get("misses", 0)
        s["hit_rate"] = round(s.get("hits", 0) / decided, 4) if decided else 0.0
        return s

    def close(self) -> None:
        with self._lock:
            self._conn.close()
