"""STEP 4: Provider-neutral embeddings with batching, retries and a persistent cache.

* One client per process (the pipeline is cached by the UI, so the model is reused).
* SQLite cache keyed by (provider, model, sha256(text)) - re-ingesting a
  document or asking a repeated question costs zero embedding API calls.
* Order-preserving batching, vector validation and L2 normalisation (so cosine
  similarity is well defined across providers).
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import threading
from typing import Any, Sequence

from utils.config import Config, settings
from utils.logger import get_logger
from utils.retry import retry_call
from utils.text import sha256_text

logger = get_logger(__name__)


class EmbeddingCache:
    """Small thread-safe SQLite key/value store for embedding vectors."""

    def __init__(self, path: str) -> None:
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
        with self._lock, self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector TEXT NOT NULL)")

    def get_many(self, keys: Sequence[str]) -> dict[str, list[float]]:
        found: dict[str, list[float]] = {}
        with self._lock:
            for start in range(0, len(keys), 500):
                part = list(keys[start : start + 500])
                placeholders = ",".join("?" * len(part))
                rows = self._conn.execute(
                    f"SELECT key, vector FROM embeddings WHERE key IN ({placeholders})", part
                ).fetchall()
                found.update({key: json.loads(vector) for key, vector in rows})
        return found

    def put_many(self, items: dict[str, list[float]]) -> None:
        if not items:
            return
        with self._lock, self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO embeddings (key, vector) VALUES (?, ?)",
                [(key, json.dumps(vector)) for key, vector in items.items()],
            )


class EmbeddingService:
    """Validated embedding service shared by ingestion and retrieval."""

    SUPPORTED_PROVIDERS = frozenset({"openai", "azure", "gemini"})

    def __init__(
        self,
        *,
        batch_size: int | None = None,
        client: Any | None = None,
        model_name: str | None = None,
        config: Config | None = None,
        cache: EmbeddingCache | None = None,
    ) -> None:
        self.config = config or settings
        self.batch_size = batch_size or self.config.embedding_batch_size
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        self.provider = self.config.embedding_provider.strip().lower()
        if client is not None:
            self.client = client
            self.model_name = model_name or getattr(client, "model", None) or "custom"
            self.provider = "custom"
        else:
            if self.provider not in self.SUPPORTED_PROVIDERS:
                raise ValueError(
                    f"Unsupported embedding provider '{self.provider}'. "
                    f"Expected one of: {', '.join(sorted(self.SUPPORTED_PROVIDERS))}."
                )
            self.config.validate_embedding()
            self.client = self._create_client()
        if cache is not None:
            self.cache: EmbeddingCache | None = cache
        elif self.config.embedding_cache_enabled and client is None:
            self.cache = EmbeddingCache(self.config.embedding_cache_path)
        else:
            self.cache = None
        logger.info(
            "EmbeddingService ready (provider=%s, model=%s, batch_size=%d, cache=%s)",
            self.provider,
            self.model_name,
            self.batch_size,
            self.cache is not None,
        )

    def _create_client(self) -> Any:
        cfg = self.config
        timeout = cfg.request_timeout_seconds
        if self.provider == "gemini":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            self.model_name = cfg.gemini_embedding_model
            return GoogleGenerativeAIEmbeddings(model=self.model_name, google_api_key=cfg.gemini_api_key)
        if self.provider == "azure":
            from langchain_openai import AzureOpenAIEmbeddings

            self.model_name = cfg.azure_openai_embedding_deployment
            return AzureOpenAIEmbeddings(
                azure_deployment=self.model_name,
                azure_endpoint=cfg.azure_openai_endpoint,
                api_key=cfg.azure_openai_api_key,
                api_version=cfg.azure_openai_api_version,
                timeout=timeout,
                max_retries=0,
            )
        from langchain_openai import OpenAIEmbeddings

        self.model_name = cfg.embedding_model
        return OpenAIEmbeddings(model=self.model_name, api_key=cfg.openai_api_key, timeout=timeout, max_retries=0)

    # ------------------------------------------------------------------
    @property
    def cache_namespace(self) -> str:
        return f"{self.provider}:{self.model_name}"

    def _key(self, text: str, kind: str) -> str:
        return f"{self.cache_namespace}:{kind}:{sha256_text(text)}"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts in bounded batches, preserving order; cached vectors are reused."""
        normalized = self._validate_texts(texts)
        if not normalized:
            return []
        keys = [self._key(text, "doc") for text in normalized]
        cached = self.cache.get_many(keys) if self.cache else {}
        missing = [i for i, key in enumerate(keys) if key not in cached]
        logger.info(
            "Embedding %d texts (%d cached, %d new, batch=%d)",
            len(normalized),
            len(normalized) - len(missing),
            len(missing),
            self.batch_size,
        )
        fresh: dict[str, list[float]] = {}
        for start in range(0, len(missing), self.batch_size):
            indices = missing[start : start + self.batch_size]
            batch = [normalized[i] for i in indices]
            vectors = retry_call(
                lambda b=batch: self.client.embed_documents(b),
                operation=f"embed_documents[{start}:{start + len(batch)}]",
                max_retries=self.config.max_retries,
                base_delay=self.config.retry_base_delay_seconds,
            )
            if len(vectors) != len(batch):
                raise RuntimeError(
                    f"Embedding provider returned {len(vectors)} vectors for {len(batch)} texts."
                )
            batch_fresh = {keys[i]: self._normalize(v) for i, v in zip(indices, vectors)}
            if self.cache:
                self.cache.put_many(batch_fresh)
            fresh.update(batch_fresh)
        result = [cached.get(key) or fresh[key] for key in keys]
        self._validate_vectors(result, expected_count=len(normalized))
        return result

    def embed_query(self, text: str) -> list[float]:
        query = self._validate_text(text, field_name="query")
        key = self._key(query, "query")
        if self.cache:
            hit = self.cache.get_many([key])
            if key in hit:
                return hit[key]
        vector = retry_call(
            lambda: self.client.embed_query(query),
            operation="embed_query",
            max_retries=self.config.max_retries,
            base_delay=self.config.retry_base_delay_seconds,
        )
        vector = self._normalize(vector)
        self._validate_vectors([vector], expected_count=1)
        if self.cache:
            self.cache.put_many({key: vector})
        return vector

    def get_langchain_embeddings(self) -> Any:
        """Return the underlying LangChain embeddings client (kept for compatibility)."""
        return self.client

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(vector: Sequence[float]) -> list[float]:
        values = [float(v) for v in vector]
        norm = math.sqrt(sum(v * v for v in values))
        if norm == 0 or not math.isfinite(norm):
            raise RuntimeError("Embedding provider returned a zero or invalid vector.")
        return [v / norm for v in values]

    @staticmethod
    def _validate_texts(texts: Sequence[str]) -> list[str]:
        if texts is None:
            raise TypeError("texts must be a sequence of strings, not None.")
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings, not a single string.")
        return [EmbeddingService._validate_text(t, field_name=f"texts[{i}]") for i, t in enumerate(texts)]

    @staticmethod
    def _validate_text(text: str, *, field_name: str) -> str:
        if not isinstance(text, str):
            raise TypeError(f"{field_name} must be a string.")
        normalized = text.strip()
        if not normalized:
            raise ValueError(f"{field_name} must not be empty or whitespace-only.")
        return normalized

    @staticmethod
    def _validate_vectors(vectors: Sequence[Sequence[float]], *, expected_count: int) -> None:
        if len(vectors) != expected_count:
            raise RuntimeError(f"Expected {expected_count} embedding vectors, received {len(vectors)}.")
        if not vectors:
            return
        dimension = len(vectors[0])
        if dimension == 0:
            raise RuntimeError("Embedding provider returned an empty vector.")
        if any(len(v) != dimension for v in vectors):
            raise RuntimeError("Embedding vectors have inconsistent dimensions.")
