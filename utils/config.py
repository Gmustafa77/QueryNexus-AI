"""Centralized, environment-driven configuration.

All settings come from environment variables (optionally loaded from a local
``.env`` file). No secrets have defaults. Import the shared instance with::

    from utils.config import settings
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields, replace
from typing import Any

from dotenv import load_dotenv

load_dotenv()

_SUPPORTED_PROVIDERS = ("openai", "azure", "gemini", "databricks")
_SUPPORTED_VECTOR_STORES = ("chroma", "databricks")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PLACEHOLDER_SECRETS = {"", "api_key", "your-key-here", "changeme", "sk-..."}


def _env_str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {raw!r}.") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"Environment variable {name} must be >= {minimum}, got {value}.")
    return value


def _env_float(name: str, default: float, *, low: float | None = None, high: float | None = None) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw.strip())
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a number, got {raw!r}.") from exc
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"Environment variable {name} must be within [{low}, {high}], got {value}.")
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _databricks_host() -> str:
    """Accepts DATABRICKS_HOST (URL) or DATABRICKS_SERVER_HOSTNAME (bare host)."""
    host = _env_str("DATABRICKS_HOST") or _env_str("DATABRICKS_SERVER_HOSTNAME")
    if host and not host.startswith(("https://", "http://")):
        host = "https://" + host
    return host.rstrip("/")


def _databricks_http_path() -> str:
    """DATABRICKS_HTTP_PATH wins; otherwise build it from DATABRICKS_WAREHOUSE_ID."""
    path = _env_str("DATABRICKS_HTTP_PATH")
    if path:
        return path
    warehouse_id = _env_str("DATABRICKS_WAREHOUSE_ID")
    return f"/sql/1.0/warehouses/{warehouse_id}" if warehouse_id else ""


def _legacy_chunk_tokens(token_var: str, char_var: str, default: int) -> int:
    """Prefer token-based settings; convert legacy character settings (~4 chars/token)."""
    if os.getenv(token_var, "").strip():
        return _env_int(token_var, default, minimum=0)
    if os.getenv(char_var, "").strip():
        return max(0, _env_int(char_var, default * 4, minimum=0) // 4)
    return default


@dataclass(frozen=True)
class Config:
    """Typed, immutable application settings."""

    # --- Providers -------------------------------------------------------------
    chat_provider: str = field(default_factory=lambda: _env_str("CHAT_PROVIDER").lower())
    use_azure_openai: bool = field(default_factory=lambda: _env_bool("USE_AZURE_OPENAI", False))
    embedding_provider: str = field(default_factory=lambda: _env_str("EMBEDDING_PROVIDER", "openai").lower())

    # --- OpenAI ----------------------------------------------------------------
    openai_api_key: str = field(default_factory=lambda: _env_str("OPENAI_API_KEY"))
    chat_model: str = field(default_factory=lambda: _env_str("OPENAI_CHAT_MODEL", "gpt-4o"))
    embedding_model: str = field(default_factory=lambda: _env_str("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"))

    # --- Azure OpenAI ----------------------------------------------------------
    azure_openai_api_key: str = field(default_factory=lambda: _env_str("AZURE_OPENAI_API_KEY"))
    azure_openai_endpoint: str = field(default_factory=lambda: _env_str("AZURE_OPENAI_ENDPOINT"))
    azure_openai_api_version: str = field(default_factory=lambda: _env_str("AZURE_OPENAI_API_VERSION", "2024-02-01"))
    azure_openai_chat_deployment: str = field(default_factory=lambda: _env_str("AZURE_OPENAI_CHAT_DEPLOYMENT"))
    azure_openai_embedding_deployment: str = field(default_factory=lambda: _env_str("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"))

    # --- Gemini ----------------------------------------------------------------
    gemini_api_key: str = field(default_factory=lambda: _env_str("GEMINI_API_KEY"))
    gemini_embedding_model: str = field(default_factory=lambda: _env_str("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"))
    gemini_chat_model: str = field(default_factory=lambda: _env_str("GEMINI_CHAT_MODEL", "gemini-2.0-flash"))

    # --- Databricks (vector store and/or model serving) --------------------------
    databricks_host: str = field(default_factory=_databricks_host)
    databricks_token: str = field(default_factory=lambda: _env_str("DATABRICKS_TOKEN"))
    databricks_http_path: str = field(default_factory=_databricks_http_path)
    databricks_catalog: str = field(default_factory=lambda: _env_str("DATABRICKS_CATALOG"))
    databricks_schema: str = field(default_factory=lambda: _env_str("DATABRICKS_SCHEMA_NAME"))
    databricks_chunk_table: str = field(default_factory=lambda: _env_str("DATABRICKS_CHUNK_TABLE", "rag_chunks"))
    databricks_vs_endpoint: str = field(default_factory=lambda: _env_str("DATABRICKS_VECTOR_SEARCH_ENDPOINT"))
    databricks_vs_index: str = field(default_factory=lambda: _env_str("DATABRICKS_VECTOR_INDEX", "rag_chunks_index"))
    databricks_sync_timeout_seconds: int = field(
        default_factory=lambda: _env_int("DATABRICKS_SYNC_TIMEOUT_SECONDS", 900, minimum=0)
    )
    databricks_sync_poll_seconds: float = field(
        default_factory=lambda: _env_float("DATABRICKS_SYNC_POLL_SECONDS", 10.0, low=0.0)
    )
    databricks_chat_endpoint: str = field(default_factory=lambda: _env_str("DATABRICKS_CHAT_ENDPOINT"))
    databricks_embedding_endpoint: str = field(default_factory=lambda: _env_str("DATABRICKS_EMBEDDING_ENDPOINT"))

    # --- LLM call behaviour ----------------------------------------------------
    llm_temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.0, low=0.0, high=2.0))
    llm_max_output_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_OUTPUT_TOKENS", 1200, minimum=64))
    request_timeout_seconds: int = field(default_factory=lambda: _env_int("REQUEST_TIMEOUT_SECONDS", 60, minimum=1))
    max_retries: int = field(default_factory=lambda: _env_int("MAX_RETRIES", 3, minimum=0))
    retry_base_delay_seconds: float = field(default_factory=lambda: _env_float("RETRY_BASE_DELAY_SECONDS", 1.0, low=0.0))

    # --- Embeddings ------------------------------------------------------------
    embedding_batch_size: int = field(default_factory=lambda: _env_int("EMBEDDING_BATCH_SIZE", 64, minimum=1))
    embedding_cache_enabled: bool = field(default_factory=lambda: _env_bool("EMBEDDING_CACHE_ENABLED", True))
    embedding_cache_path: str = field(default_factory=lambda: _env_str("EMBEDDING_CACHE_PATH", "data/embedding_cache.sqlite3"))

    # --- Vector store ----------------------------------------------------------
    # VECTOR_STORE sets the DEFAULT knowledge source; the UI can switch between
    # Local docs, Databricks and Hybrid at runtime.
    vector_store: str = field(default_factory=lambda: _env_str("VECTOR_STORE", "chroma").lower())
    chroma_persist_dir: str = field(default_factory=lambda: _env_str("CHROMA_PERSIST_DIR", "data/chroma_db"))
    chroma_collection_name: str = field(default_factory=lambda: _env_str("CHROMA_COLLECTION_NAME", "querynexus_docs"))

    # --- Chunking (tokens) -----------------------------------------------------
    chunk_size: int = field(default_factory=lambda: _legacy_chunk_tokens("CHUNK_SIZE_TOKENS", "CHUNK_SIZE", 350))
    chunk_overlap: int = field(default_factory=lambda: _legacy_chunk_tokens("CHUNK_OVERLAP_TOKENS", "CHUNK_OVERLAP", 50))
    min_chunk_tokens: int = field(default_factory=lambda: _env_int("MIN_CHUNK_TOKENS", 40, minimum=1))

    # --- Retrieval -------------------------------------------------------------
    retrieval_top_k: int = field(default_factory=lambda: _env_int("RETRIEVAL_TOP_K", 5, minimum=1))
    retrieval_candidate_k: int = field(default_factory=lambda: _env_int("RETRIEVAL_CANDIDATE_K", 25, minimum=1))
    hybrid_search_enabled: bool = field(default_factory=lambda: _env_bool("HYBRID_SEARCH_ENABLED", True))
    rrf_k: int = field(default_factory=lambda: _env_int("RRF_K", 60, minimum=1))
    vector_weight: float = field(default_factory=lambda: _env_float("VECTOR_WEIGHT", 1.0, low=0.0))
    keyword_weight: float = field(default_factory=lambda: _env_float("KEYWORD_WEIGHT", 1.0, low=0.0))
    min_similarity: float = field(default_factory=lambda: _env_float("MIN_SIMILARITY", 0.30, low=-1.0, high=1.0))
    min_keyword_coverage: float = field(default_factory=lambda: _env_float("MIN_KEYWORD_COVERAGE", 0.60, low=0.0, high=1.0))
    reranker: str = field(default_factory=lambda: _env_str("RERANKER", "none").lower())
    reranker_model: str = field(default_factory=lambda: _env_str("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    max_context_tokens: int = field(default_factory=lambda: _env_int("MAX_CONTEXT_TOKENS", 6000, minimum=200))
    near_duplicate_threshold: float = field(default_factory=lambda: _env_float("NEAR_DUPLICATE_THRESHOLD", 0.85, low=0.0, high=1.0))

    # --- Answer cache (exact + semantic) ----------------------------------------
    answer_cache_enabled: bool = field(default_factory=lambda: _env_bool("ANSWER_CACHE_ENABLED", True))
    answer_cache_semantic_enabled: bool = field(default_factory=lambda: _env_bool("ANSWER_CACHE_SEMANTIC_ENABLED", True))
    answer_cache_path: str = field(default_factory=lambda: _env_str("ANSWER_CACHE_PATH", "data/answer_cache.sqlite3"))
    answer_cache_ttl_seconds: int = field(default_factory=lambda: _env_int("ANSWER_CACHE_TTL_SECONDS", 86400, minimum=1))
    answer_cache_max_entries: int = field(default_factory=lambda: _env_int("ANSWER_CACHE_MAX_ENTRIES", 1000, minimum=10))
    answer_cache_similarity_threshold: float = field(
        default_factory=lambda: _env_float("ANSWER_CACHE_SIMILARITY_THRESHOLD", 0.93, low=0.5, high=1.0)
    )
    # How long the document list (used by the sidebar and the cache fingerprint) is reused. 0 disables.
    doc_list_cache_ttl_seconds: int = field(default_factory=lambda: _env_int("DOC_LIST_CACHE_TTL_SECONDS", 30, minimum=0))

    # --- Grounding / safety ----------------------------------------------------
    strict_citations: bool = field(default_factory=lambda: _env_bool("STRICT_CITATIONS", True))
    max_question_chars: int = field(default_factory=lambda: _env_int("MAX_QUESTION_CHARS", 2000, minimum=10))

    # --- Files -----------------------------------------------------------------
    pdf_upload_dir: str = field(default_factory=lambda: _env_str("PDF_UPLOAD_DIR", "data/pdfs"))
    max_upload_mb: int = field(default_factory=lambda: _env_int("MAX_UPLOAD_MB", 50, minimum=1))
    max_pdf_pages: int = field(default_factory=lambda: _env_int("MAX_PDF_PAGES", 2000, minimum=1))

    # --- Logging ---------------------------------------------------------------
    log_level: str = field(default_factory=lambda: _env_str("LOG_LEVEL", "INFO").upper())
    log_format: str = field(default_factory=lambda: _env_str("LOG_FORMAT", "text").lower())

    # ------------------------------------------------------------------------
    @property
    def resolved_chat_provider(self) -> str:
        """CHAT_PROVIDER wins; legacy USE_AZURE_OPENAI is honoured as a fallback."""
        if self.chat_provider in _SUPPORTED_PROVIDERS:
            return self.chat_provider
        return "azure" if self.use_azure_openai else "openai"

    @property
    def databricks_table_fqn(self) -> str:
        return f"{self.databricks_catalog}.{self.databricks_schema}.{self.databricks_chunk_table}"

    @property
    def databricks_index_fqn(self) -> str:
        return f"{self.databricks_catalog}.{self.databricks_schema}.{self.databricks_vs_index}"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def with_overrides(self, **overrides: Any) -> "Config":
        """Return a copy with selected values replaced (used by tests and evaluation)."""
        valid = {f.name for f in fields(self)}
        unknown = set(overrides) - valid
        if unknown:
            raise ValueError(f"Unknown config fields: {sorted(unknown)}")
        return replace(self, **overrides)

    # --- Validation ------------------------------------------------------------
    @staticmethod
    def _is_missing(value: str) -> bool:
        return value.strip().lower() in _PLACEHOLDER_SECRETS

    def _provider_missing(self, provider: str, purpose: str) -> list[str]:
        if provider == "openai":
            return ["OPENAI_API_KEY"] if self._is_missing(self.openai_api_key) else []
        if provider == "gemini":
            return ["GEMINI_API_KEY"] if self._is_missing(self.gemini_api_key) else []
        if provider == "databricks":
            endpoint_var = "DATABRICKS_CHAT_ENDPOINT" if purpose == "chat" else "DATABRICKS_EMBEDDING_ENDPOINT"
            endpoint = self.databricks_chat_endpoint if purpose == "chat" else self.databricks_embedding_endpoint
            pairs = [
                ("DATABRICKS_HOST", self.databricks_host),
                ("DATABRICKS_TOKEN", self.databricks_token),
                (endpoint_var, endpoint),
            ]
            return [name for name, value in pairs if self._is_missing(value)]
        if provider == "azure":
            deployment_var = (
                "AZURE_OPENAI_CHAT_DEPLOYMENT" if purpose == "chat" else "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
            )
            deployment = (
                self.azure_openai_chat_deployment if purpose == "chat" else self.azure_openai_embedding_deployment
            )
            pairs = [
                ("AZURE_OPENAI_API_KEY", self.azure_openai_api_key),
                ("AZURE_OPENAI_ENDPOINT", self.azure_openai_endpoint),
                (deployment_var, deployment),
            ]
            return [name for name, value in pairs if self._is_missing(value)]
        raise ValueError(
            f"Unknown {purpose} provider '{provider}'. Must be one of: {', '.join(_SUPPORTED_PROVIDERS)}."
        )

    def validate_chat(self) -> None:
        missing = self._provider_missing(self.resolved_chat_provider, "chat")
        if missing:
            raise ValueError(
                f"Chat provider '{self.resolved_chat_provider}' is missing required settings: {', '.join(missing)}."
            )

    def validate_embedding(self) -> None:
        missing = self._provider_missing(self.embedding_provider, "embedding")
        if missing:
            raise ValueError(
                f"Embedding provider '{self.embedding_provider}' is missing required settings: {', '.join(missing)}."
            )

    def validate_retrieval(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_SIZE_TOKENS.")
        if self.retrieval_candidate_k < self.retrieval_top_k:
            raise ValueError("RETRIEVAL_CANDIDATE_K must be >= RETRIEVAL_TOP_K.")
        if self.reranker not in {"none", "cross_encoder"}:
            raise ValueError("RERANKER must be 'none' or 'cross_encoder'.")
        if self.log_format not in {"text", "json"}:
            raise ValueError("LOG_FORMAT must be 'text' or 'json'.")

    def validate_databricks(self) -> None:
        """Validate the settings needed to use Databricks as a knowledge source."""
        pairs = [
            ("DATABRICKS_HOST", self.databricks_host),
            ("DATABRICKS_TOKEN", self.databricks_token),
            ("DATABRICKS_HTTP_PATH or DATABRICKS_WAREHOUSE_ID", self.databricks_http_path),
            ("DATABRICKS_CATALOG", self.databricks_catalog),
            ("DATABRICKS_SCHEMA_NAME", self.databricks_schema),
            ("DATABRICKS_VECTOR_SEARCH_ENDPOINT", self.databricks_vs_endpoint),
        ]
        missing = [name for name, value in pairs if self._is_missing(value)]
        if missing:
            raise ValueError(f"Databricks is missing required settings: {', '.join(missing)}.")
        for name, value in (
            ("DATABRICKS_CATALOG", self.databricks_catalog),
            ("DATABRICKS_SCHEMA_NAME", self.databricks_schema),
            ("DATABRICKS_CHUNK_TABLE", self.databricks_chunk_table),
            ("DATABRICKS_VECTOR_INDEX", self.databricks_vs_index),
        ):
            if not _IDENTIFIER_RE.match(value):
                raise ValueError(f"{name} must contain only letters, digits and underscores, got {value!r}.")

    def validate_vector_store(self) -> None:
        if self.vector_store not in _SUPPORTED_VECTOR_STORES:
            raise ValueError(f"VECTOR_STORE must be one of: {', '.join(_SUPPORTED_VECTOR_STORES)}.")
        if self.vector_store == "databricks":
            self.validate_databricks()

    def validate(self) -> None:
        """Validate everything required to run the full application."""
        self.validate_retrieval()
        self.validate_vector_store()
        self.validate_chat()
        self.validate_embedding()


settings = Config()
