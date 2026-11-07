"""RAG orchestration: ingestion (load -> chunk -> embed -> store) and grounded Q&A
(validate -> retrieve -> gate -> build context -> generate -> verify citations).

Knowledge sources ("modes"):
    local       - ChromaDB on this machine
    databricks  - Delta table + Databricks Vector Search
    hybrid      - search both and merge results; each citation shows its source

Stores are created lazily, so the app starts even when Databricks is not
configured or unreachable; that source is reported as unavailable instead.

Public API kept from earlier versions:
    RAGPipeline().ingest_documents(file_paths) -> int
    RAGPipeline().ask(question) -> RAGResponse(answer, sources, usage_metadata)
    RAGPipeline().chroma_manager / .retriever  (default source)
New:
    ask(question, mode="hybrid", filters=..., top_k=...)
    ingest_with_report(paths, targets=["local", "databricks"])
    storage_status(name), reset_storage(name), list_documents(mode)
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, List, Optional, Sequence

from caching.answer_cache import AnswerCache, make_scope
from chunking.text_splitter import DocumentChunker
from embeddings.embedding_service import EmbeddingService
from ingestion.pdf_loader import LoadResult, PDFLoader
from prompts.prompt_template import NO_ANSWER_MESSAGE, SYSTEM_PROMPT, build_user_prompt
from rag.grounding import Citation, build_context, detect_prompt_injection, verify_answer
from retrieval.reranker import build_reranker
from retrieval.retriever import STORAGE_LABELS, RetrievedChunk, Retriever
from utils.config import Config, settings
from utils.logger import get_logger, request_context
from utils.retry import retry_call
from utils.text import sha256_text
from vectorstore.chroma_manager import StoredDocument
from vectorstore.factory import STORAGE_DATABRICKS, STORAGE_LOCAL, STORAGES, create_store, storage_for_backend
from vectorstore.multi_store import MultiVectorStore

logger = get_logger(__name__)

MODE_HYBRID = "hybrid"
SOURCE_MODES = (STORAGE_LOCAL, STORAGE_DATABRICKS, MODE_HYBRID)
MODE_LABELS = {**STORAGE_LABELS, MODE_HYBRID: "Hybrid (Local + Databricks)"}


class StorageUnavailableError(RuntimeError):
    """A knowledge source cannot be used (missing settings, no access, network error)."""


# ----------------------------------------------------------------------------
# Provider response helpers
# ----------------------------------------------------------------------------
def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block.get("text", "")))
        return "".join(parts).strip()
    return "" if content is None else str(content)


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _extract_usage_metadata(ai_message: Any) -> dict[str, Any]:
    usage = getattr(ai_message, "usage_metadata", None) or {}
    response = getattr(ai_message, "response_metadata", None) or {}
    token_usage = response.get("token_usage") or response.get("usage") or {}
    input_details = usage.get("input_token_details") or {}
    output_details = usage.get("output_token_details") or {}

    def pick(*keys: str) -> int:
        for source in (usage, token_usage):
            for key in keys:
                if source.get(key):
                    return _safe_int(source.get(key))
        return 0

    input_tokens = pick("input_tokens", "prompt_tokens", "prompt_token_count")
    output_tokens = pick("output_tokens", "completion_tokens", "candidates_token_count")
    total_tokens = pick("total_tokens", "total_token_count") or input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "thinking_tokens": _safe_int(output_details.get("reasoning")) or pick("thoughts_token_count", "reasoning_tokens"),
        "cached_tokens": _safe_int(input_details.get("cache_read")) or pick("cached_tokens", "cached_content_token_count"),
        "model": response.get("model_name", "") or response.get("model", ""),
    }


# ----------------------------------------------------------------------------
# Data contracts
# ----------------------------------------------------------------------------
@dataclass
class RAGResponse:
    """Final response returned to the UI. New fields have defaults (backward compatible)."""

    answer: str
    sources: list[Any]
    usage_metadata: dict[str, Any] = field(default_factory=dict)
    citations: list[Citation] = field(default_factory=list)
    retrieved: list[Any] = field(default_factory=list)
    is_grounded: bool = False
    is_no_answer: bool = False
    warnings: list[str] = field(default_factory=list)
    latency_ms: int = 0
    request_id: str = ""
    mode: str = ""
    searched: list[str] = field(default_factory=list)
    cache_status: str = ""  # hit_exact | hit_semantic | miss | bypass | disabled
    cache_similarity: Optional[float] = None
    timings_ms: dict[str, int] = field(default_factory=dict)


@dataclass
class FileIngestionStatus:
    source_file: str
    status: str  # indexed | updated | unchanged | failed
    chunks_added: int = 0
    pages: int = 0
    message: str = ""
    storage: str = ""


@dataclass
class IngestionReport:
    files: list[FileIngestionStatus] = field(default_factory=list)

    @property
    def chunks_added(self) -> int:
        return sum(f.chunks_added for f in self.files)

    @property
    def failed(self) -> list[FileIngestionStatus]:
        return [f for f in self.files if f.status == "failed"]


@dataclass
class StorageStatus:
    name: str
    label: str
    available: bool
    chunk_count: int = 0
    error: str = ""


# ----------------------------------------------------------------------------
def create_chat_model(config: Config) -> tuple[Any, str]:
    """Create the chat model for the configured provider (single place for provider wiring)."""
    config.validate_chat()
    provider = config.resolved_chat_provider
    common = {"temperature": config.llm_temperature, "max_retries": 0}
    if provider == "databricks":
        os.environ.setdefault("DATABRICKS_HOST", config.databricks_host)
        os.environ.setdefault("DATABRICKS_TOKEN", config.databricks_token)
        from databricks_langchain import ChatDatabricks

        model = ChatDatabricks(
            endpoint=config.databricks_chat_endpoint,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_output_tokens,
        )
        return model, config.databricks_chat_endpoint
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = ChatGoogleGenerativeAI(
            model=config.gemini_chat_model,
            google_api_key=config.gemini_api_key,
            max_output_tokens=config.llm_max_output_tokens,
            timeout=config.request_timeout_seconds,
            **common,
        )
        return model, config.gemini_chat_model
    if provider == "azure":
        from langchain_openai import AzureChatOpenAI

        model = AzureChatOpenAI(
            azure_deployment=config.azure_openai_chat_deployment,
            azure_endpoint=config.azure_openai_endpoint,
            api_key=config.azure_openai_api_key,
            api_version=config.azure_openai_api_version,
            max_tokens=config.llm_max_output_tokens,
            timeout=config.request_timeout_seconds,
            **common,
        )
        return model, config.azure_openai_chat_deployment
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(
        model=config.chat_model,
        api_key=config.openai_api_key,
        max_tokens=config.llm_max_output_tokens,
        timeout=config.request_timeout_seconds,
        **common,
    )
    return model, config.chat_model


def _messages(system: str, user: str) -> list[Any]:
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        return [SystemMessage(content=system), HumanMessage(content=user)]
    except ImportError:  # pragma: no cover - langchain is a runtime dependency
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _short_error(exc: BaseException) -> str:
    message = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    return message[:300]


def _canonical_filters(filters: Optional[dict[str, Any]]) -> Any:
    if not filters:
        return None
    out: dict[str, Any] = {}
    for key, value in filters.items():
        if value is None or value == [] or value == "":
            continue
        if isinstance(value, (list, tuple, set, frozenset)):
            value = sorted(value, key=str)
        out[str(key)] = value
    return out or None


def _file_digest(module_name: str) -> str:
    try:
        module = __import__(module_name, fromlist=["_"])
        return hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()[:16]
    except Exception:  # noqa: BLE001
        return "unknown"


def _response_to_payload(response: "RAGResponse") -> dict[str, Any]:
    return {
        "answer": response.answer,
        "sources": [asdict(c) for c in response.sources],
        "retrieved": [asdict(c) for c in response.retrieved],
        "citations": [asdict(c) for c in response.citations],
        "warnings": list(response.warnings),
        "searched": list(response.searched),
        "model": (response.usage_metadata or {}).get("model", ""),
    }


def _build(cls: Any, data: dict[str, Any]) -> Any:
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})


def _payload_to_response(payload: dict[str, Any]) -> "RAGResponse":
    return RAGResponse(
        answer=payload["answer"],
        sources=[_build(RetrievedChunk, c) for c in payload.get("sources", [])],
        usage_metadata={
            "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "thinking_tokens": 0,
            "cached_tokens": 0, "model": payload.get("model", ""),
        },
        citations=[_build(Citation, c) for c in payload.get("citations", [])],
        retrieved=[_build(RetrievedChunk, c) for c in payload.get("retrieved", [])],
        is_grounded=True,
        is_no_answer=False,
        warnings=list(payload.get("warnings", [])),
        searched=list(payload.get("searched", [])),
    )


class RAGPipeline:
    """Main pipeline used by the Streamlit UI and the evaluation harness. Safe to share across sessions."""

    def __init__(
        self,
        *,
        config: Config | None = None,
        embedding_service: EmbeddingService | None = None,
        chroma_manager: Any | None = None,
        llm: Any | None = None,
        retriever: Retriever | None = None,
        stores: Optional[dict[str, Any]] = None,
    ) -> None:
        self.config = config or settings
        self.config.validate_retrieval()
        self.pdf_loader = PDFLoader(max_bytes=self.config.max_upload_bytes, max_pages=self.config.max_pdf_pages)
        self.chunker = DocumentChunker(
            self.config.chunk_size, self.config.chunk_overlap, min_chunk_tokens=self.config.min_chunk_tokens
        )
        self._lock = threading.RLock()
        self._stores: dict[str, Any] = {}
        self._store_errors: dict[str, str] = {}
        self._retrievers: dict[tuple[str, ...], Retriever] = {}
        for name, store in (stores or {}).items():
            if name not in STORAGES:
                raise ValueError(f"Unknown storage '{name}'. Expected one of: {', '.join(STORAGES)}.")
            self._stores[name] = store
        if chroma_manager is not None:
            self._stores[storage_for_backend(getattr(chroma_manager, "backend_name", "chroma"))] = chroma_manager

        first_store = next(iter(self._stores.values()), None)
        self.embedding_service = embedding_service or (
            first_store.embedding_service if first_store is not None else EmbeddingService(config=self.config)
        )
        self.default_mode = STORAGE_DATABRICKS if self.config.vector_store == "databricks" else STORAGE_LOCAL
        self._retriever_override = retriever
        self._reranker = build_reranker(self.config.reranker, self.config.reranker_model)
        if llm is not None:
            self.llm, self._chat_model_label = llm, getattr(llm, "model_name", "custom")
        else:
            self.llm, self._chat_model_label = create_chat_model(self.config)
        self._doc_list_memo: dict[Any, tuple[float, list[StoredDocument]]] = {}
        self.answer_cache: Optional[AnswerCache] = None
        self._cache_config_fp: dict[str, Any] = {}
        if self.config.answer_cache_enabled:
            try:
                self.answer_cache = AnswerCache(
                    self.config.answer_cache_path,
                    ttl_seconds=self.config.answer_cache_ttl_seconds,
                    max_entries=self.config.answer_cache_max_entries,
                    similarity_threshold=self.config.answer_cache_similarity_threshold,
                    semantic_enabled=self.config.answer_cache_semantic_enabled,
                )
                self._cache_config_fp = self._build_cache_config_fingerprint()
            except Exception as exc:  # noqa: BLE001 - never block startup because of the cache
                logger.error("Answer cache disabled (%s: %s)", type(exc).__name__, _short_error(exc))
                self.answer_cache = None
        logger.info("RAGPipeline ready (default_source=%s, chat_model=%s)", self.default_mode, self._chat_model_label)

    # ------------------------------------------------------------------
    # Stores
    # ------------------------------------------------------------------
    def get_store(self, name: str) -> Any:
        """Return the store for ``name``, creating it on first use. Raises StorageUnavailableError."""
        if name not in STORAGES:
            raise ValueError(f"Unknown storage '{name}'. Expected one of: {', '.join(STORAGES)}.")
        with self._lock:
            if name in self._stores:
                return self._stores[name]
            if name in self._store_errors:
                raise StorageUnavailableError(self._store_errors[name])
            try:
                logger.info("Connecting to %s", STORAGE_LABELS[name])
                store = create_store(name, self.config, self.embedding_service)
            except Exception as exc:  # noqa: BLE001
                message = _short_error(exc)
                self._store_errors[name] = message
                logger.error("%s unavailable: %s", STORAGE_LABELS[name], message)
                raise StorageUnavailableError(message) from exc
            self._stores[name] = store
            return store

    def reset_storage(self, name: str) -> None:
        """Forget a failed (or existing) connection so the next use retries it."""
        with self._lock:
            self._store_errors.pop(name, None)
            if name == STORAGE_DATABRICKS:
                stale = self._stores.pop(name, None)
                table = getattr(stale, "table", None)
                if table is not None and hasattr(table, "close"):
                    table.close()
            self._retrievers = {k: v for k, v in self._retrievers.items() if name not in k}

    def storage_checked(self, name: str) -> bool:
        """True once a connection to ``name`` has been attempted (successfully or not)."""
        return name in self._stores or name in self._store_errors

    def storage_status(self, name: str) -> StorageStatus:
        label = STORAGE_LABELS.get(name, name)
        try:
            store = self.get_store(name)
        except StorageUnavailableError as exc:
            return StorageStatus(name, label, False, error=str(exc))
        try:
            return StorageStatus(name, label, True, chunk_count=int(store.document_count()))
        except Exception as exc:  # noqa: BLE001
            return StorageStatus(name, label, False, error=_short_error(exc))

    @staticmethod
    def storages_for_mode(mode: str) -> list[str]:
        if mode not in SOURCE_MODES:
            raise ValueError(f"Unknown source mode '{mode}'. Expected one of: {', '.join(SOURCE_MODES)}.")
        return list(STORAGES) if mode == MODE_HYBRID else [mode]

    def _available_for_mode(self, mode: str) -> tuple[list[str], list[str]]:
        """Return (usable storage names, warnings). Single-source modes raise if unavailable."""
        available: list[str] = []
        warnings: list[str] = []
        for name in self.storages_for_mode(mode):
            try:
                self.get_store(name)
                available.append(name)
            except StorageUnavailableError as exc:
                if mode != MODE_HYBRID:
                    raise
                warnings.append(f"{STORAGE_LABELS[name]} is unavailable ({exc}); searched the other source only.")
        if not available:
            raise StorageUnavailableError("No knowledge source is available.")
        return available, warnings

    def retriever_for(self, mode: Optional[str] = None) -> Retriever:
        mode = mode or self.default_mode
        if self._retriever_override is not None and mode == self.default_mode:
            return self._retriever_override
        available, _ = self._available_for_mode(mode)
        key = tuple(available)
        with self._lock:
            if key not in self._retrievers:
                view = MultiVectorStore({name: self._stores[name] for name in available})
                self._retrievers[key] = Retriever(view, config=self.config, reranker=self._reranker)
            return self._retrievers[key]

    # --- backward-compatible attributes (default source) --------------------------
    @property
    def chroma_manager(self) -> Any:
        return self.get_store(self.default_mode)

    @property
    def vector_store(self) -> Any:
        return self.chroma_manager

    @property
    def retriever(self) -> Retriever:
        return self.retriever_for(self.default_mode)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def ingest_documents(self, file_paths: List[str]) -> int:
        """Backward-compatible: index into the default source; returns number of chunks added."""
        return self.ingest_with_report(file_paths).chunks_added

    def ingest_with_report(self, file_paths: List[str], targets: Optional[Sequence[str]] = None) -> IngestionReport:
        targets = list(dict.fromkeys(targets or [self.default_mode]))
        for target in targets:
            if target not in STORAGES:
                raise ValueError(f"Upload target must be one of: {', '.join(STORAGES)} (got '{target}').")
        report = IngestionReport()
        with request_context():
            logger.info("Ingestion started for %d file(s) into %s", len(file_paths), ", ".join(targets))
            for result in self.pdf_loader.load_many(file_paths):
                if result.error:
                    for target in targets:
                        report.files.append(
                            FileIngestionStatus(result.source_file, "failed", message=result.error, storage=target)
                        )
                    continue
                chunks = self.chunker.chunk_pages(result.pages)
                for target in targets:
                    report.files.append(self._index_into(result, chunks, target))
            logger.info(
                "Ingestion finished: %d chunks added, %d failed item(s)", report.chunks_added, len(report.failed)
            )
        if any(f.status in ("indexed", "updated") for f in report.files):
            self._invalidate_answers("ingestion")
        return report

    def _index_into(self, result: LoadResult, chunks: list, target: str) -> FileIngestionStatus:
        pages = len(result.pages)
        try:
            store = self.get_store(target)
            if store.has_document(result.document_id):
                return FileIngestionStatus(result.source_file, "unchanged", 0, pages, "Already indexed; skipped.", target)
            if not chunks:
                return FileIngestionStatus(result.source_file, "failed", 0, pages, "No chunks could be created.", target)
            added = store.add_chunks(chunks, sync=False)
            # Remove older versions only after the new version is safely stored.
            replaced = store.delete_document(
                source_file=result.source_file, exclude_document_id=result.document_id, sync=False
            )
            in_sync = store.sync()  # one index sync per document (no-op for local Chroma)
        except StorageUnavailableError as exc:
            return FileIngestionStatus(result.source_file, "failed", 0, pages, f"Not connected: {exc}", target)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Indexing failed for '%s' in %s", result.source_file, target)
            return FileIngestionStatus(
                result.source_file, "failed", 0, pages, f"Indexing failed ({type(exc).__name__}: {exc})", target
            )
        notes = []
        if replaced:
            notes.append(f"Replaced {replaced} chunks from the previous version.")
        if not in_sync:
            notes.append("Search index is still syncing; new content will be searchable shortly.")
        status = "updated" if replaced else "indexed"
        return FileIngestionStatus(result.source_file, status, added, pages, " ".join(notes), target)

    def delete_document(self, document_id: str, storage: Optional[str] = None) -> int:
        removed = self.get_store(storage or self.default_mode).delete_document(document_id=document_id)
        if removed:
            self._invalidate_answers("document_deleted")
        return removed

    def clear_storage(self, storage: str) -> None:
        self.get_store(storage).clear_collection()
        self._invalidate_answers("storage_cleared")

    # ------------------------------------------------------------------
    # Answer cache
    # ------------------------------------------------------------------
    def _invalidate_answers(self, reason: str) -> None:
        self._clear_doc_memo()
        if self.answer_cache is not None:
            self.answer_cache.invalidate(reason)

    def clear_answer_cache(self) -> int:
        self._clear_doc_memo()
        return self.answer_cache.invalidate("manual") if self.answer_cache is not None else 0

    def answer_cache_stats(self) -> dict[str, Any]:
        return self.answer_cache.stats() if self.answer_cache is not None else {"enabled": False}

    def _build_cache_config_fingerprint(self) -> dict[str, Any]:
        c = self.config
        return {
            "model": self._chat_model_label,
            "provider": c.resolved_chat_provider,
            "embedding": getattr(self.embedding_service, "cache_namespace", "custom"),
            "temperature": c.llm_temperature,
            "max_output": c.llm_max_output_tokens,
            "strict": c.strict_citations,
            "ctx": [c.max_context_tokens, c.near_duplicate_threshold],
            "retrieval": [
                c.retrieval_candidate_k, c.hybrid_search_enabled, c.rrf_k, c.vector_weight, c.keyword_weight,
                c.min_similarity, c.min_keyword_coverage, c.reranker, c.reranker_model,
            ],
            "chunking": [c.chunk_size, c.chunk_overlap, c.min_chunk_tokens],
            # answer logic / prompt code: editing these files changes the scope, so old answers are not reused
            "code": [_file_digest("prompts.prompt_template"), _file_digest("rag.grounding"), _file_digest("rag.rag_pipeline")],
        }

    def _cache_scope(self, mode: str, filters: Optional[dict[str, Any]], top_k: Optional[int], available: list[str]) -> str:
        docs = self._documents(mode, strict=True)  # raises -> caller bypasses the cache
        corpus = sorted((d.storage, d.document_id, d.chunk_count, d.page_count) for d in docs)
        return make_scope(
            {
                "cfg": self._cache_config_fp,
                "mode": mode,
                "stores": sorted(available),
                "filters": _canonical_filters(filters),
                "top_k": top_k or self.config.retrieval_top_k,
                "corpus": corpus,
            }
        )


    def _memo_key(self, mode: str) -> tuple:
        return (mode, tuple((n, getattr(self._stores.get(n), "version", None)) for n in self.storages_for_mode(mode)))

    def _clear_doc_memo(self) -> None:
        with self._lock:
            self._doc_list_memo.clear()

    def _collect_documents(self, mode: str, *, strict: bool) -> tuple[list[StoredDocument], bool]:
        documents: list[StoredDocument] = []
        complete = True
        for name in self.storages_for_mode(mode):
            try:
                store = self.get_store(name)
                documents.extend(replace(doc, storage=name) for doc in store.list_documents())
            except StorageUnavailableError:
                if mode != MODE_HYBRID or strict:
                    raise
                complete = False
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not list documents in %s (%s)", name, type(exc).__name__)
                if mode != MODE_HYBRID or strict:
                    raise
                complete = False
        return sorted(documents, key=lambda d: (d.source_file.lower(), d.storage)), complete

    def _documents(self, mode: str, *, strict: bool) -> list[StoredDocument]:
        """Document list, reused for DOC_LIST_CACHE_TTL_SECONDS and until any store is written to.

        The sidebar re-runs this on every Streamlit rerun (twice per question); for Databricks each call is a
        SQL query. Only complete results are memoised.
        """
        ttl = self.config.doc_list_cache_ttl_seconds
        key = self._memo_key(mode)
        if ttl > 0:
            with self._lock:
                hit = self._doc_list_memo.get(key)
            if hit and time.monotonic() - hit[0] < ttl:
                return list(hit[1])
        documents, complete = self._collect_documents(mode, strict=strict)
        if ttl > 0 and complete:
            with self._lock:
                self._doc_list_memo[self._memo_key(mode)] = (time.monotonic(), documents)
        return list(documents)

    def list_documents(self, mode: Optional[str] = None) -> list[StoredDocument]:
        return self._documents(mode or self.default_mode, strict=False)

    # ------------------------------------------------------------------
    # Question answering
    # ------------------------------------------------------------------
    def ask(
        self,
        question: str,
        *,
        filters: Optional[dict[str, Any]] = None,
        top_k: Optional[int] = None,
        mode: Optional[str] = None,
    ) -> RAGResponse:
        started = time.perf_counter()
        mode = mode or self.default_mode
        with request_context() as request_id:
            response = self._ask(question, filters, top_k, mode)
            response.latency_ms = int((time.perf_counter() - started) * 1000)
            response.request_id = request_id
            response.mode = mode
            logger.info(
                "Answered question (mode=%s, len=%d, grounded=%s, no_answer=%s, sources=%d, latency_ms=%d, "
                "cache=%s, timings_ms=%s)",
                mode,
                len(question or ""),
                response.is_grounded,
                response.is_no_answer,
                len(response.sources),
                response.latency_ms,
                response.cache_status or "n/a",
                response.timings_ms,
            )
            return response

    def _ask(
        self, question: str, filters: Optional[dict[str, Any]], top_k: Optional[int], mode: str
    ) -> RAGResponse:
        cleaned = (question or "").strip()
        if not cleaned:
            return RAGResponse(answer="Please enter a question.", sources=[])
        if len(cleaned) > self.config.max_question_chars:
            return RAGResponse(
                answer=f"Your question is too long. Please keep it under {self.config.max_question_chars} characters.",
                sources=[],
            )
        warnings: list[str] = []
        injection_flagged = detect_prompt_injection(cleaned)
        if injection_flagged:
            logger.warning("Possible prompt-injection attempt (question_hash=%s)", sha256_text(cleaned)[:12])
            warnings.append("Parts of your question looked like instructions to the assistant and were ignored.")

        try:
            available, source_warnings = self._available_for_mode(mode)
            retriever = self.retriever_for(mode)
        except StorageUnavailableError as exc:
            return RAGResponse(
                answer=(
                    f"{MODE_LABELS.get(mode, mode)} is not available right now ({exc}). "
                    "Switch the knowledge source in the sidebar or retry the connection."
                ),
                sources=[],
                is_no_answer=True,
                warnings=warnings,
            )
        warnings.extend(source_warnings)
        searched = [STORAGE_LABELS[name] for name in available]

        if not retriever.has_documents():
            return RAGResponse(
                answer=(
                    f"There are no documents in {MODE_LABELS.get(mode, mode)} yet. Upload a PDF, such as your "
                    "Evidence of Coverage or Summary of Benefits and Coverage, using the sidebar, or switch the "
                    "knowledge source."
                ),
                sources=[],
                is_no_answer=True,
                warnings=warnings,
                searched=searched,
            )

        cache = self.answer_cache if not injection_flagged else None
        scope = key = None
        vector = None
        status, similarity = ("disabled" if self.answer_cache is None else "bypass"), None
        if cache is not None:
            if source_warnings:
                cache.record_bypass("partial_sources")
            else:
                try:
                    scope = self._cache_scope(mode, filters, top_k, available)
                except Exception as exc:  # noqa: BLE001
                    cache.record_bypass("fingerprint_error")
                    logger.warning("Answer cache skipped (%s)", type(exc).__name__)
        elif self.answer_cache is not None:
            self.answer_cache.record_bypass("prompt_injection_flag")

        if scope is not None:
            t0 = time.perf_counter()
            embed = self.embedding_service.embed_query
            lookup = cache.lookup(cleaned, scope, embed=embed)
            if lookup.hit:
                response = _payload_to_response(lookup.payload)
                response.warnings = warnings + response.warnings
                response.cache_status, response.cache_similarity = lookup.status, lookup.similarity
                response.timings_ms = {"cache_lookup": int((time.perf_counter() - t0) * 1000)}
                return response
            if lookup.status == "bypass":
                scope = None
            else:
                key, vector, status, similarity = lookup.key, lookup.vector, "miss", lookup.similarity

        if scope is None:
            response, _ = self._retrieve_and_generate(cleaned, filters, top_k, retriever, warnings, searched)
            response.cache_status = status
            return response

        # Single flight: identical concurrent questions wait for one LLM call, then read its cached answer.
        with cache.inflight(key):
            peer = cache.peek_exact(cleaned, scope)
            if peer.hit:
                response = _payload_to_response(peer.payload)
                response.warnings = warnings + response.warnings
                response.cache_status, response.cache_similarity = peer.status, peer.similarity
                return response
            response, clean = self._retrieve_and_generate(cleaned, filters, top_k, retriever, warnings, searched)
            response.cache_status, response.cache_similarity = "miss", similarity
            if clean and response.is_grounded and not response.is_no_answer and not warnings:
                usage = response.usage_metadata or {}
                cache.store(
                    cleaned,
                    scope,
                    _response_to_payload(response),
                    vector=vector,
                    meta={
                        "mode": mode,
                        "filters": _canonical_filters(filters),
                        "top_k": top_k or self.config.retrieval_top_k,
                        "model": usage.get("model", ""),
                        "total_tokens": usage.get("total_tokens", 0),
                        "latency_ms": sum(response.timings_ms.values()),
                        "sources": sorted({c.source_file for c in response.sources}),
                        "searched": response.searched,
                    },
                )
            return response

    def _retrieve_and_generate(
        self,
        cleaned: str,
        filters: Optional[dict[str, Any]],
        top_k: Optional[int],
        retriever: Retriever,
        warnings: list[str],
        searched: list[str],
    ) -> tuple[RAGResponse, bool]:
        """Retrieval -> gate -> context -> LLM -> verification (unchanged logic). Returns (response, no_retrieval_warnings)."""
        timings: dict[str, int] = {}
        t0 = time.perf_counter()
        retrieval = retriever.retrieve_with_diagnostics(cleaned, top_k=top_k, filters=filters)
        timings["retrieval"] = int((time.perf_counter() - t0) * 1000)
        clean = not retrieval.warnings
        warnings.extend(retrieval.warnings)
        if not retrieval.evidence_sufficient:
            logger.info(
                "Evidence below threshold (best_sim=%.3f, best_cov=%.2f); returning no-answer",
                retrieval.best_similarity,
                retrieval.best_keyword_coverage,
            )
            return (
                RAGResponse(
                    answer=NO_ANSWER_MESSAGE, sources=[], is_no_answer=True, warnings=warnings, searched=searched,
                    timings_ms=timings,
                ),
                clean,
            )

        context = build_context(
            retrieval.chunks,
            max_tokens=self.config.max_context_tokens,
            near_duplicate_threshold=self.config.near_duplicate_threshold,
        )
        messages = _messages(SYSTEM_PROMPT, build_user_prompt(cleaned, context))
        t1 = time.perf_counter()
        ai_message = retry_call(
            lambda: self.llm.invoke(messages),
            operation="llm.invoke",
            max_retries=self.config.max_retries,
            base_delay=self.config.retry_base_delay_seconds,
        )
        timings["llm"] = int((time.perf_counter() - t1) * 1000)
        usage = _extract_usage_metadata(ai_message)
        usage["model"] = usage.get("model") or self._chat_model_label
        verified = verify_answer(
            _extract_text(getattr(ai_message, "content", ai_message)), context, strict=self.config.strict_citations
        )
        return RAGResponse(
            answer=verified.answer,
            sources=verified.cited_chunks,
            usage_metadata=usage,
            citations=verified.citations,
            retrieved=context,
            is_grounded=verified.is_grounded,
            is_no_answer=verified.is_no_answer,
            warnings=warnings + verified.warnings,
            searched=searched,
            timings_ms=timings,
        ), clean
