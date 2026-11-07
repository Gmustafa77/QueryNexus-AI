"""RAG orchestration: ingestion (load -> chunk -> embed -> store) and grounded Q&A
(validate -> retrieve -> gate -> build context -> generate -> verify citations).

Public API kept from the original version:
    RAGPipeline().ingest_documents(file_paths) -> int
    RAGPipeline().ask(question) -> RAGResponse(answer, sources, usage_metadata)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

from chunking.text_splitter import DocumentChunker
from embeddings.embedding_service import EmbeddingService
from ingestion.pdf_loader import PDFLoader
from prompts.prompt_template import NO_ANSWER_MESSAGE, SYSTEM_PROMPT, build_user_prompt
from rag.grounding import Citation, build_context, detect_prompt_injection, verify_answer
from retrieval.retriever import Retriever
from utils.config import Config, settings
from utils.logger import get_logger, request_context
from utils.retry import retry_call
from utils.text import sha256_text
from vectorstore.chroma_manager import ChromaManager

logger = get_logger(__name__)


# ----------------------------------------------------------------------------
# Provider response helpers (unchanged behaviour)
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


@dataclass
class FileIngestionStatus:
    source_file: str
    status: str  # indexed | updated | unchanged | failed
    chunks_added: int = 0
    pages: int = 0
    message: str = ""


@dataclass
class IngestionReport:
    files: list[FileIngestionStatus] = field(default_factory=list)

    @property
    def chunks_added(self) -> int:
        return sum(f.chunks_added for f in self.files)

    @property
    def failed(self) -> list[FileIngestionStatus]:
        return [f for f in self.files if f.status == "failed"]


# ----------------------------------------------------------------------------
def create_chat_model(config: Config) -> tuple[Any, str]:
    """Create the chat model for the configured provider (single place for provider wiring)."""
    config.validate_chat()
    provider = config.resolved_chat_provider
    common = {"temperature": config.llm_temperature, "max_retries": 0}
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


class RAGPipeline:
    """Main pipeline used by the Streamlit UI and the evaluation harness."""

    def __init__(
        self,
        *,
        config: Config | None = None,
        embedding_service: EmbeddingService | None = None,
        chroma_manager: ChromaManager | None = None,
        llm: Any | None = None,
        retriever: Retriever | None = None,
    ) -> None:
        self.config = config or settings
        self.config.validate_retrieval()
        self.pdf_loader = PDFLoader(max_bytes=self.config.max_upload_bytes, max_pages=self.config.max_pdf_pages)
        self.chunker = DocumentChunker(
            self.config.chunk_size, self.config.chunk_overlap, min_chunk_tokens=self.config.min_chunk_tokens
        )
        self.embedding_service = embedding_service or (
            chroma_manager.embedding_service if chroma_manager else EmbeddingService(config=self.config)
        )
        self.chroma_manager = chroma_manager or ChromaManager(self.embedding_service, config=self.config)
        self.retriever = retriever or Retriever(self.chroma_manager, config=self.config)
        if llm is not None:
            self.llm, self._chat_model_label = llm, getattr(llm, "model_name", "custom")
        else:
            self.llm, self._chat_model_label = create_chat_model(self.config)
        logger.info("RAGPipeline ready (chat_model=%s)", self._chat_model_label)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def ingest_documents(self, file_paths: List[str]) -> int:
        """Backward-compatible: returns number of chunks added."""
        return self.ingest_with_report(file_paths).chunks_added

    def ingest_with_report(self, file_paths: List[str]) -> IngestionReport:
        report = IngestionReport()
        with request_context():
            logger.info("Ingestion started for %d file(s)", len(file_paths))
            for result in self.pdf_loader.load_many(file_paths):
                if result.error:
                    report.files.append(FileIngestionStatus(result.source_file, "failed", message=result.error))
                    continue
                try:
                    report.files.append(self._index_document(result))
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Indexing failed for '%s'", result.source_file)
                    report.files.append(
                        FileIngestionStatus(
                            result.source_file, "failed", message=f"Indexing failed ({type(exc).__name__}: {exc})"
                        )
                    )
            logger.info(
                "Ingestion finished: %d chunks added, %d failed file(s)", report.chunks_added, len(report.failed)
            )
        return report

    def _index_document(self, result) -> FileIngestionStatus:
        pages = len(result.pages)
        if self.chroma_manager.has_document(result.document_id):
            return FileIngestionStatus(result.source_file, "unchanged", 0, pages, "Already indexed; skipped.")
        chunks = self.chunker.chunk_pages(result.pages)
        if not chunks:
            return FileIngestionStatus(result.source_file, "failed", 0, pages, "No chunks could be created.")
        added = self.chroma_manager.add_chunks(chunks)
        # Remove older versions only after the new version is safely stored.
        replaced = self.chroma_manager.delete_document(
            source_file=result.source_file, exclude_document_id=result.document_id
        )
        status = "updated" if replaced else "indexed"
        message = f"Replaced {replaced} chunks from the previous version." if replaced else ""
        return FileIngestionStatus(result.source_file, status, added, pages, message)

    def delete_document(self, document_id: str) -> int:
        return self.chroma_manager.delete_document(document_id=document_id)

    def list_documents(self):
        return self.chroma_manager.list_documents()

    # ------------------------------------------------------------------
    # Question answering
    # ------------------------------------------------------------------
    def ask(
        self, question: str, *, filters: Optional[dict[str, Any]] = None, top_k: Optional[int] = None
    ) -> RAGResponse:
        started = time.perf_counter()
        with request_context() as request_id:
            response = self._ask(question, filters, top_k)
            response.latency_ms = int((time.perf_counter() - started) * 1000)
            response.request_id = request_id
            logger.info(
                "Answered question (len=%d, grounded=%s, no_answer=%s, sources=%d, latency_ms=%d)",
                len(question or ""),
                response.is_grounded,
                response.is_no_answer,
                len(response.sources),
                response.latency_ms,
            )
            return response

    def _ask(self, question: str, filters: Optional[dict[str, Any]], top_k: Optional[int]) -> RAGResponse:
        cleaned = (question or "").strip()
        if not cleaned:
            return RAGResponse(answer="Please enter a question.", sources=[])
        if len(cleaned) > self.config.max_question_chars:
            return RAGResponse(
                answer=f"Your question is too long. Please keep it under {self.config.max_question_chars} characters.",
                sources=[],
            )
        warnings: list[str] = []
        if detect_prompt_injection(cleaned):
            logger.warning("Possible prompt-injection attempt (question_hash=%s)", sha256_text(cleaned)[:12])
            warnings.append("Parts of your question looked like instructions to the assistant and were ignored.")

        if not self.retriever.has_documents():
            return RAGResponse(
                answer=(
                    "I don't have any insurance documents to search yet. Please upload a PDF, such as your "
                    "Evidence of Coverage or Summary of Benefits and Coverage, using the sidebar first."
                ),
                sources=[],
                is_no_answer=True,
            )

        retrieval = self.retriever.retrieve_with_diagnostics(cleaned, top_k=top_k, filters=filters)
        if not retrieval.evidence_sufficient:
            logger.info(
                "Evidence below threshold (best_sim=%.3f, best_cov=%.2f); returning no-answer",
                retrieval.best_similarity,
                retrieval.best_keyword_coverage,
            )
            return RAGResponse(answer=NO_ANSWER_MESSAGE, sources=[], is_no_answer=True, warnings=warnings)

        context = build_context(
            retrieval.chunks,
            max_tokens=self.config.max_context_tokens,
            near_duplicate_threshold=self.config.near_duplicate_threshold,
        )
        messages = _messages(SYSTEM_PROMPT, build_user_prompt(cleaned, context))
        ai_message = retry_call(
            lambda: self.llm.invoke(messages),
            operation="llm.invoke",
            max_retries=self.config.max_retries,
            base_delay=self.config.retry_base_delay_seconds,
        )
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
        )
