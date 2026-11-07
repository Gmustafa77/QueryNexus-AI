"""STEP 6: Hybrid retrieval (vector + BM25) with RRF fusion, filters, thresholds and reranking.

Flow:
  question -> embed -> vector top-N  ┐
  question -> BM25 top-N             ┴-> weighted Reciprocal Rank Fusion
  -> optional cross-encoder rerank -> relevance gate -> top-K RetrievedChunk

Works with a single store or a ``MultiVectorStore`` (Local + Databricks).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, List, Optional

from retrieval.bm25 import BM25Index, keyword_coverage
from retrieval.reranker import CrossEncoderReranker, build_reranker
from utils.config import Config, settings
from utils.logger import get_logger
from vectorstore.chroma_manager import matches_filters

logger = get_logger(__name__)

STORAGE_LABELS = {"local": "Local docs", "databricks": "Databricks"}


@dataclass
class RetrievedChunk:
    """One retrieved chunk, ready for the prompt and for UI citations.

    ``similarity_score`` is cosine similarity (higher = more relevant).
    ``storage`` is where the chunk lives ("local" or "databricks").
    """

    text: str
    source_file: str
    page_number: int
    similarity_score: float
    chunk_id: str = ""
    section_title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    keyword_score: float = 0.0
    keyword_coverage: float = 0.0
    rerank_score: Optional[float] = None
    fused_score: float = 0.0
    citation_label: str = ""
    storage: str = ""


@dataclass
class RetrievalResult:
    chunks: List[RetrievedChunk]
    evidence_sufficient: bool
    best_similarity: float
    best_keyword_coverage: float
    warnings: List[str] = field(default_factory=list)


class Retriever:
    """Purpose-built API: given a question, return the most relevant evidence."""

    def __init__(
        self,
        chroma_manager: Any,
        *,
        config: Config | None = None,
        reranker: Optional[CrossEncoderReranker] = None,
    ) -> None:
        self.chroma_manager = chroma_manager
        self.config = config or settings
        self.reranker = reranker if reranker is not None else build_reranker(
            self.config.reranker, self.config.reranker_model
        )
        self._bm25: Optional[BM25Index] = None
        self._bm25_key: tuple[int, int] = (-1, -1)
        self._bm25_lock = threading.Lock()

    # ------------------------------------------------------------------
    def retrieve(
        self, question: str, top_k: int = None, filters: Optional[dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        """Backward-compatible API returning only relevant chunks."""
        return self.retrieve_with_diagnostics(question, top_k=top_k, filters=filters).chunks

    def _vector_search(
        self, vector: list[float], k: int, filters: Optional[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[str]]:
        store = self.chroma_manager
        if hasattr(store, "query_by_vector_with_errors"):
            rows, errors = store.query_by_vector_with_errors(vector, k, filters)
            warnings = [
                f"{STORAGE_LABELS.get(name, name)} could not be searched for this question "
                f"({message.split(':')[0]}); results come from the other source only."
                for name, message in errors.items()
            ]
            return rows, warnings
        return store.query_by_vector(vector, k, filters), []

    def retrieve_with_diagnostics(
        self, question: str, top_k: int = None, filters: Optional[dict[str, Any]] = None
    ) -> RetrievalResult:
        k = top_k or self.config.retrieval_top_k
        candidate_k = max(self.config.retrieval_candidate_k, k)
        query_vector = self.chroma_manager.embedding_service.embed_query(question)
        vector_rows, warnings = self._vector_search(query_vector, candidate_k, filters)

        candidates: dict[str, dict[str, Any]] = {}
        fused: dict[str, float] = {}
        for rank, row in enumerate(vector_rows):
            candidates[row["id"]] = {**row, "keyword_score": 0.0}
            fused[row["id"]] = self.config.vector_weight / (self.config.rrf_k + rank + 1)

        if self.config.hybrid_search_enabled:
            index = self._keyword_index()
            for rank, (row_index, score) in enumerate(
                index.search(question, candidate_k, predicate=lambda m: matches_filters(m, filters))
            ):
                chunk_id = index.ids[row_index]
                entry = candidates.setdefault(
                    chunk_id,
                    {
                        "id": chunk_id,
                        "text": index.texts[row_index],
                        "metadata": index.metadatas[row_index],
                        "similarity": None,
                    },
                )
                entry["keyword_score"] = score
                fused[chunk_id] = fused.get(chunk_id, 0.0) + self.config.keyword_weight / (
                    self.config.rrf_k + rank + 1
                )

        ordered = sorted(candidates.values(), key=lambda c: fused[c["id"]], reverse=True)
        chunks = [self._to_chunk(c, question, fused[c["id"]]) for c in ordered]
        relevant = [c for c in chunks if self._passes_threshold(c)]

        if self.reranker and len(relevant) > 1:
            scores = self.reranker.score(question, [c.text for c in relevant])
            if scores:
                for chunk, score in zip(relevant, scores):
                    chunk.rerank_score = score
                relevant.sort(key=lambda c: c.rerank_score, reverse=True)

        final = relevant[:k]
        best_sim = max((c.similarity_score for c in chunks), default=0.0)
        best_cov = max((c.keyword_coverage for c in chunks), default=0.0)
        logger.info(
            "Retrieved %d/%d candidates (vector=%d, best_sim=%.3f, best_cov=%.2f, filters=%s, store=%s)",
            len(final),
            len(chunks),
            len(vector_rows),
            best_sim,
            best_cov,
            bool(filters),
            getattr(self.chroma_manager, "backend_name", "custom"),
        )
        return RetrievalResult(final, bool(final), best_sim, best_cov, warnings)

    def has_documents(self) -> bool:
        return self.chroma_manager.document_count() > 0

    # ------------------------------------------------------------------
    def _passes_threshold(self, chunk: RetrievedChunk) -> bool:
        return (
            chunk.similarity_score >= self.config.min_similarity
            or chunk.keyword_coverage >= self.config.min_keyword_coverage
        )

    def _to_chunk(self, row: dict[str, Any], question: str, fused_score: float) -> RetrievedChunk:
        meta = dict(row.get("metadata") or {})
        text = row.get("text") or ""
        similarity = row.get("similarity")
        return RetrievedChunk(
            text=text,
            source_file=str(meta.get("source_file") or meta.get("source") or "unknown"),
            page_number=int(meta.get("page_number") or meta.get("page") or 0),
            similarity_score=float(similarity) if similarity is not None else 0.0,
            chunk_id=str(row.get("id") or meta.get("chunk_id") or ""),
            section_title=str(meta.get("section_title") or ""),
            metadata=meta,
            keyword_score=float(row.get("keyword_score") or 0.0),
            keyword_coverage=keyword_coverage(question, text),
            fused_score=fused_score,
            storage=str(meta.get("storage") or ""),
        )

    def _keyword_index(self) -> BM25Index:
        key = (self.chroma_manager.version, self.chroma_manager.document_count())
        with self._bm25_lock:
            if self._bm25 is None or key != self._bm25_key:
                ids, texts, metas = self.chroma_manager.get_all()
                self._bm25 = BM25Index(ids=ids, texts=texts, metadatas=metas)
                self._bm25_key = key
                logger.info("Keyword index rebuilt (%d chunks)", len(ids))
            return self._bm25
