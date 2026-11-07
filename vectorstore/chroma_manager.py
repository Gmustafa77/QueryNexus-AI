"""STEP 5: Persistent ChromaDB storage with deterministic IDs, dedup and document updates.

Uses the native ``chromadb`` client with precomputed (cached) embeddings, so we
control batching, retries and IDs. Collections are created with cosine
distance. Existing L2 collections created by the earlier version keep working:
distances are converted to cosine similarity (vectors are L2-normalised).

The public methods here define the vector-store contract that
``vectorstore/databricks_store.py`` also implements.
"""
from __future__ import annotations

import os
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional

from chunking.text_splitter import Chunk
from embeddings.embedding_service import EmbeddingService
from utils.config import Config, settings
from utils.logger import get_logger

logger = get_logger(__name__)

_UPSERT_BATCH = 256


@dataclass
class StoredDocument:
    document_id: str
    source_file: str
    chunk_count: int
    page_count: int
    storage: str = ""  # "local" | "databricks"; filled in by the pipeline


def build_where(filters: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Convert ``{"source_file": "a.pdf", "page_number": [1, 2]}`` to a Chroma ``where`` clause."""
    if not filters:
        return None
    clauses = []
    for key, value in filters.items():
        if value is None or value == [] or value == "":
            continue
        if isinstance(value, dict):
            clauses.append({key: value})
        elif isinstance(value, (list, tuple, set)):
            clauses.append({key: {"$in": list(value)}})
        else:
            clauses.append({key: {"$eq": value}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def matches_filters(metadata: dict[str, Any], filters: Optional[dict[str, Any]]) -> bool:
    """Python-side equivalent of ``build_where`` used by keyword search."""
    if not filters:
        return True
    for key, value in filters.items():
        if value is None or value == [] or value == "":
            continue
        actual = metadata.get(key)
        if isinstance(value, dict):
            if "$in" in value and actual not in value["$in"]:
                return False
            if "$eq" in value and actual != value["$eq"]:
                return False
            if "$ne" in value and actual == value["$ne"]:
                return False
        elif isinstance(value, (list, tuple, set)):
            if actual not in value:
                return False
        elif actual != value:
            return False
    return True


class ChromaManager:
    """Manages one persistent collection: upsert, query, list, delete."""

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        *,
        collection: Any | None = None,
        config: Config | None = None,
    ) -> None:
        self.config = config or settings
        self.embedding_service = embedding_service or EmbeddingService(config=self.config)
        self._lock = threading.RLock()
        self._version = 0
        if collection is not None:
            self._client = None
            self._collection = collection
        else:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            os.makedirs(self.config.chroma_persist_dir, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=self.config.chroma_persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False, allow_reset=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=self.config.chroma_collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        self.space = (getattr(self._collection, "metadata", None) or {}).get("hnsw:space", "l2")
        if self.space != "cosine":
            logger.warning(
                "Collection uses '%s' distance; scores are converted to cosine similarity. "
                "Re-index into a new collection for best results.",
                self.space,
            )
        logger.info(
            "ChromaManager ready (collection=%s, persist_dir=%s, chunks=%d)",
            self.config.chroma_collection_name,
            self.config.chroma_persist_dir,
            self.document_count(),
        )

    # ------------------------------------------------------------------
    @property
    def backend_name(self) -> str:
        return "chroma"

    @property
    def version(self) -> int:
        """Incremented on every write; lets the keyword index know when to rebuild."""
        return self._version

    def distance_to_similarity(self, distance: float) -> float:
        if self.space == "l2":
            return 1.0 - float(distance) / 2.0
        return 1.0 - float(distance)

    def sync(self) -> bool:
        """Chroma writes are visible immediately; kept for parity with the Databricks backend."""
        return True

    # ------------------------------------------------------------------
    def add_chunks(self, chunks: List[Chunk], *, sync: bool = True) -> int:
        """Embed and upsert chunks. Already-stored IDs are skipped (idempotent re-ingestion)."""
        if not chunks:
            logger.warning("add_chunks() called with no chunks")
            return 0
        unique: dict[str, Chunk] = {}
        for chunk in chunks:
            unique.setdefault(chunk.chunk_id, chunk)
        ids = list(unique)
        with self._lock:
            existing = set(self._existing_ids(ids))
            new_chunks = [unique[i] for i in ids if i not in existing]
            if not new_chunks:
                logger.info("All %d chunks already indexed; nothing to add", len(ids))
                return 0
            vectors = self.embedding_service.embed_documents([c.text for c in new_chunks])
            for start in range(0, len(new_chunks), _UPSERT_BATCH):
                batch = new_chunks[start : start + _UPSERT_BATCH]
                self._collection.upsert(
                    ids=[c.chunk_id for c in batch],
                    documents=[c.text for c in batch],
                    metadatas=[c.metadata for c in batch],
                    embeddings=vectors[start : start + len(batch)],
                )
            self._version += 1
        logger.info("Stored %d new chunks (%d already present)", len(new_chunks), len(existing))
        return len(new_chunks)

    def _existing_ids(self, ids: Iterable[str]) -> list[str]:
        ids = list(ids)
        found: list[str] = []
        for start in range(0, len(ids), 1000):
            found.extend(self._collection.get(ids=ids[start : start + 1000], include=[])["ids"])
        return found

    # ------------------------------------------------------------------
    def has_document(self, document_id: str) -> bool:
        return bool(self._collection.get(where={"document_id": {"$eq": document_id}}, limit=1, include=[])["ids"])

    def delete_document(
        self,
        *,
        document_id: str | None = None,
        source_file: str | None = None,
        exclude_document_id: str | None = None,
        sync: bool = True,
    ) -> int:
        """Delete chunks by document hash, or all versions of a filename (optionally keeping one version)."""
        if not document_id and not source_file:
            raise ValueError("document_id or source_file is required.")
        if document_id:
            where = build_where({"document_id": document_id})
        else:
            filters: dict[str, Any] = {"source_file": source_file}
            if exclude_document_id:
                filters["document_id"] = {"$ne": exclude_document_id}
            where = build_where(filters)
        with self._lock:
            ids = self._collection.get(where=where, include=[])["ids"]
            if ids:
                self._collection.delete(ids=ids)
                self._version += 1
        logger.info("Deleted %d chunks (document_id=%s, source_file=%s)", len(ids), document_id, source_file)
        return len(ids)

    def list_documents(self) -> list[StoredDocument]:
        data = self._collection.get(include=["metadatas"])
        grouped: dict[str, dict[str, Any]] = defaultdict(lambda: {"chunks": 0, "pages": set(), "source": ""})
        for meta in data.get("metadatas") or []:
            meta = meta or {}
            doc_id = str(meta.get("document_id") or meta.get("source_file") or meta.get("source") or "unknown")
            entry = grouped[doc_id]
            entry["chunks"] += 1
            entry["pages"].add(meta.get("page_number", meta.get("page")))
            entry["source"] = str(meta.get("source_file") or meta.get("source") or "unknown")
        docs = [
            StoredDocument(doc_id, v["source"], v["chunks"], len(v["pages"])) for doc_id, v in grouped.items()
        ]
        return sorted(docs, key=lambda d: d.source_file.lower())

    def get_all(self, where: Optional[dict[str, Any]] = None) -> tuple[list[str], list[str], list[dict]]:
        data = self._collection.get(where=where, include=["documents", "metadatas"])
        return data.get("ids") or [], data.get("documents") or [], [m or {} for m in data.get("metadatas") or []]

    # ------------------------------------------------------------------
    def query_by_vector(
        self, vector: list[float], top_k: int, filters: Optional[dict[str, Any]] = None
    ) -> list[dict[str, Any]]:
        """Return ``[{id, text, metadata, similarity}]`` sorted by similarity (desc)."""
        count = self.document_count()
        if count == 0:
            return []
        result = self._collection.query(
            query_embeddings=[vector],
            n_results=min(top_k, count),
            where=build_where(filters),
            include=["documents", "metadatas", "distances"],
        )
        rows = []
        for chunk_id, text, meta, dist in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            rows.append(
                {"id": chunk_id, "text": text or "", "metadata": meta or {}, "similarity": self.distance_to_similarity(dist)}
            )
        return rows

    def similarity_search_with_scores(self, query: str, top_k: int = None):
        """Backward-compatible API: list of (Document, distance) where lower = more similar."""
        from langchain_core.documents import Document

        k = top_k or self.config.retrieval_top_k
        rows = self.query_by_vector(self.embedding_service.embed_query(query), k)
        return [(Document(page_content=r["text"], metadata=r["metadata"]), 1.0 - r["similarity"]) for r in rows]

    def get_retriever(self, top_k: int = None):
        """Backward-compatible LangChain retriever over the same persistent collection."""
        from langchain_chroma import Chroma

        if self._client is None:
            raise RuntimeError("get_retriever() requires a persistent Chroma client.")
        store = Chroma(
            client=self._client,
            collection_name=self.config.chroma_collection_name,
            embedding_function=self.embedding_service.get_langchain_embeddings(),
        )
        return store.as_retriever(search_kwargs={"k": top_k or self.config.retrieval_top_k})

    def document_count(self) -> int:
        return int(self._collection.count())

    def clear_collection(self) -> None:
        logger.warning("Clearing all vectors from collection '%s'", self.config.chroma_collection_name)
        with self._lock:
            while True:
                ids = self._collection.get(limit=5000, include=[])["ids"]
                if not ids:
                    break
                self._collection.delete(ids=ids)
            self._version += 1
        logger.info("Collection cleared")
