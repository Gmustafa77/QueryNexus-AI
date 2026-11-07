"""Databricks vector-store backend.

Design
------
* **Delta table** ``<catalog>.<schema>.<DATABRICKS_CHUNK_TABLE>`` is the source of
  truth: chunk text, citation metadata and the embedding vector. Written via a
  SQL warehouse (``databricks-sql-connector``) with parameterised statements.
* **Databricks Vector Search Delta Sync index** (self-managed embeddings) on that
  table, served by your existing Vector Search endpoint. Writes trigger a sync and
  wait (bounded) until the index reflects the table.
* Embeddings are produced by the app's EmbeddingService (same model for ingest and
  query, same cache), so retrieval quality does not change when switching backends.
* Cosine similarity is computed locally from returned vectors, so MIN_SIMILARITY
  means exactly the same thing as on the Chroma backend.
* Implements the same contract as ``ChromaManager`` so the retriever, pipeline and
  UI work unchanged.
"""
from __future__ import annotations

import json
import math
import re
import threading
import time
from typing import Any, Callable, List, Optional, Sequence
from urllib.parse import urlparse

from chunking.text_splitter import Chunk
from embeddings.embedding_service import EmbeddingService
from utils.config import Config, settings
from utils.logger import get_logger
from utils.retry import retry_call
from vectorstore.chroma_manager import StoredDocument

logger = get_logger(__name__)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_FILTERABLE_COLUMNS = frozenset({"document_id", "source_file", "page_number", "section_title", "content_hash"})
_RESULT_COLUMNS = ["chunk_id", "text", "metadata_json", "embedding"]
_INSERT_BATCH = 20
_ID_BATCH = 200
_COUNT_TTL_SECONDS = 15.0
_BUSY_STATES = ("PROVISIONING", "TRIGGERED_UPDATE", "UPDATING", "PIPELINE_RUNNING", "INITIAL_SNAPSHOT")


def quote_identifier(name: str) -> str:
    """Validate and back-quote a Unity Catalog identifier (identifiers cannot be bound as parameters)."""
    if not _IDENTIFIER_RE.match(name or ""):
        raise ValueError(f"Invalid Databricks identifier: {name!r}")
    return f"`{name}`"


def _is_not_found(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(m in message for m in ("does not exist", "not found", "resource_does_not_exist", "404"))


def to_vector_search_filters(filters: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Convert the app's filter dict to Databricks Vector Search (standard endpoint) syntax."""
    if not filters:
        return None
    converted: dict[str, Any] = {}
    for key, value in filters.items():
        if value is None or value == [] or value == "":
            continue
        if key not in _FILTERABLE_COLUMNS:
            raise ValueError(
                f"Filtering on '{key}' is not supported by the Databricks backend. "
                f"Supported: {', '.join(sorted(_FILTERABLE_COLUMNS))}."
            )
        if isinstance(value, dict):
            if "$in" in value:
                converted[key] = list(value["$in"])
            if "$eq" in value:
                converted[key] = value["$eq"]
            if "$ne" in value:
                converted[f"{key} NOT"] = value["$ne"]
        elif isinstance(value, (list, tuple, set)):
            converted[key] = list(value)
        else:
            converted[key] = value
    return converted or None


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# ----------------------------------------------------------------------------
# Delta table access (SQL warehouse)
# ----------------------------------------------------------------------------
class DeltaChunkTable:
    """All SQL for the chunk table. Values are always bound as named parameters."""

    def __init__(self, config: Config, connection_factory: Callable[[], Any] | None = None) -> None:
        self.config = config
        self.fqn = ".".join(
            quote_identifier(part)
            for part in (config.databricks_catalog, config.databricks_schema, config.databricks_chunk_table)
        )
        self._connection_factory = connection_factory or self._default_connection
        self._connection: Any | None = None
        self._lock = threading.RLock()

    def _default_connection(self) -> Any:
        from databricks import sql

        hostname = urlparse(self.config.databricks_host).hostname or self.config.databricks_host
        return sql.connect(
            server_hostname=hostname,
            http_path=self.config.databricks_http_path,
            access_token=self.config.databricks_token,
        )

    def _reset_connection(self) -> None:
        if self._connection is not None:
            try:
                self._connection.close()
            except Exception:  # noqa: BLE001
                pass
        self._connection = None

    def execute(self, query: str, params: Optional[dict[str, Any]] = None, *, fetch: bool = True) -> list[Any]:
        def _run() -> list[Any]:
            with self._lock:
                try:
                    if self._connection is None:
                        self._connection = self._connection_factory()
                    with self._connection.cursor() as cursor:
                        cursor.execute(query, params or {})
                        return list(cursor.fetchall()) if fetch else []
                except Exception:
                    self._reset_connection()
                    raise

        return retry_call(
            _run,
            operation="databricks.sql",
            max_retries=self.config.max_retries,
            base_delay=self.config.retry_base_delay_seconds,
        )

    def close(self) -> None:
        with self._lock:
            self._reset_connection()

    # --- schema ------------------------------------------------------------------
    def ensure_table(self) -> None:
        self.execute(
            f"""CREATE TABLE IF NOT EXISTS {self.fqn} (
                chunk_id STRING NOT NULL,
                text STRING,
                document_id STRING,
                source_file STRING,
                page_number INT,
                section_title STRING,
                content_hash STRING,
                metadata_json STRING,
                embedding ARRAY<FLOAT>,
                ingested_at TIMESTAMP
            ) TBLPROPERTIES (delta.enableChangeDataFeed = true)""",
            fetch=False,
        )
        try:
            # Delta Sync indexes require Change Data Feed; enable it on pre-existing tables too.
            self.execute(f"ALTER TABLE {self.fqn} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)", fetch=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not confirm Change Data Feed on %s (%s)", self.fqn, type(exc).__name__)

    # --- reads -------------------------------------------------------------------
    @staticmethod
    def _in_clause(prefix: str, values: Sequence[Any]) -> tuple[str, dict[str, Any]]:
        params = {f"{prefix}{i}": value for i, value in enumerate(values)}
        return ", ".join(f":{name}" for name in params), params

    def existing_ids(self, ids: Sequence[str]) -> set[str]:
        found: set[str] = set()
        for start in range(0, len(ids), _ID_BATCH):
            placeholders, params = self._in_clause("id", ids[start : start + _ID_BATCH])
            rows = self.execute(f"SELECT chunk_id FROM {self.fqn} WHERE chunk_id IN ({placeholders})", params)
            found.update(str(row[0]) for row in rows)
        return found

    def count(self) -> int:
        rows = self.execute(f"SELECT COUNT(*) FROM {self.fqn}")
        return int(rows[0][0]) if rows else 0

    def has_document(self, document_id: str) -> bool:
        rows = self.execute(f"SELECT 1 FROM {self.fqn} WHERE document_id = :doc LIMIT 1", {"doc": document_id})
        return bool(rows)

    def select_all(self) -> list[tuple[str, str, str]]:
        rows = self.execute(f"SELECT chunk_id, text, metadata_json FROM {self.fqn}")
        return [(str(r[0]), str(r[1] or ""), str(r[2] or "{}")) for r in rows]

    def list_documents(self) -> list[tuple[str, str, int, int]]:
        rows = self.execute(
            f"""SELECT document_id, MAX(source_file), COUNT(*), COUNT(DISTINCT page_number)
                FROM {self.fqn} GROUP BY document_id"""
        )
        return [(str(r[0]), str(r[1] or "unknown"), int(r[2]), int(r[3])) for r in rows]

    # --- writes ------------------------------------------------------------------
    def insert(self, rows: Sequence[dict[str, Any]]) -> None:
        for start in range(0, len(rows), _INSERT_BATCH):
            batch = rows[start : start + _INSERT_BATCH]
            values_sql: list[str] = []
            params: dict[str, Any] = {}
            for i, row in enumerate(batch):
                names = {col: f"{col}_{i}" for col in row}
                params.update({names[col]: value for col, value in row.items()})
                values_sql.append(
                    f"(:{names['chunk_id']}, :{names['text']}, :{names['document_id']}, :{names['source_file']}, "
                    f":{names['page_number']}, :{names['section_title']}, :{names['content_hash']}, "
                    f":{names['metadata_json']}, from_json(:{names['embedding_json']}, 'array<float>'), "
                    "current_timestamp())"
                )
            self.execute(
                f"""INSERT INTO {self.fqn}
                    (chunk_id, text, document_id, source_file, page_number, section_title,
                     content_hash, metadata_json, embedding, ingested_at)
                    VALUES {", ".join(values_sql)}""",
                params,
                fetch=False,
            )

    def delete(self, *, document_id: str | None, source_file: str | None, exclude_document_id: str | None) -> int:
        if document_id:
            where, params = "document_id = :doc", {"doc": document_id}
        elif source_file:
            where, params = "source_file = :src", {"src": source_file}
            if exclude_document_id:
                where += " AND document_id <> :keep"
                params["keep"] = exclude_document_id
        else:
            where, params = "1 = 1", {}
        rows = self.execute(f"SELECT COUNT(*) FROM {self.fqn} WHERE {where}", params)
        affected = int(rows[0][0]) if rows else 0
        if affected:
            self.execute(f"DELETE FROM {self.fqn} WHERE {where}", params, fetch=False)
        return affected


# ----------------------------------------------------------------------------
# Vector store
# ----------------------------------------------------------------------------
class DatabricksVectorStore:
    """Same public contract as ``ChromaManager``, backed by Delta + Databricks Vector Search."""

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        *,
        config: Config | None = None,
        table: DeltaChunkTable | None = None,
        vector_search_client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config or settings
        if table is None or vector_search_client is None:
            self.config.validate_databricks()
        self.embedding_service = embedding_service or EmbeddingService(config=self.config)
        self.table = table or DeltaChunkTable(self.config)
        self.table.ensure_table()
        self._vs_client = vector_search_client or self._default_vs_client()
        self.endpoint_name = self.config.databricks_vs_endpoint
        self.index_name = self.config.databricks_index_fqn
        self.source_table_name = self.config.databricks_table_fqn
        self._sleep = sleep
        self._index: Any | None = None
        self._lock = threading.RLock()
        self._version = 0
        self._count_cache: tuple[int, float] | None = None
        self.space = "cosine"
        logger.info(
            "DatabricksVectorStore ready (table=%s, index=%s, endpoint=%s, chunks=%d)",
            self.source_table_name,
            self.index_name,
            self.endpoint_name,
            self.document_count(),
        )

    def _default_vs_client(self) -> Any:
        from databricks.vector_search.client import VectorSearchClient

        return VectorSearchClient(
            workspace_url=self.config.databricks_host,
            personal_access_token=self.config.databricks_token,
            disable_notice=True,
        )

    # ------------------------------------------------------------------
    @property
    def backend_name(self) -> str:
        return "databricks"

    @property
    def version(self) -> int:
        return self._version

    def _invalidate(self) -> None:
        self._version += 1
        self._count_cache = None

    def document_count(self) -> int:
        cached = self._count_cache
        if cached and time.monotonic() - cached[1] < _COUNT_TTL_SECONDS:
            return cached[0]
        count = self.table.count()
        self._count_cache = (count, time.monotonic())
        return count

    # --- index management -------------------------------------------------------
    def _get_index(self, *, create_with_dimension: int | None = None) -> Any | None:
        if self._index is not None:
            return self._index
        try:
            self._index = self._vs_client.get_index(endpoint_name=self.endpoint_name, index_name=self.index_name)
        except Exception as exc:  # noqa: BLE001
            if not _is_not_found(exc):
                raise
            if create_with_dimension is None:
                return None
            logger.info("Creating Vector Search index %s (dim=%d)", self.index_name, create_with_dimension)
            self._index = self._vs_client.create_delta_sync_index(
                endpoint_name=self.endpoint_name,
                index_name=self.index_name,
                source_table_name=self.source_table_name,
                pipeline_type="TRIGGERED",
                primary_key="chunk_id",
                embedding_dimension=create_with_dimension,
                embedding_vector_column="embedding",
            )
        return self._index

    def _check_dimension(self, index: Any, dimension: int) -> None:
        try:
            spec = (index.describe() or {}).get("delta_sync_index_spec") or {}
            columns = spec.get("embedding_vector_columns") or []
            expected = int(columns[0].get("embedding_dimension")) if columns else None
        except Exception:  # noqa: BLE001
            return
        if expected and expected != dimension:
            raise RuntimeError(
                f"Embedding dimension {dimension} does not match index {self.index_name} ({expected}). "
                "Use a new DATABRICKS_CHUNK_TABLE and DATABRICKS_VECTOR_INDEX when changing embedding models."
            )

    def sync(self) -> bool:
        """Trigger an index sync and wait (bounded) until it reflects the Delta table."""
        index = self._get_index()
        if index is None:
            return True
        expected = self.table.count()
        try:
            index.sync()
        except Exception as exc:  # noqa: BLE001
            message = str(exc).lower()
            if not any(m in message for m in ("not ready", "provision", "already", "in progress")):
                raise
            logger.info("Index sync not triggered now (%s); waiting for current update", type(exc).__name__)
        deadline = time.monotonic() + self.config.databricks_sync_timeout_seconds
        poll = max(self.config.databricks_sync_poll_seconds, 0.0)
        self._sleep(poll)
        while True:
            status = (index.describe() or {}).get("status") or {}
            state = str(status.get("detailed_state", ""))
            indexed = status.get("indexed_row_count")
            if "FAILED" in state:
                raise RuntimeError(f"Vector Search index {self.index_name} is in state {state}: {status.get('message')}")
            busy = any(marker in state for marker in _BUSY_STATES)
            if status.get("ready") and not busy and indexed is not None and int(indexed) == expected:
                logger.info("Vector Search index in sync (%d rows)", expected)
                return True
            if time.monotonic() >= deadline:
                logger.warning(
                    "Index sync still running after %ss (state=%s, indexed=%s, expected=%d). "
                    "New content will be searchable once it completes.",
                    self.config.databricks_sync_timeout_seconds,
                    state,
                    indexed,
                    expected,
                )
                return False
            self._sleep(poll)

    # --- writes -----------------------------------------------------------------
    def add_chunks(self, chunks: List[Chunk], *, sync: bool = True) -> int:
        if not chunks:
            logger.warning("add_chunks() called with no chunks")
            return 0
        unique: dict[str, Chunk] = {}
        for chunk in chunks:
            unique.setdefault(chunk.chunk_id, chunk)
        with self._lock:
            existing = self.table.existing_ids(list(unique))
            new_chunks = [c for cid, c in unique.items() if cid not in existing]
            if not new_chunks:
                logger.info("All %d chunks already stored; nothing to add", len(unique))
                return 0
            vectors = self.embedding_service.embed_documents([c.text for c in new_chunks])
            rows = [
                {
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "document_id": str(c.metadata.get("document_id", "")),
                    "source_file": str(c.metadata.get("source_file", "")),
                    "page_number": int(c.metadata.get("page_number") or 0),
                    "section_title": str(c.metadata.get("section_title", "")),
                    "content_hash": str(c.metadata.get("content_hash", "")),
                    "metadata_json": json.dumps(c.metadata, ensure_ascii=False),
                    "embedding_json": json.dumps(v),
                }
                for c, v in zip(new_chunks, vectors)
            ]
            self.table.insert(rows)
            self._invalidate()
            index = self._get_index(create_with_dimension=len(vectors[0]))
            self._check_dimension(index, len(vectors[0]))
        logger.info("Stored %d new chunks in %s (%d already present)", len(new_chunks), self.source_table_name, len(existing))
        if sync:
            self.sync()
        return len(new_chunks)

    def delete_document(
        self,
        *,
        document_id: str | None = None,
        source_file: str | None = None,
        exclude_document_id: str | None = None,
        sync: bool = True,
    ) -> int:
        if not document_id and not source_file:
            raise ValueError("document_id or source_file is required.")
        with self._lock:
            deleted = self.table.delete(
                document_id=document_id, source_file=source_file, exclude_document_id=exclude_document_id
            )
            if deleted:
                self._invalidate()
        logger.info("Deleted %d chunks (document_id=%s, source_file=%s)", deleted, document_id, source_file)
        if deleted and sync:
            self.sync()
        return deleted

    def clear_collection(self) -> None:
        logger.warning("Deleting all rows from %s", self.source_table_name)
        with self._lock:
            self.table.delete(document_id=None, source_file=None, exclude_document_id=None)
            self._invalidate()
        self.sync()

    # --- reads ------------------------------------------------------------------
    def has_document(self, document_id: str) -> bool:
        return self.table.has_document(document_id)

    def list_documents(self) -> list[StoredDocument]:
        docs = [StoredDocument(d, s, c, p) for d, s, c, p in self.table.list_documents()]
        return sorted(docs, key=lambda d: d.source_file.lower())

    def get_all(self, where: Optional[dict[str, Any]] = None) -> tuple[list[str], list[str], list[dict]]:
        ids, texts, metas = [], [], []
        for chunk_id, text, metadata_json in self.table.select_all():
            try:
                meta = json.loads(metadata_json)
            except json.JSONDecodeError:
                meta = {}
            ids.append(chunk_id)
            texts.append(text)
            metas.append(meta)
        return ids, texts, metas

    def query_by_vector(
        self, vector: list[float], top_k: int, filters: Optional[dict[str, Any]] = None
    ) -> list[dict[str, Any]]:
        if self.document_count() == 0:
            return []
        index = self._get_index()
        if index is None:
            logger.warning("Vector Search index %s does not exist yet", self.index_name)
            return []
        kwargs: dict[str, Any] = {"query_vector": vector, "columns": _RESULT_COLUMNS, "num_results": top_k}
        vs_filters = to_vector_search_filters(filters)
        if vs_filters:
            kwargs["filters"] = vs_filters
        response = retry_call(
            lambda: index.similarity_search(**kwargs),
            operation="databricks.vector_search",
            max_retries=self.config.max_retries,
            base_delay=self.config.retry_base_delay_seconds,
        )
        columns = [c.get("name") for c in (response.get("manifest") or {}).get("columns", [])]
        rows: list[dict[str, Any]] = []
        for values in (response.get("result") or {}).get("data_array") or []:
            record = dict(zip(columns, values))
            try:
                meta = json.loads(record.get("metadata_json") or "{}")
            except json.JSONDecodeError:
                meta = {}
            embedding = record.get("embedding")
            similarity = _cosine(vector, embedding) if embedding else float(record.get("score") or 0.0)
            rows.append(
                {"id": str(record.get("chunk_id")), "text": record.get("text") or "", "metadata": meta, "similarity": similarity}
            )
        rows.sort(key=lambda r: r["similarity"], reverse=True)
        return rows

    def similarity_search_with_scores(self, query: str, top_k: int = None):
        """Backward-compatible API: list of (Document, distance) where lower = more similar."""
        from langchain_core.documents import Document

        k = top_k or self.config.retrieval_top_k
        rows = self.query_by_vector(self.embedding_service.embed_query(query), k)
        return [(Document(page_content=r["text"], metadata=r["metadata"]), 1.0 - r["similarity"]) for r in rows]

    def get_retriever(self, top_k: int = None):
        raise RuntimeError(
            "get_retriever() is only available with the local Chroma store. "
            "Use RAGPipeline.retriever (hybrid retrieval) with the Databricks backend."
        )
