"""Read-only view over one or more vector stores (Local, Databricks, or both = Hybrid).

* Every returned row is tagged with ``metadata["storage"]`` so citations show
  where the evidence came from.
* Vector results from all stores are merged by cosine similarity (comparable
  because both stores hold vectors from the same embedding model).
* If one store fails during a query, the other store's results are still
  returned and the failure is reported back so the UI can warn the user.
* A ``storage`` filter key (e.g. ``{"storage": ["databricks"]}``) selects stores
  and is never forwarded to the backends.
"""
from __future__ import annotations

from typing import Any, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


class MultiVectorStore:
    def __init__(self, stores: dict[str, Any]) -> None:
        if not stores:
            raise ValueError("MultiVectorStore needs at least one store.")
        self.stores = dict(stores)
        self.embedding_service = next(iter(self.stores.values())).embedding_service

    @property
    def backend_name(self) -> str:
        return "+".join(self.stores)

    @property
    def version(self) -> int:
        return sum(int(getattr(store, "version", 0)) for store in self.stores.values())

    # ------------------------------------------------------------------
    def _select(self, filters: Optional[dict[str, Any]]) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
        if not filters or "storage" not in filters:
            return self.stores, filters
        wanted = filters["storage"]
        wanted = set(wanted if isinstance(wanted, (list, tuple, set)) else [wanted])
        rest = {k: v for k, v in filters.items() if k != "storage"} or None
        return {name: store for name, store in self.stores.items() if name in wanted}, rest

    @staticmethod
    def _tag(metadata: dict[str, Any], storage: str) -> dict[str, Any]:
        return {**(metadata or {}), "storage": storage}

    # ------------------------------------------------------------------
    def document_count(self) -> int:
        total = 0
        for name, store in self.stores.items():
            try:
                total += int(store.document_count())
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not count chunks in %s (%s)", name, type(exc).__name__)
        return total

    def query_by_vector_with_errors(
        self, vector: list[float], top_k: int, filters: Optional[dict[str, Any]] = None
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        selected, backend_filters = self._select(filters)
        merged: dict[str, dict[str, Any]] = {}
        errors: dict[str, str] = {}
        for name, store in selected.items():
            try:
                rows = store.query_by_vector(vector, top_k, backend_filters)
            except Exception as exc:  # noqa: BLE001
                errors[name] = f"{type(exc).__name__}: {exc}"
                logger.warning("Vector search failed in %s (%s)", name, type(exc).__name__)
                continue
            for row in rows:
                tagged = {**row, "metadata": self._tag(row.get("metadata", {}), name)}
                current = merged.get(tagged["id"])
                if current is None or tagged["similarity"] > current["similarity"]:
                    merged[tagged["id"]] = tagged
        ordered = sorted(merged.values(), key=lambda r: r["similarity"], reverse=True)
        return ordered[:top_k], errors

    def query_by_vector(
        self, vector: list[float], top_k: int, filters: Optional[dict[str, Any]] = None
    ) -> list[dict[str, Any]]:
        return self.query_by_vector_with_errors(vector, top_k, filters)[0]

    def get_all(self, where: Optional[dict[str, Any]] = None) -> tuple[list[str], list[str], list[dict]]:
        ids: list[str] = []
        texts: list[str] = []
        metas: list[dict] = []
        seen: set[str] = set()
        for name, store in self.stores.items():
            try:
                store_ids, store_texts, store_metas = store.get_all()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Keyword index could not read %s (%s)", name, type(exc).__name__)
                continue
            for chunk_id, text, meta in zip(store_ids, store_texts, store_metas):
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
                ids.append(chunk_id)
                texts.append(text)
                metas.append(self._tag(meta, name))
        return ids, texts, metas
