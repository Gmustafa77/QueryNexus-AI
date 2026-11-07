"""Create vector-store backends.

Storage names used across the app:
  * ``local``      - ChromaDB on this machine
  * ``databricks`` - Delta table + Databricks Vector Search
"""
from __future__ import annotations

from typing import Any

from embeddings.embedding_service import EmbeddingService
from utils.config import Config

STORAGE_LOCAL = "local"
STORAGE_DATABRICKS = "databricks"
STORAGES = (STORAGE_LOCAL, STORAGE_DATABRICKS)


def storage_for_backend(backend_name: str) -> str:
    """Map a store's ``backend_name`` ("chroma" / "databricks") to a storage name."""
    return STORAGE_DATABRICKS if backend_name == "databricks" else STORAGE_LOCAL


def create_store(storage: str, config: Config, embedding_service: EmbeddingService) -> Any:
    if storage == STORAGE_DATABRICKS:
        from vectorstore.databricks_store import DatabricksVectorStore

        config.validate_databricks()
        return DatabricksVectorStore(embedding_service, config=config)
    if storage == STORAGE_LOCAL:
        from vectorstore.chroma_manager import ChromaManager

        return ChromaManager(embedding_service, config=config)
    raise ValueError(f"Unknown storage '{storage}'. Expected one of: {', '.join(STORAGES)}.")


def create_vector_store(config: Config, embedding_service: EmbeddingService) -> Any:
    """Backward-compatible: create the store selected by VECTOR_STORE."""
    config.validate_vector_store()
    storage = STORAGE_DATABRICKS if config.vector_store == "databricks" else STORAGE_LOCAL
    return create_store(storage, config, embedding_service)
