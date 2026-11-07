"""Pre-flight check for VECTOR_STORE=databricks. Run from the project root:

    uv run python -m scripts.check_databricks

Verifies configuration, SQL warehouse access, table permissions, the Vector
Search endpoint, and (optionally) the Databricks serving endpoints. Never
prints secrets.
"""
from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.config import settings  # noqa: E402

OK, FAIL = "[ OK ]", "[FAIL]"


def _step(label: str, func) -> bool:
    try:
        detail = func()
        print(f"{OK} {label}" + (f": {detail}" if detail else ""))
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"{FAIL} {label}: {type(exc).__name__}: {exc}")
        return False


def main() -> int:
    cfg = settings
    print(f"Workspace: {cfg.databricks_host or '(not set)'}")
    print(f"Table:     {cfg.databricks_table_fqn}")
    print(f"Index:     {cfg.databricks_index_fqn} on endpoint '{cfg.databricks_vs_endpoint}'\n")

    if not _step("Configuration", lambda: cfg.validate_vector_store() or "VECTOR_STORE settings complete"):
        return 1

    from vectorstore.databricks_store import DeltaChunkTable

    table = DeltaChunkTable(cfg)
    results = [
        _step("SQL warehouse connection", lambda: f"SELECT 1 -> {table.execute('SELECT 1')[0][0]}"),
        _step("Create/verify chunk table (needs CREATE TABLE or existing table + MODIFY)",
              lambda: table.ensure_table() or "ready"),
        _step("Read chunk table", lambda: f"{table.count()} rows"),
    ]

    def _endpoint() -> str:
        from databricks.vector_search.client import VectorSearchClient

        client = VectorSearchClient(workspace_url=cfg.databricks_host,
                                    personal_access_token=cfg.databricks_token, disable_notice=True)
        info = client.get_endpoint(cfg.databricks_vs_endpoint)
        state = (info.get("endpoint_status") or {}).get("state", "unknown")
        return f"state={state}"

    results.append(_step("Vector Search endpoint", _endpoint))

    for purpose, provider in (("embedding", cfg.embedding_provider), ("chat", cfg.resolved_chat_provider)):
        if provider == "databricks":
            if purpose == "embedding":
                def _embed() -> str:
                    from embeddings.embedding_service import EmbeddingService

                    svc = EmbeddingService(config=cfg.with_overrides(embedding_cache_enabled=False))
                    return f"dimension={len(svc.embed_query('connectivity check'))}"
                results.append(_step("Databricks embedding endpoint", _embed))
            else:
                def _chat() -> str:
                    from rag.rag_pipeline import create_chat_model

                    model, name = create_chat_model(cfg)
                    model.invoke("Reply with OK.")
                    return name
                results.append(_step("Databricks chat endpoint", _chat))

    table.close()
    print("\nAll checks passed." if all(results) else "\nSome checks failed - see messages above.")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
