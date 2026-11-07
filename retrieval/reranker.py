"""Optional cross-encoder reranking (RERANKER=cross_encoder).

Requires ``sentence-transformers``. If the package or model is unavailable the
reranker disables itself and retrieval falls back to hybrid fusion order.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

from utils.logger import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str, model: Any | None = None) -> None:
        self.model_name = model_name
        self._model = model
        self.available = model is not None
        if model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(model_name, max_length=512)
                self.available = True
                logger.info("Cross-encoder reranker loaded: %s", model_name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Reranker disabled (%s: %s)", type(exc).__name__, exc)

    def score(self, query: str, texts: Sequence[str]) -> Optional[list[float]]:
        if not self.available or not texts:
            return None
        try:
            return [float(s) for s in self._model.predict([(query, t) for t in texts])]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Reranking failed; using fusion order (%s)", type(exc).__name__)
            return None


def build_reranker(kind: str, model_name: str) -> Optional[CrossEncoderReranker]:
    if kind == "cross_encoder":
        reranker = CrossEncoderReranker(model_name)
        return reranker if reranker.available else None
    return None
