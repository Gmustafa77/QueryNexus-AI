"""Dependency-free Okapi BM25 keyword index for hybrid retrieval.

Exact-term matching complements embeddings for plan codes, dollar amounts,
form numbers and defined insurance terms that dense vectors often miss.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from utils.text import tokenize


@dataclass
class BM25Index:
    ids: list[str]
    texts: list[str]
    metadatas: list[dict[str, Any]]
    k1: float = 1.5
    b: float = 0.75
    _tfs: list[Counter] = field(default_factory=list, init=False, repr=False)
    _lengths: list[int] = field(default_factory=list, init=False, repr=False)
    _idf: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _avgdl: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not (len(self.ids) == len(self.texts) == len(self.metadatas)):
            raise ValueError("ids, texts and metadatas must have equal length.")
        df: Counter = Counter()
        for text in self.texts:
            tokens = tokenize(text)
            tf = Counter(tokens)
            self._tfs.append(tf)
            self._lengths.append(len(tokens))
            df.update(tf.keys())
        n = len(self.texts)
        self._avgdl = (sum(self._lengths) / n) if n else 0.0
        self._idf = {term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def __len__(self) -> int:
        return len(self.ids)

    def search(
        self,
        query: str,
        top_k: int,
        predicate: Optional[Callable[[dict[str, Any]], bool]] = None,
    ) -> list[tuple[int, float]]:
        """Return ``[(row_index, score)]`` for rows with score > 0, best first."""
        terms = [t for t in dict.fromkeys(tokenize(query)) if t in self._idf]
        if not terms or not self.ids:
            return []
        scored: list[tuple[int, float]] = []
        for i, tf in enumerate(self._tfs):
            if predicate is not None and not predicate(self.metadatas[i]):
                continue
            length_norm = self.k1 * (1 - self.b + self.b * self._lengths[i] / (self._avgdl or 1.0))
            score = 0.0
            for term in terms:
                freq = tf.get(term, 0)
                if freq:
                    score += self._idf[term] * freq * (self.k1 + 1) / (freq + length_norm)
            if score > 0:
                scored.append((i, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]


def keyword_coverage(query: str, text: str) -> float:
    """Fraction of distinct query keywords present in ``text`` (0..1)."""
    terms = set(tokenize(query))
    if not terms:
        return 0.0
    return len(terms & set(tokenize(text))) / len(terms)
