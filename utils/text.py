"""Shared text utilities: token counting, keyword tokenization, hashing, similarity."""
from __future__ import annotations

import hashlib
import re
from functools import lru_cache

from utils.logger import get_logger

logger = get_logger(__name__)

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[.'\-][A-Za-z0-9]+)*|\$?\d+(?:[.,]\d+)*%?")
_APPROX_TOKEN_RE = re.compile(r"\w+|[^\w\s]")

STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have how i if in into is it its
    me my of on or our should so than that the their them then there these they this those to under was we were
    what when where which who whom why will with would you your about any all also am may might must not no""".split()
)


@lru_cache(maxsize=1)
def _encoder():
    """Load tiktoken once; fall back to a regex estimate when unavailable (offline/proxy)."""
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception as exc:  # noqa: BLE001
        logger.warning("tiktoken unavailable (%s); using approximate token counts", type(exc).__name__)
        return None


def count_tokens(text: str) -> int:
    if not text:
        return 0
    encoder = _encoder()
    if encoder is not None:
        return len(encoder.encode(text, disallowed_special=()))
    return int(len(_APPROX_TOKEN_RE.findall(text)) * 1.3) + 1


def tokenize(text: str, *, drop_stopwords: bool = True) -> list[str]:
    """Lower-cased keyword tokens for BM25 and lexical checks."""
    tokens = [t.lower().strip(".'-") for t in _WORD_RE.findall(text or "")]
    tokens = [t for t in tokens if t]
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_for_hash(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def jaccard_similarity(a: str, b: str, n: int = 3) -> float:
    """Word n-gram Jaccard similarity for near-duplicate detection."""
    ta, tb = tokenize(a, drop_stopwords=False), tokenize(b, drop_stopwords=False)
    if not ta or not tb:
        return 0.0
    sa = {tuple(ta[i : i + n]) for i in range(max(1, len(ta) - n + 1))}
    sb = {tuple(tb[i : i + n]) for i in range(max(1, len(tb) - n + 1))}
    return len(sa & sb) / len(sa | sb)
