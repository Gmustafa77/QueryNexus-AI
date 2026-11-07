"""Context building and answer verification (the anti-hallucination layer).

* ``build_context``: removes exact/near duplicates, enforces a token budget,
  assigns stable citation labels ([S1]..[Sn]).
* ``verify_answer``: validates citation labels, removes fabricated ones,
  checks that every number in the answer exists in the cited evidence, and
  enforces strict-citation / no-answer behaviour.
* ``detect_prompt_injection``: flags common injection phrasing in questions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from prompts.prompt_template import NO_ANSWER_MESSAGE
from retrieval.retriever import RetrievedChunk
from utils.text import count_tokens, jaccard_similarity

_CITATION_GROUP_RE = re.compile(r"\[\s*(S\d+(?:\s*[,;]\s*S\d+)*)\s*\]", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore (all |any )?(previous|prior|above|earlier) (instructions|rules|prompts?)",
        r"disregard (the |your )?(system|previous|above) (prompt|instructions)",
        r"(reveal|show|print|repeat) (me )?(your|the) (system )?(prompt|instructions)",
        r"you are now\b",
        r"act as (an? )?(unrestricted|different|jailbroken)",
        r"developer mode",
        r"</?(evidence|source|question|system)>",
    )
]


@dataclass
class Citation:
    label: str
    source_file: str
    page_number: int
    section_title: str
    chunk_id: str
    storage: str = ""  # "local" | "databricks"


@dataclass
class VerifiedAnswer:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    cited_chunks: list[RetrievedChunk] = field(default_factory=list)
    is_grounded: bool = False
    is_no_answer: bool = False
    warnings: list[str] = field(default_factory=list)
    citation_coverage: float = 0.0


def detect_prompt_injection(text: str) -> bool:
    return any(p.search(text or "") for p in _INJECTION_PATTERNS)


def build_context(
    chunks: Sequence[RetrievedChunk], *, max_tokens: int, near_duplicate_threshold: float
) -> list[RetrievedChunk]:
    """Deduplicate, fit to a token budget and label evidence in relevance order."""
    selected: list[RetrievedChunk] = []
    seen_hashes: set[str] = set()
    used_tokens = 0
    for chunk in chunks:
        content_hash = str(chunk.metadata.get("content_hash") or "")
        if content_hash and content_hash in seen_hashes:
            continue
        if any(jaccard_similarity(chunk.text, kept.text) >= near_duplicate_threshold for kept in selected):
            continue
        tokens = count_tokens(chunk.text) + 40  # + metadata envelope
        if selected and used_tokens + tokens > max_tokens:
            continue
        seen_hashes.add(content_hash)
        used_tokens += tokens
        selected.append(chunk)
    for position, chunk in enumerate(selected, start=1):
        chunk.citation_label = f"S{position}"
    return selected


def _normalize_number(value: str) -> str:
    return value.replace(",", "").replace("$", "").rstrip("%").rstrip(".")


def is_no_answer(text: str) -> bool:
    lowered = text.strip().lower()
    return lowered.startswith(NO_ANSWER_MESSAGE[:60].lower()) or lowered.startswith(
        "i couldn't find enough information"
    )


def verify_answer(answer: str, context: Sequence[RetrievedChunk], *, strict: bool) -> VerifiedAnswer:
    text = (answer or "").strip()
    if not text:
        return VerifiedAnswer(answer=NO_ANSWER_MESSAGE, is_no_answer=True, warnings=["Model returned an empty answer."])
    if is_no_answer(text):
        return VerifiedAnswer(answer=NO_ANSWER_MESSAGE, is_no_answer=True)

    by_label = {c.citation_label.upper(): c for c in context if c.citation_label}
    warnings: list[str] = []
    invalid: set[str] = set()
    used: list[str] = []

    def _rewrite(match: re.Match) -> str:
        labels = [part.strip().upper() for part in re.split(r"[,;]", match.group(1))]
        valid = []
        for label in labels:
            if label in by_label:
                valid.append(label)
                if label not in used:
                    used.append(label)
            else:
                invalid.add(label)
        return "".join(f"[{label}]" for label in valid)

    cleaned = _CITATION_GROUP_RE.sub(_rewrite, text)
    cleaned = re.sub(r"[ \t]+\n", "\n", re.sub(r"[ \t]{2,}", " ", cleaned)).strip()
    if invalid:
        warnings.append(f"Removed citation(s) not present in the evidence: {', '.join(sorted(invalid))}.")

    if not used:
        if strict:
            return VerifiedAnswer(
                answer=NO_ANSWER_MESSAGE,
                is_no_answer=True,
                warnings=["The generated answer had no verifiable citations, so it was withheld."],
            )
        warnings.append("The answer has no verifiable citations; treat it with caution.")

    cited_chunks = [by_label[label] for label in used]
    cited_text = " ".join(c.text for c in cited_chunks)
    evidence_numbers = {_normalize_number(n) for n in _NUMBER_RE.findall(cited_text)}
    answer_numbers = {
        _normalize_number(n)
        for n in _NUMBER_RE.findall(_CITATION_GROUP_RE.sub(" ", cleaned))
        if len(_normalize_number(n)) > 0
    }
    unsupported = sorted(n for n in answer_numbers if n not in evidence_numbers and not re.fullmatch(r"[1-9]", n))
    if unsupported:
        warnings.append(
            "These figures could not be found in the cited sources and should be verified: "
            + ", ".join(unsupported)
        )

    sentences = [s for s in _SENTENCE_RE.split(_strip_headings(cleaned)) if len(s.split()) >= 4]
    cited_sentences = [s for s in sentences if re.search(r"\[S\d+\]", s, re.IGNORECASE)]
    coverage = len(cited_sentences) / len(sentences) if sentences else (1.0 if used else 0.0)

    citations = [
        Citation(
            label=label,
            source_file=chunk.source_file,
            page_number=chunk.page_number,
            section_title=chunk.section_title,
            chunk_id=chunk.chunk_id,
            storage=chunk.storage,
        )
        for label, chunk in ((label, by_label[label]) for label in used)
    ]
    return VerifiedAnswer(
        answer=cleaned,
        citations=citations,
        cited_chunks=cited_chunks,
        is_grounded=bool(used) and not unsupported,
        warnings=warnings,
        citation_coverage=round(coverage, 3),
    )


def _strip_headings(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.strip().endswith(":"))
