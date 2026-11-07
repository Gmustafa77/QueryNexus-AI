"""STEP 3: Section-aware, token-aware chunking with rich, citation-ready metadata.

Processing order per document:
  1. Detect headings (PDF font hints first, conservative regex rules second).
  2. Carry the active section across page breaks (a section that starts on
     page 3 and continues on page 4 keeps its title on page 4).
  3. Pack paragraphs -> sentences -> word windows into chunks of at most
     ``chunk_size`` tokens with ``chunk_overlap`` tokens of sentence overlap.
  4. Chunks never span pages, so every chunk cites exactly one page.
  5. Attach deterministic IDs and content hashes for dedup and updates.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from ingestion.pdf_loader import PageContent
from utils.config import settings
from utils.logger import get_logger
from utils.text import count_tokens, normalize_for_hash, sha256_text

logger = get_logger(__name__)

UNTITLED_SECTION = "Untitled Section"


@dataclass(slots=True)
class Chunk:
    """A retrieval-ready chunk with source and structural metadata."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_id: str = ""

    def __post_init__(self) -> None:
        self.text = self.text.strip()
        if not self.text:
            raise ValueError("Chunk text must not be empty.")
        if not self.chunk_id:
            raise ValueError("chunk_id must not be empty.")


@dataclass(slots=True)
class _Section:
    title: str
    body: str
    level: int = 1


class DocumentChunker:
    """Section-aware, token-aware chunker. ``chunk_pages(pages)`` is unchanged for callers."""

    NUMBERED_HEADING_RE = re.compile(r"^(?P<number>\d+(?:\.\d+){0,4})[.)]?\s+(?P<title>[A-Z][^\n]{1,110})$")
    LABELLED_HEADING_RE = re.compile(
        r"^(?P<label>section|article|chapter|part)\s+(?P<number>[A-Z0-9IVXLC.\-]+)\s*[:.\-–—]?\s*(?P<title>.*)$",
        re.IGNORECASE,
    )
    SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9\"'(•\-])")
    WHITESPACE_RE = re.compile(r"[ \t]+")
    EXCESS_NEWLINES_RE = re.compile(r"\n{3,}")

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        *,
        prepend_section_title: bool = True,
        min_heading_length: int = 3,
        max_heading_length: int = 120,
        min_chunk_tokens: int | None = None,
    ) -> None:
        self.chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
        self.min_chunk_tokens = min_chunk_tokens if min_chunk_tokens is not None else settings.min_chunk_tokens
        self.prepend_section_title = prepend_section_title
        self.min_heading_length = min_heading_length
        self.max_heading_length = max_heading_length
        self._validate_configuration()
        logger.info(
            "DocumentChunker ready (chunk_size=%d tokens, overlap=%d tokens, min=%d tokens)",
            self.chunk_size,
            self.chunk_overlap,
            self.min_chunk_tokens,
        )

    # ------------------------------------------------------------------
    def chunk_pages(self, pages: Iterable[PageContent]) -> list[Chunk]:
        if pages is None:
            raise TypeError("pages must be an iterable of PageContent objects.")
        all_chunks: list[Chunk] = []
        seen_ids: set[str] = set()
        active: dict[str, tuple[str, int]] = {}  # source -> (section_title, level) carried across pages
        doc_chunk_index: dict[str, int] = {}

        for page_index, page in enumerate(pages):
            text = self._clean_text(getattr(page, "text", "") or "")
            if not text:
                continue
            source_file = str(getattr(page, "source_file", "unknown_source"))
            page_number = int(getattr(page, "page_number", page_index + 1) or page_index + 1)
            page_meta = dict(getattr(page, "metadata", {}) or {})
            document_id = str(page_meta.get("document_id") or sha256_text(source_file)[:32])
            hints = {h.strip() for h in page_meta.get("headings", []) or [] if isinstance(h, str)}
            carry_title, carry_level = active.get(document_id, (UNTITLED_SECTION, 1))

            sections = self._extract_sections(text, hints, carry_title, carry_level)
            if sections:
                active[document_id] = (sections[-1].title, sections[-1].level)

            page_chunk_index = 0
            for section_index, section in enumerate(sections):
                for chunk_index_in_section, body in enumerate(self._split_tokens(section.body)):
                    display = body
                    if self.prepend_section_title and section.title != UNTITLED_SECTION and not body.startswith(
                        section.title
                    ):
                        display = f"{section.title}\n{body}"
                    content_hash = sha256_text(normalize_for_hash(display))
                    index_in_doc = doc_chunk_index.get(document_id, 0)
                    doc_chunk_index[document_id] = index_in_doc + 1
                    chunk_id = self._build_chunk_id(
                        document_id=document_id,
                        page_number=page_number,
                        chunk_index=page_chunk_index,
                        content_hash=content_hash,
                    )
                    if chunk_id in seen_ids:
                        continue
                    seen_ids.add(chunk_id)
                    metadata = {
                        **self._scalar_metadata(page_meta),
                        "source": source_file,
                        "page": page_number,
                        "source_file": source_file,
                        "page_number": page_number,
                        "document_id": document_id,
                        "section_title": section.title,
                        "section_level": section.level,
                        "section_index_on_page": section_index,
                        "chunk_index_on_page": page_chunk_index,
                        "chunk_index_in_section": chunk_index_in_section,
                        "chunk_index_in_document": index_in_doc,
                        "chunk_tokens": count_tokens(display),
                        "chunk_length_chars": len(display),
                        "content_hash": content_hash,
                        "chunk_id": chunk_id,
                    }
                    all_chunks.append(Chunk(text=display, metadata=metadata, chunk_id=chunk_id))
                    page_chunk_index += 1

        logger.info("Created %d section-aware chunks", len(all_chunks))
        return all_chunks

    # ------------------------------------------------------------------
    def _extract_sections(self, text: str, hints: set[str], carry_title: str, carry_level: int) -> list[_Section]:
        sections: list[_Section] = []
        title, level = carry_title, carry_level
        body: list[str] = []

        def flush() -> None:
            content = "\n".join(body).strip()
            if content:
                sections.append(_Section(title=title, body=content, level=level))

        for line in text.splitlines():
            heading = self._parse_heading(line.strip(), hints)
            if heading is None:
                body.append(line)
                continue
            flush()
            title, level = heading
            body = []
        flush()
        if not sections and title != carry_title:
            sections.append(_Section(title=title, body=title, level=level))
        return sections

    def _parse_heading(self, line: str, hints: set[str]) -> tuple[str, int] | None:
        if not line or not self.min_heading_length <= len(line) <= self.max_heading_length:
            return None
        if line.endswith(("?", "!", ",", ";")) or sum(c.isalpha() for c in line) < 3:
            return None
        labelled = self.LABELLED_HEADING_RE.match(line)
        if labelled and len(line.split()) <= 14:
            return line, 1
        numbered = self.NUMBERED_HEADING_RE.match(line)
        if numbered and len(line.split()) <= 12 and not line.endswith("."):
            return line, min(numbered.group("number").count(".") + 1, 5)
        if line in hints and len(line.split()) <= 14:
            return line, 2
        letters = [c for c in line if c.isalpha()]
        if len(letters) >= 4 and line.upper() == line and not line.endswith((".", ":")) and len(line.split()) <= 10:
            return line, 1
        return None

    # ------------------------------------------------------------------
    def _split_tokens(self, text: str) -> list[str]:
        """Greedy-pack units into <= chunk_size tokens with sentence-level overlap."""
        units = self._units(text)
        chunks: list[list[tuple[str, int]]] = []
        current: list[tuple[str, int]] = []
        current_tokens = 0
        for unit, tokens in units:
            if current and current_tokens + tokens > self.chunk_size:
                chunks.append(current)
                overlap: list[tuple[str, int]] = []
                overlap_tokens = 0
                for prev in reversed(current):
                    if overlap_tokens + prev[1] > self.chunk_overlap:
                        break
                    overlap.insert(0, prev)
                    overlap_tokens += prev[1]
                if overlap_tokens + tokens > self.chunk_size:
                    overlap, overlap_tokens = [], 0
                current, current_tokens = overlap, overlap_tokens
            current.append((unit, tokens))
            current_tokens += tokens
        if current:
            if chunks and current_tokens < self.min_chunk_tokens:
                tail_new = [u for u in current if u not in chunks[-1]]
                merged_tokens = sum(t for _, t in chunks[-1]) + sum(t for _, t in tail_new)
                if merged_tokens <= int(self.chunk_size * 1.25):
                    chunks[-1].extend(tail_new)
                    current = []
            if current:
                chunks.append(current)
        return [self._join(chunk) for chunk in chunks if self._join(chunk)]

    def _units(self, text: str) -> list[tuple[str, int]]:
        units: list[tuple[str, int]] = []
        for paragraph in re.split(r"\n\s*\n", text):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            tokens = count_tokens(paragraph)
            if tokens <= self.chunk_size:
                units.append((paragraph, tokens))
                continue
            for sentence in self.SENTENCE_SPLIT_RE.split(paragraph):
                sentence = sentence.strip()
                if not sentence:
                    continue
                s_tokens = count_tokens(sentence)
                if s_tokens <= self.chunk_size:
                    units.append((sentence, s_tokens))
                else:
                    units.extend(self._word_windows(sentence))
        return units

    def _word_windows(self, text: str) -> list[tuple[str, int]]:
        words = text.split()
        windows: list[tuple[str, int]] = []
        step = max(1, int(self.chunk_size * 0.6))
        for start in range(0, len(words), step):
            piece = " ".join(words[start : start + step])
            windows.append((piece, count_tokens(piece)))
        return windows

    @staticmethod
    def _join(units: list[tuple[str, int]]) -> str:
        return "\n".join(u for u, _ in units).strip()

    # ------------------------------------------------------------------
    def _clean_text(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
        lines = [self.WHITESPACE_RE.sub(" ", line).strip() for line in text.splitlines()]
        return self.EXCESS_NEWLINES_RE.sub("\n\n", "\n".join(lines)).strip()

    @staticmethod
    def _scalar_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """Vector stores only accept scalar metadata values."""
        return {
            k: v
            for k, v in metadata.items()
            if isinstance(v, (str, int, float, bool)) and k != "headings"
        }

    def _validate_configuration(self) -> None:
        if not isinstance(self.chunk_size, int) or self.chunk_size < 50:
            raise ValueError("chunk_size must be an integer >= 50 tokens.")
        if not isinstance(self.chunk_overlap, int) or self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be a non-negative integer.")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")
        if self.min_heading_length < 1 or self.max_heading_length < self.min_heading_length:
            raise ValueError("Invalid heading length bounds.")

    @staticmethod
    def _build_chunk_id(*, document_id: str, page_number: int, chunk_index: int, content_hash: str) -> str:
        identity = f"{document_id}|{page_number}|{chunk_index}|{content_hash}"
        return "chunk_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
