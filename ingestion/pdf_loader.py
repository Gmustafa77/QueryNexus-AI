"""STEP 1-2: Validate PDFs and extract clean, page-level text with structural hints.

Improvements over the original loader:
  * validates existence, extension, size, magic bytes, encryption and page limits
  * per-page error isolation (one bad page never loses the whole document)
  * font-size/bold based heading detection (passed to the chunker as hints)
  * repeated header/footer removal and hyphenated line-break repair
  * document identity: SHA-256 file hash used as ``document_id`` for dedup/updates
"""
from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List

import pymupdf

from utils.config import settings
from utils.logger import get_logger
from utils.text import sha256_bytes

logger = get_logger(__name__)


class PDFLoadError(Exception):
    """Raised when a PDF cannot be loaded; the message is safe to show to users."""


@dataclass
class PageContent:
    """Extracted text for one PDF page plus metadata that travels to citations."""

    text: str
    page_number: int  # 1-indexed
    source_file: str
    metadata: dict = field(default_factory=dict)


@dataclass
class LoadResult:
    """Per-file outcome so the UI can report exactly what happened."""

    file_path: str
    source_file: str
    pages: List[PageContent] = field(default_factory=list)
    document_id: str = ""
    error: str = ""
    skipped_pages: int = 0

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.pages)


class PDFLoader:
    """Loads PDFs from disk and extracts page text using PyMuPDF."""

    _HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")
    _MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
    _MULTI_SPACE_RE = re.compile(r"[ \t\u00a0]{2,}")
    _PAGE_NUMBER_RE = re.compile(r"^(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?$", re.IGNORECASE)

    def __init__(self, *, max_bytes: int | None = None, max_pages: int | None = None) -> None:
        self.max_bytes = max_bytes or settings.max_upload_bytes
        self.max_pages = max_pages or settings.max_pdf_pages

    # ------------------------------------------------------------------
    def load_pdf(self, file_path: str) -> List[PageContent]:
        """Extract text from one PDF (one PageContent per non-empty page)."""
        result = self.load_pdf_with_report(file_path)
        if result.error:
            raise PDFLoadError(result.error)
        return result.pages

    def load_multiple_pdfs(self, file_paths: List[str]) -> List[PageContent]:
        """Load several PDFs; failures are logged and skipped (original behaviour)."""
        pages: List[PageContent] = []
        for result in self.load_many(file_paths):
            pages.extend(result.pages)
        return pages

    def load_many(self, file_paths: List[str]) -> List[LoadResult]:
        return [self.load_pdf_with_report(path) for path in file_paths]

    def load_pdf_with_report(self, file_path: str) -> LoadResult:
        filename = os.path.basename(file_path)
        result = LoadResult(file_path=file_path, source_file=filename)
        try:
            data = self._read_and_validate(file_path)
            result.document_id = sha256_bytes(data)[:32]
            result.pages, result.skipped_pages = self._extract(data, filename, result.document_id)
            if not result.pages:
                result.error = (
                    "No extractable text found. The PDF may be scanned/image-only and requires OCR."
                )
        except PDFLoadError as exc:
            result.error = str(exc)
        except Exception as exc:  # noqa: BLE001 - corrupted PDFs raise many exception types
            logger.exception("Unexpected failure loading '%s'", filename)
            result.error = f"Could not read PDF ({type(exc).__name__})."
        if result.error:
            logger.error("Failed to load '%s': %s", filename, result.error)
        else:
            logger.info(
                "Loaded '%s': %d text pages (%d skipped)", filename, len(result.pages), result.skipped_pages
            )
        return result

    # ------------------------------------------------------------------
    def _read_and_validate(self, file_path: str) -> bytes:
        if not os.path.isfile(file_path):
            raise PDFLoadError(f"PDF not found: {os.path.basename(file_path)}")
        if not file_path.lower().endswith(".pdf"):
            raise PDFLoadError("Only .pdf files are supported.")
        size = os.path.getsize(file_path)
        if size == 0:
            raise PDFLoadError("The file is empty.")
        if size > self.max_bytes:
            raise PDFLoadError(f"File exceeds the {self.max_bytes // (1024 * 1024)} MB limit.")
        with open(file_path, "rb") as handle:
            data = handle.read()
        if b"%PDF-" not in data[:1024]:
            raise PDFLoadError("The file is not a valid PDF.")
        return data

    def _extract(self, data: bytes, filename: str, document_id: str) -> tuple[List[PageContent], int]:
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            if doc.needs_pass and not doc.authenticate(""):
                raise PDFLoadError("The PDF is password-protected.")
            if doc.page_count > self.max_pages:
                raise PDFLoadError(f"PDF has {doc.page_count} pages; the limit is {self.max_pages}.")
            title = (doc.metadata or {}).get("title") or ""
            raw_pages: list[tuple[int, list[tuple[str, float, bool]]]] = []
            skipped = 0
            for index in range(doc.page_count):
                try:
                    raw_pages.append((index + 1, self._page_lines(doc.load_page(index))))
                except Exception as exc:  # noqa: BLE001
                    skipped += 1
                    logger.warning("Skipping unreadable page %d of '%s': %s", index + 1, filename, exc)
            total_pages = doc.page_count

        body_size = self._body_font_size(raw_pages)
        boilerplate = self._repeated_boilerplate(raw_pages)
        ingested_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        pages: List[PageContent] = []
        for page_number, lines in raw_pages:
            kept = [line for line in lines if self._norm(line[0]) not in boilerplate]
            headings = [
                text for text, size, bold in kept if self._looks_like_heading(text, size, bold, body_size)
            ]
            text = self._clean_text("\n".join(line[0] for line in kept))
            if not text:
                skipped += 1
                continue
            pages.append(
                PageContent(
                    text=text,
                    page_number=page_number,
                    source_file=filename,
                    metadata={
                        "source": filename,
                        "page": page_number,
                        "source_file": filename,
                        "page_number": page_number,
                        "document_id": document_id,
                        "document_title": title,
                        "total_pages": total_pages,
                        "ingested_at": ingested_at,
                        "headings": headings,
                    },
                )
            )
        return pages, skipped

    @staticmethod
    def _page_lines(page) -> list[tuple[str, float, bool]]:
        """Return (text, max_font_size, is_bold) per visual line in reading order."""
        lines: list[tuple[str, float, bool]] = []
        layout = page.get_text("dict", sort=True)
        for block in layout.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
                if not spans:
                    continue
                text = "".join(s["text"] for s in line["spans"]).strip()
                size = max(float(s.get("size", 0)) for s in spans)
                bold = all((int(s.get("flags", 0)) & 16) or "bold" in str(s.get("font", "")).lower() for s in spans)
                lines.append((text, size, bool(bold)))
            lines.append(("", 0.0, False))  # paragraph break between blocks
        return lines

    @staticmethod
    def _body_font_size(raw_pages) -> float:
        counter: Counter = Counter()
        for _, lines in raw_pages:
            for text, size, _ in lines:
                if text:
                    counter[round(size, 1)] += len(text)
        return counter.most_common(1)[0][0] if counter else 0.0

    def _repeated_boilerplate(self, raw_pages) -> set[str]:
        """Lines repeated at the top/bottom of >=50% of pages (headers, footers, page numbers)."""
        if len(raw_pages) < 4:
            return set()
        counter: Counter = Counter()
        for _, lines in raw_pages:
            texts = [t for t, _, _ in lines if t]
            edge = {self._norm(t) for t in texts[:2] + texts[-2:]}
            counter.update(e for e in edge if e)
        threshold = max(3, int(len(raw_pages) * 0.5))
        return {line for line, count in counter.items() if count >= threshold}

    def _norm(self, line: str) -> str:
        norm = re.sub(r"\d+", "#", line.strip().lower())
        return "#" if self._PAGE_NUMBER_RE.match(line.strip()) else norm

    @staticmethod
    def _looks_like_heading(text: str, size: float, bold: bool, body_size: float) -> bool:
        stripped = text.strip()
        if not 3 <= len(stripped) <= 120:
            return False
        if stripped.endswith((".", ",", ";")):
            return False
        if sum(ch.isalpha() for ch in stripped) < 3:
            return False
        larger = body_size > 0 and size >= body_size * 1.15
        return larger or (bold and len(stripped.split()) <= 12)

    @classmethod
    def _clean_text(cls, text: str) -> str:
        text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
        text = cls._HYPHEN_BREAK_RE.sub(r"\1-\2", text)  # keep "in-network" intact
        text = cls._MULTI_SPACE_RE.sub(" ", text)
        text = cls._MULTI_NEWLINE_RE.sub("\n\n", text)
        return text.strip()
