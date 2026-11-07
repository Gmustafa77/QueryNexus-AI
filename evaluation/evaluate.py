"""RAG evaluation: retrieval accuracy, answer accuracy, groundedness, faithfulness, citations.

Dataset: JSONL, one case per line::

    {"id": "deductible", "question": "What is the in-network deductible?",
     "answerable": true,
     "expected_sources": [{"source_file": "EOC_2026.pdf", "page_number": 12}],
     "expected_answer_contains": ["$1,500"]}

Run (uses the real configured providers)::

    python -m evaluation.evaluate --dataset evaluation/sample_eval.jsonl \
        --docs data/pdfs/EOC_2026.pdf --output evaluation/report.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Iterable

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.text import tokenize  # noqa: E402


@dataclass
class EvalCase:
    question: str
    answerable: bool = True
    expected_sources: list[dict[str, Any]] = field(default_factory=list)
    expected_answer_contains: list[str] = field(default_factory=list)
    id: str = ""


def load_cases(path: str) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                raw = json.loads(line)
                cases.append(
                    EvalCase(
                        question=raw["question"],
                        answerable=bool(raw.get("answerable", True)),
                        expected_sources=list(raw.get("expected_sources", [])),
                        expected_answer_contains=list(raw.get("expected_answer_contains", [])),
                        id=str(raw.get("id") or f"case_{line_number}"),
                    )
                )
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"Invalid evaluation case on line {line_number}: {exc}") from exc
    return cases


def _matches(expected: dict[str, Any], source_file: str, page_number: int) -> bool:
    if expected.get("source_file") and expected["source_file"] != source_file:
        return False
    if expected.get("page_number") is not None and int(expected["page_number"]) != int(page_number):
        return False
    return True


def _faithfulness(answer: str, cited_texts: Iterable[str]) -> float:
    """Share of answer content words that appear in the cited evidence (lexical proxy)."""
    words = [w for w in tokenize(answer.replace("[", " ").replace("]", " ")) if not w.startswith("s") or not w[1:].isdigit()]
    if not words:
        return 0.0
    evidence = set(tokenize(" ".join(cited_texts)))
    return sum(1 for w in words if w in evidence) / len(words)


def evaluate_case(pipeline, case: EvalCase, top_k: int | None = None) -> dict[str, Any]:
    retrieval = pipeline.retriever.retrieve_with_diagnostics(case.question, top_k=top_k)
    ranks = [
        rank
        for rank, chunk in enumerate(retrieval.chunks, start=1)
        if any(_matches(e, chunk.source_file, chunk.page_number) for e in case.expected_sources)
    ]
    found_expected = sum(
        1
        for e in case.expected_sources
        if any(_matches(e, c.source_file, c.page_number) for c in retrieval.chunks)
    )
    response = pipeline.ask(case.question, top_k=top_k)
    answer_lower = response.answer.lower()
    citation_hits = [
        any(_matches(e, c.source_file, c.page_number) for e in case.expected_sources) for c in response.citations
    ]
    row: dict[str, Any] = {
        "id": case.id,
        "question": case.question,
        "answerable": case.answerable,
        "is_no_answer": response.is_no_answer,
        "is_grounded": response.is_grounded,
        "warnings": response.warnings,
        "answer": response.answer,
        "citations": [f"{c.source_file} p.{c.page_number}" for c in response.citations],
        "no_answer_correct": response.is_no_answer == (not case.answerable),
    }
    if case.answerable and case.expected_sources:
        row["hit_at_k"] = 1.0 if ranks else 0.0
        row["mrr"] = 1.0 / ranks[0] if ranks else 0.0
        row["recall_at_k"] = found_expected / len(case.expected_sources)
    if case.answerable and not response.is_no_answer:
        if case.expected_answer_contains:
            row["answer_contains"] = mean(
                1.0 if s.lower() in answer_lower else 0.0 for s in case.expected_answer_contains
            )
        if case.expected_sources and citation_hits:
            row["citation_precision"] = sum(citation_hits) / len(citation_hits)
        row["faithfulness"] = _faithfulness(response.answer, [c.text for c in response.sources])
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def avg(key: str) -> float | None:
        values = [r[key] for r in rows if key in r]
        return round(mean(values), 4) if values else None

    answerable = [r for r in rows if r["answerable"]]
    return {
        "cases": len(rows),
        "retrieval_hit_at_k": avg("hit_at_k"),
        "retrieval_mrr": avg("mrr"),
        "retrieval_recall_at_k": avg("recall_at_k"),
        "answer_accuracy": avg("answer_contains"),
        "citation_precision": avg("citation_precision"),
        "faithfulness": avg("faithfulness"),
        "groundedness_rate": round(mean(1.0 if r["is_grounded"] else 0.0 for r in answerable), 4)
        if answerable
        else None,
        "no_answer_accuracy": avg("no_answer_correct"),
    }


def run_evaluation(pipeline, cases: list[EvalCase], top_k: int | None = None) -> dict[str, Any]:
    rows = [evaluate_case(pipeline, case, top_k) for case in cases]
    for row in rows:
        row["no_answer_correct"] = 1.0 if row["no_answer_correct"] else 0.0
    return {"summary": summarize(rows), "cases": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline.")
    parser.add_argument("--dataset", required=True, help="Path to JSONL evaluation cases.")
    parser.add_argument("--docs", nargs="*", default=[], help="PDF paths/globs to ingest before evaluating.")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--output", default="evaluation/report.json")
    parser.add_argument("--min-hit-rate", type=float, default=None, help="Exit 1 if hit@k is below this.")
    args = parser.parse_args(argv)

    from rag.rag_pipeline import RAGPipeline

    from utils.config import settings

    # Evaluation must exercise the real retrieval + LLM path, never cached answers.
    pipeline = RAGPipeline(config=settings.with_overrides(answer_cache_enabled=False))
    paths = [p for pattern in args.docs for p in glob.glob(pattern)]
    if paths:
        report = pipeline.ingest_with_report(paths)
        for status in report.files:
            print(f"[ingest] {status.source_file}: {status.status} {status.message}")
    result = run_evaluation(pipeline, load_cases(args.dataset), args.top_k)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(json.dumps(result["summary"], indent=2))
    hit = result["summary"]["retrieval_hit_at_k"]
    if args.min_hit_rate is not None and (hit is None or hit < args.min_hit_rate):
        print(f"Retrieval hit@k {hit} is below the required {args.min_hit_rate}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
