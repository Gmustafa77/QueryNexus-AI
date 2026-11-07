"""Inspect the persistent answer cache: python -m scripts.cache_stats [--clear]

Per-process hit/miss counters are in the app logs (search for 'Answer cache HIT/MISS') and in
RAGPipeline.answer_cache_stats(). This script shows what is stored on disk.
"""
import argparse
import json
import sqlite3
import time

from utils.config import settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=settings.answer_cache_path)
    parser.add_argument("--clear", action="store_true", help="Delete every cached answer.")
    args = parser.parse_args()
    conn = sqlite3.connect(args.path)
    if args.clear:
        with conn:
            conn.execute("DELETE FROM answer_cache")
        print("Cache cleared.")
        return
    now = time.time()
    rows = conn.execute(
        "SELECT question, meta, created_at, expires_at, hit_count FROM answer_cache ORDER BY hit_count DESC"
    ).fetchall()
    live = [r for r in rows if r[3] > now]
    print(f"{len(live)} live entries ({len(rows) - len(live)} expired), {sum(r[4] for r in live)} total hits")
    for question, meta, created, expires, hits in live[:25]:
        m = json.loads(meta)
        print(f"- hits={hits:<4} age={int(now - created)}s ttl_left={int(expires - now)}s tokens_saved_per_hit={m.get('total_tokens', 0)} "
              f"sources={m.get('sources')} | {question[:80]}")


if __name__ == "__main__":
    main()
