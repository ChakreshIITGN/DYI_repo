"""
Day 1 driver.

    python run_day1.py              # 200 queries
    python run_day1.py --n 20       # smoke test first -- do this one
    python run_day1.py --no-llm     # retrieval only, zero API cost

Writes results/day1.md and appends a row to results.md.
"""

import argparse
import json
import os
import time
from pathlib import Path

from tqdm import tqdm

import rag
from evaluate import (answer_correct, format_report, gold_urls, is_abstention,
                      retrieval_scores, summarise, unique_docs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--k", type=int, default=rag.TOP_K)
    ap.add_argument("--no-llm", action="store_true",
                    help="retrieval metrics only, no API calls")
    ap.add_argument("--max-per-doc", type=int, default=None,
                    help="cap chunks per document (try 1) -- forces diversity")
    ap.add_argument("--no-title", action="store_true",
                    help="do NOT prepend the document title to each chunk")
    ap.add_argument("--label", default="day1-baseline")
    args = ap.parse_args()

    print("loading corpus (all 609 docs -- never sliced)...")
    corpus = rag.load_corpus()
    queries, nulls = rag.load_queries(n=args.n)
    print(f"  {len(corpus)} documents, {len(queries)} queries")
    print(f"  {len(nulls)} null_query rows held back for Day 7 abstention test")

    print(f"chunking (title prefix: {not args.no_title})...")
    chunks = rag.build_chunks(corpus, include_title=not args.no_title)
    print(f"  {len(chunks)} chunks (~{len(chunks)/len(corpus):.1f} per doc)")

    print(f"embedding with {rag.EMBED_MODEL}...")
    embedder = rag.make_embedder()
    client = rag.build_index(chunks, embedder)

    llm = None if args.no_llm else rag.make_llm()

    rows, latencies = [], []
    for q in tqdm(queries, desc="querying"):
        t0 = time.perf_counter()
        passages = rag.retrieve(client, embedder, q["query"], k=args.k,
                                max_per_doc=args.max_per_doc)
        partial, all_ev = retrieval_scores(passages, gold_urls(q))

        generated, (ptok, ctok) = "", (0, 0)
        if llm:
            generated, (ptok, ctok) = rag.answer(llm, q["query"], passages)
        latencies.append((time.perf_counter() - t0) * 1000)

        rows.append({
            "query": q["query"],
            "question_type": q["question_type"],
            "expected": q["answer"],
            "generated": generated,
            "partial_recall": partial,
            "all_evidence_recall": all_ev,
            "correct": answer_correct(generated, q["answer"]) if llm else 0.0,
            "abstained": is_abstention(generated) if llm else 0.0,
            "n_gold": len(gold_urls(q)),
            "unique_docs": unique_docs(passages),
            "prompt_tokens": ptok,
            "completion_tokens": ctok,
        })

    summary = summarise(rows)
    report = format_report(summary, latencies)
    print("\n" + report)

    Path("results").mkdir(exist_ok=True)
    # Per-label filename, NOT a fixed one -- otherwise each run overwrites the
    # last and you cannot compare Day 2's configs against Day 1's baseline.
    Path(f"results/{args.label}_rows.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows))
    Path(f"results/{args.label}.md").write_text(
        f"# {args.label}\n\n"
        f"chunk={rag.CHUNK_WORDS}w/{rag.CHUNK_OVERLAP}ov  k={args.k}  "
        f"title_prefix={not args.no_title}  max_per_doc={args.max_per_doc}  "
        f"embed={rag.EMBED_MODEL}\n\n{report}\n"
    )

    with open("results.md", "a") as f:
        o = summary["overall"]
        f.write(
            f"| {args.label} | k={args.k} chunk={rag.CHUNK_WORDS}w | "
            f"{o['partial_recall']:.3f} | {o['all_evidence_recall']:.3f} | "
            f"{o['accuracy']:.3f} |\n"
        )

    print(f"\nwrote results/{args.label}_rows.jsonl")
    print(f"  python inspect_failures.py --label {args.label}")
    print(f"  python bucket.py --label {args.label}")


if __name__ == "__main__":
    main()