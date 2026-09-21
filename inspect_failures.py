"""
Read your failures by hand. Do not skip this -- it is where the war stories
your interviewer is fishing for actually come from.

    python inspect_failures.py            # 10 worst
    python inspect_failures.py --type temporal_query

For each one, decide which bucket it falls in. You will use these four buckets
on Day 3, and they are the backbone of "how would you debug a RAG system that
gives confident wrong answers?":

    A) retrieval missed evidence      -> fix retrieval (Day 2)
    B) retrieved everything, wrong    -> generation failure. No amount of
                                         retrieval tuning helps.
    C) retrieved everything, right,   -> your METRIC is wrong, not the system
       but scored wrong                  (e.g. "SBF" vs "Sam Bankman-Fried")
    D) question is genuinely ambiguous
"""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--type", default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in
            Path("results/day1_rows.jsonl").read_text().splitlines() if l.strip()]
    if args.type:
        rows = [r for r in rows if r["question_type"] == args.type]

    # Worst first: wrong answers with the least evidence retrieved.
    rows.sort(key=lambda r: (r["correct"], r["partial_recall"]))

    for i, r in enumerate(rows[:args.n], 1):
        print("=" * 72)
        print(f"{i}. [{r['question_type']}] needed {r['n_gold']} documents")
        print(f"Q: {r['query']}")
        print(f"expected:  {r['expected']}")
        print(f"generated: {r['generated']}")
        print(f"partial recall {r['partial_recall']:.2f} | "
              f"all-evidence {r['all_evidence_recall']:.0f} | "
              f"scored {r['correct']:.0f}")
        print("bucket? (A retrieval / B generation / C metric / D ambiguous)")
    print("=" * 72)
    print(f"\n{len(rows)} rows considered. Write your bucket counts into results.md.")


if __name__ == "__main__":
    main()