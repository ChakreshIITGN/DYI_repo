"""
Compare runs side by side. This is the Day 2 workhorse.

    python compare.py                              # every run in results/
    python compare.py day1-baseline hybrid rerank  # just these, in this order

Deltas are against the FIRST run listed, so put your baseline first. That one
number -- "hybrid bought me X points" -- is the whole point of Day 2.
"""

import argparse
import json
from pathlib import Path

from evaluate import summarise


def load(label):
    p = Path(f"results/{label}_rows.jsonl")
    if not p.exists():
        raise SystemExit(f"no such run: {p}\n"
                         f"run it with:  python run_day1.py --label {label}")
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", nargs="*")
    ap.add_argument("--by-type", action="store_true",
                    help="also break every run down by question_type")
    args = ap.parse_args()

    labels = args.labels or sorted(
        p.name.replace("_rows.jsonl", "")
        for p in Path("results").glob("*_rows.jsonl"))
    if not labels:
        raise SystemExit("no runs found in results/")

    runs = {lab: summarise(load(lab)) for lab in labels}
    base = runs[labels[0]]["overall"]

    cols = [("partial_recall", "partial"), ("all_evidence_recall", "ALL-ev"),
            ("accuracy", "acc"), ("avg_unique_docs", "uniq"),
            ("avg_prompt_tokens", "ptok")]

    print("=" * 86)
    print(f"{'run':<22}{'n':<5}" + "".join(f"{h:<13}" for _, h in cols))
    print("=" * 86)
    for lab in labels:
        o = runs[lab]["overall"]
        line = f"{lab:<22}{o['n']:<5}"
        for key, _ in cols:
            val = o[key]
            if lab == labels[0]:
                line += f"{val:<13.3f}"
            else:
                d = val - base[key]
                line += f"{val:.3f} {d:+.3f} "[:13].ljust(13)
        print(line)
    print("-" * 86)
    print(f"deltas are vs '{labels[0]}'")

    if args.by_type:
        types = sorted({t for r in runs.values() for t in r["by_type"]})
        for qt in types:
            print(f"\n{qt}")
            print("-" * 86)
            print(f"{'run':<22}{'n':<5}" + "".join(f"{h:<13}" for _, h in cols))
            for lab in labels:
                s = runs[lab]["by_type"].get(qt)
                if not s:
                    continue
                print(f"{lab:<22}{s['n']:<5}" +
                      "".join(f"{s[k]:<13.3f}" for k, _ in cols))


if __name__ == "__main__":
    main()