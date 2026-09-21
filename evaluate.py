"""
The eval harness. This file is the single most valuable thing you build all week.

Your interviewer said it plainly: "most people have vibes-based evaluation and
no actual metrics, and that tells me everything." Three metrics, each one you
can define precisely when asked.
"""

import re
import string


def normalise(s):
    s = s.lower().strip()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return re.sub(r"\s+", " ", s)


# ---------------------------------------------------------------------------
# Retrieval metrics
#
# Two recall numbers, because multi-hop makes the distinction matter and most
# candidates only know one of them:
#
#   partial_recall -- fraction of gold documents retrieved.
#                     Generous. Looks respectable even when the system fails.
#
#   all_evidence_recall -- did we get EVERY gold document?
#                     This is the one that predicts whether the answer can
#                     possibly be right. A multi-hop question needing 4
#                     documents is unanswerable if you retrieved 3.
#
# Reporting both, and knowing why they diverge, is a complete answer to
# "what does your eval pipeline look like?"
# ---------------------------------------------------------------------------

def gold_urls(query_row):
    return {e["url"] for e in query_row["evidence_list"]}


def retrieval_scores(retrieved_passages, gold):
    got = {p["url"] for p in retrieved_passages}
    if not gold:
        return 0.0, 0.0
    hit = len(got & gold)
    return hit / len(gold), float(hit == len(gold))


def unique_docs(retrieved_passages):
    """
    How many DISTINCT documents did the k chunks come from?

    With k=5 this is between 1 and 5. If it sits near 1-2, your slots are being
    eaten by several chunks of the same article -- which is fatal for multi-hop,
    where the answer needs 2-4 different documents.

    This is the number that makes the "titles cause clustering" effect visible.
    """
    return len({p["url"] for p in retrieved_passages})


# ---------------------------------------------------------------------------
# Answer metric
#
# Deliberately crude: normalised substring match. MultiHop-RAG answers are
# short (2-48 chars), so this mostly works and costs nothing.
#
# Be honest about what it gets wrong -- it UNDER-reports. "Sam Bankman-Fried"
# vs "SBF" scores zero. That is a feature on Day 1: you want a pessimistic
# baseline you can only improve on, and no LLM-judge bias in your first number.
#
# On Day 3 you upgrade to an LLM judge and report BOTH, so you can say how far
# apart they are. Knowing when to graduate from the dirty solution is the
# point; graduating before you have measured anything is not.
# ---------------------------------------------------------------------------

def answer_correct(generated, expected):
    g, e = normalise(generated), normalise(expected)
    return float(bool(e) and e in g)


def is_abstention(generated):
    return normalise(generated).startswith("i dont know")


# ---------------------------------------------------------------------------
# Aggregation, sliced by question_type.
#
# A single average hides everything. "Fine on inference queries, fell over on
# temporal ones, here's why" is the sentence that gets you hired.
# ---------------------------------------------------------------------------

def summarise(rows):
    def agg(subset):
        n = len(subset)
        if n == 0:
            return None
        return {
            "n": n,
            "partial_recall": sum(r["partial_recall"] for r in subset) / n,
            "all_evidence_recall": sum(r["all_evidence_recall"] for r in subset) / n,
            "accuracy": sum(r["correct"] for r in subset) / n,
            "abstain_rate": sum(r["abstained"] for r in subset) / n,
            "avg_prompt_tokens": sum(r["prompt_tokens"] for r in subset) / n,
            "avg_unique_docs": sum(r["unique_docs"] for r in subset) / n,
            "avg_gold_needed": sum(r["n_gold"] for r in subset) / n,
        }

    out = {"overall": agg(rows), "by_type": {}}
    for qt in sorted({r["question_type"] for r in rows}):
        out["by_type"][qt] = agg([r for r in rows if r["question_type"] == qt])
    return out


def format_report(summary, latencies_ms):
    lines = []
    o = summary["overall"]
    lines.append(f"n = {o['n']} queries\n")
    lines.append("| slice | n | partial recall@k | ALL-evidence recall@k | accuracy "
                 "| uniq docs in top-k | gold needed |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    lines.append(
        f"| **overall** | {o['n']} | {o['partial_recall']:.3f} | "
        f"{o['all_evidence_recall']:.3f} | {o['accuracy']:.3f} | "
        f"{o['avg_unique_docs']:.2f} | {o['avg_gold_needed']:.2f} |"
    )
    for qt, s in summary["by_type"].items():
        lines.append(
            f"| {qt} | {s['n']} | {s['partial_recall']:.3f} | "
            f"{s['all_evidence_recall']:.3f} | {s['accuracy']:.3f} | "
            f"{s['avg_unique_docs']:.2f} | {s['avg_gold_needed']:.2f} |"
        )
    lat = sorted(latencies_ms)
    if lat:
        p50 = lat[len(lat) // 2]
        p95 = lat[int(len(lat) * 0.95) - 1]
        lines.append(f"\nlatency p50 {p50:.0f} ms | p95 {p95:.0f} ms")
    lines.append(f"avg prompt tokens {o['avg_prompt_tokens']:.0f}")
    return "\n".join(lines)