"""
Make 'gold documents' and 'recall@k' concrete by walking through ONE query.

    python explain_one.py            # first query in your slice
    python explain_one.py --i 7
    python explain_one.py --k 10     # see how recall changes with k

Prints, in order:
  1. the question
  2. the GOLD documents  -- the answer key that ships with the dataset
  3. the k chunks your system actually retrieved
  4. the recall arithmetic, done by hand

No API key needed. This is retrieval only.
"""

import argparse

import rag
from evaluate import gold_urls, retrieval_scores, unique_docs


def short(url, n=60):
    return url if len(url) <= n else url[:n - 3] + "..."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--i", type=int, default=0, help="which query in the slice")
    ap.add_argument("--k", type=int, default=rag.TOP_K)
    ap.add_argument("--n", type=int, default=50, help="slice size to draw from")
    ap.add_argument("--max-per-doc", type=int, default=None,
                    help="cap chunks per document (try 1)")
    args = ap.parse_args()

    corpus = rag.load_corpus()
    queries, _ = rag.load_queries(n=args.n)
    q = queries[args.i]

    chunks = rag.build_chunks(corpus)
    embedder = rag.make_embedder()
    client = rag.build_index(chunks, embedder)
    passages = rag.retrieve(client, embedder, q["query"], k=args.k,
                            max_per_doc=args.max_per_doc)

    gold = gold_urls(q)

    print("\n" + "=" * 70)
    print("THE QUESTION")
    print("=" * 70)
    print(q["query"])
    print(f"\nexpected answer : {q['answer']}")
    print(f"question type   : {q['question_type']}")

    print("\n" + "=" * 70)
    print(f"GOLD DOCUMENTS ({len(gold)})  <- the answer key, shipped with the dataset")
    print("=" * 70)
    print("A human wrote this question FROM these articles and recorded which")
    print("ones. Your system never sees this -- it is only used to score.\n")
    for e in q["evidence_list"]:
        print(f"  * {e['title'][:64]}")
        print(f"    {short(e['url'])}")

    print("\n" + "=" * 70)
    print(f"WHAT YOUR SYSTEM RETRIEVED (top {args.k} CHUNKS)")
    print("=" * 70)
    print("Note: Qdrant returns CHUNKS, not documents. Several chunks can come")
    print("from the same article -- that is the whole problem.\n")
    for rank, p in enumerate(passages, 1):
        mark = "HIT " if p["url"] in gold else "miss"
        print(f"  {rank}. [{mark}] {p['title'][:58]}")
        print(f"          {short(p['url'])}")

    got = {p["url"] for p in passages}
    found = got & gold

    print("\n" + "=" * 70)
    print("THE ARITHMETIC")
    print("=" * 70)
    print(f"  chunks retrieved        : {len(passages)}   (this is k)")
    print(f"  DISTINCT documents      : {unique_docs(passages)}")
    print(f"  gold documents needed   : {len(gold)}")
    print(f"  gold documents found    : {len(found)}")

    partial, all_ev = retrieval_scores(passages, gold)
    print(f"\n  partial recall@{args.k}       = {len(found)}/{len(gold)} = {partial:.2f}")
    print(f"  ALL-evidence recall@{args.k}  = {'1 (got every one)' if all_ev else '0 (missed at least one)'}")

    if not all_ev:
        print("\n  -> At least one required article never reached the model.")
        print("     No prompt engineering can fix that. This is a RETRIEVAL failure.")
    if unique_docs(passages) < len(gold):
        print(f"\n  -> Only {unique_docs(passages)} distinct documents in {args.k} slots, but "
              f"{len(gold)} were needed.")
        print("     All-evidence recall was impossible before the LLM was even called.")
        print("     Try: --k 10, or run with --no-title (see README).")
    print()


if __name__ == "__main__":
    main()