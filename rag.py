"""
Day 1: the simplest RAG pipeline that can be honestly measured.

No LangChain, no LlamaIndex. ~120 lines you can read top to bottom and explain
out loud. That matters: you cannot defend a design you did not write.

Pipeline: corpus -> chunks -> embeddings -> Qdrant -> top-k -> LLM -> answer
"""

from dotenv import load_dotenv
import os
import re
from dataclasses import dataclass

from datasets import load_dataset
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

load_dotenv()
# ---------------------------------------------------------------------------
# Config. Every knob you will turn this week lives here, so experiments are a
# one-line diff and you always know what changed between two numbers.
# ---------------------------------------------------------------------------

EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # 384-dim, CPU-friendly, no API key
CHUNK_WORDS = 300                       # ~400 tokens
CHUNK_OVERLAP = 50
TOP_K = 5
COLLECTION = "multihop"

# Works with any OpenAI-compatible endpoint: OpenAI, Together, Groq,
# OpenRouter, Ollama, or a local vLLM server. Set two env vars and you are done.
OLLAMA_API_KEY      = os.getenv("OLLAMA_API_KEY")
OLLAMA_HOST         = os.getenv("OLLAMA_HOST", "https://ollama.com")
LLM_MODEL           = os.getenv("LLM_MODEL", "gpt-oss:120b-cloud")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST       = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
# ---------------------------------------------------------------------------
# REQUIREMENT: we need CHAIN OF THOUGHT and TOKEN COUNTS from the model.
#
# Token counts pay for every cost number this week (accuracy per dollar on
# Day 4, the token-vs-accuracy regression on Day 7). The reasoning trace is
# what lets you answer "how do you debug a failure three hops deep?" with
# evidence instead of a shrug.
#
# The catch: the OpenAI-compatible shim (/v1/chat/completions) is a LOWEST
# COMMON DENOMINATOR. It has no standard field for a reasoning trace, and
# whether it populates `usage` at all is up to the server. `answer()` below
# is written defensively for exactly this reason.
#
# Ollama's NATIVE endpoint (/api/chat) exposes both properly:
#
#   request : "think": true            (also "low"/"medium"/"high"/"max")
#   response: message.thinking         <- the reasoning trace
#             prompt_eval_count        <- input tokens
#             prompt_eval_cached_count <- input tokens served from cache
#             eval_count               <- output tokens
#
# So: keep /v1 for portability now, and when you need the trace (Day 4
# onward), switch this file to /api/chat. Check what YOUR server actually
# returns before trusting either -- these shims change:
#
#   curl http://localhost:11434/api/chat -d '{
#     "model":"gpt-oss:120b-cloud","stream":false,"think":true,
#     "messages":[{"role":"user","content":"why is the sky blue?"}]}' | jq
#
# Rule of thumb for picking a model this week: if it cannot give you a token
# count, you cannot cost it, and an experiment you cannot cost is a demo.
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    text: str
    url: str    # document id -- joins to evidence_list[].url in the query set
    title: str


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------

def load_corpus():
    """All 609 documents. Never slice this -- the gold evidence lives here."""
    return load_dataset("yixuantt/MultiHopRAG", "corpus", split="train")


def load_queries(n=200, seed=0, drop_null=True):
    """
    Slice the QUERIES, not the corpus.

    drop_null=True holds back the `null_query` rows -- questions the corpus
    genuinely cannot answer. They are your Day 7 abstention test; keeping them
    out of the Day 1 slice stops them polluting the accuracy number.
    """
    qs = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG", split="train")
    nulls = qs.filter(lambda r: r["question_type"] == "null_query")
    if drop_null:
        qs = qs.filter(lambda r: r["question_type"] != "null_query")
    qs = qs.shuffle(seed=seed).select(range(min(n, len(qs))))
    return qs, nulls


# ---------------------------------------------------------------------------
# 2. Chunking -- deliberately the dumbest thing that works.
#    Day 2 replaces this. You need a bad baseline to measure against.
# ---------------------------------------------------------------------------

def chunk_document(body, url, title, size=CHUNK_WORDS, overlap=CHUNK_OVERLAP,
                   include_title=True):
    words = body.split()
    step = size - overlap
    out = []
    for i in range(0, max(len(words), 1), step):
        piece = " ".join(words[i:i + size])
        if piece.strip():
            # Prepending the title DISAMBIGUATES: a chunk saying "he denied the
            # allegations" is meaningless alone, but "Sam Bankman-Fried trial
            # begins\n\nhe denied..." encodes who "he" is.
            #
            # Side effect: every chunk from one document now shares a prefix,
            # so their vectors get pulled toward each other. Good for finding
            # the right document, potentially BAD for multi-hop, where you need
            # k slots spread over 2-4 DIFFERENT documents.
            # Run --no-title to measure this. See README.
            text = f"{title}\n\n{piece}" if include_title else piece
            out.append(Chunk(text=text, url=url, title=title))
        if i + size >= len(words):
            break
    return out


def build_chunks(corpus, include_title=True):
    chunks = []
    for doc in corpus:
        chunks.extend(chunk_document(doc["body"], doc["url"], doc["title"],
                                     include_title=include_title))
    return chunks


# ---------------------------------------------------------------------------
# 3. Index
# ---------------------------------------------------------------------------

def build_index(chunks, embedder):
    """In-memory Qdrant. No Docker needed today -- keep Day 1 frictionless."""
    client = QdrantClient(":memory:")
    dim = embedder.get_sentence_embedding_dimension()
    client.create_collection(
        COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    vectors = embedder.encode(
        [c.text for c in chunks], batch_size=64,
        show_progress_bar=True, normalize_embeddings=True,
    )
    client.upsert(
        COLLECTION,
        points=[
            PointStruct(id=i, vector=v.tolist(),
                        payload={"text": c.text, "url": c.url, "title": c.title})
            for i, (c, v) in enumerate(zip(chunks, vectors))
        ],
    )
    return client


def retrieve(client, embedder, query, k=TOP_K, max_per_doc=None, overfetch=5):
    """
    max_per_doc=None -> pure similarity. Nothing stops one document owning all
                        k slots. This is the default and it is the problem.

    max_per_doc=1    -> DIVERSITY. Over-fetch k*overfetch chunks, then walk the
                        ranked list and take at most `max_per_doc` per document
                        until you have k. Five lines, no new dependency, and it
                        is usually the single biggest all-evidence-recall win
                        available on multi-hop questions.

    Note what this is NOT: it is not reranking (nothing re-scores anything) and
    it is not a graph. It is a constraint applied to an existing ranked list.
    """
    vec = embedder.encode(query, normalize_embeddings=True)
    limit = k if max_per_doc is None else k * overfetch
    hits = client.query_points(COLLECTION, query=vec.tolist(), limit=limit).points
    ranked = [h.payload for h in hits]          # already sorted, best first

    if max_per_doc is None:
        return ranked[:k]

    seen, out = {}, []
    for p in ranked:
        if seen.get(p["url"], 0) < max_per_doc:
            out.append(p)
            seen[p["url"]] = seen.get(p["url"], 0) + 1
        if len(out) == k:
            break
    return out


# ---------------------------------------------------------------------------
# 4. Generate
# ---------------------------------------------------------------------------

PROMPT = """Answer the question using ONLY the sources below.
If the sources do not contain the answer, reply exactly: I don't know.
Answer in as few words as possible -- usually a name, number or date.

Sources:
{context}

Question: {question}
Answer:"""


def answer(llm, question, passages):
    context = "\n\n---\n\n".join(
        f"[{i+1}] {p['title']}\n{p['text']}" for i, p in enumerate(passages)
    )
    resp = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user",
                   "content": PROMPT.format(context=context, question=question)}],
        temperature=0,
    )
    # Defensive: not every OpenAI-compatible server returns a usage object,
    # and reasoning models can return None content. Don't crash on either.
    #
    # If ptok comes back 0 on your setup, the /v1 shim is not reporting usage
    # -- see the REQUIREMENT note at the top of this file and switch to
    # Ollama's /api/chat, which always reports prompt_eval_count/eval_count.
    #
    # The reasoning trace is NOT available here. /v1 has no standard field for
    # it; some servers use message.reasoning_content, most drop it silently.
    msg = (resp.choices[0].message.content or "").strip()
    usage = getattr(resp, "usage", None)
    ptok = getattr(usage, "prompt_tokens", 0) or 0
    ctok = getattr(usage, "completion_tokens", 0) or 0
    return msg, (ptok, ctok)


def make_llm():
    from openai import OpenAI
    return OpenAI(base_url=LLM_BASE_URL) if LLM_BASE_URL else OpenAI()


def make_embedder():
    return SentenceTransformer(EMBED_MODEL)