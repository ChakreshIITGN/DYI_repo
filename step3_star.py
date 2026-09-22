"""
STEP 3 -- five agents in a star, one hidden message, traced in Langfuse.

    python step3_star.py
    python step3_star.py --secret SOS --carrier "one two three four"
    python step3_star.py --spokes glm-5.3,gpt-oss:20b,gemma4:31b,kimi-k2.6

    HUB broadcasts -> spoke1..4 (different models) -> HUB aggregates

The hub acts twice, so five agents. The answer is known exactly, which is what
makes this an experiment rather than a demo: every agent is scored against
ground truth.
"""

import argparse
import json
import re
from pathlib import Path

import config
import morse
from llm import ask
from tracing import make_tracer

config.require("OLLAMA_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")

DEFAULT_SPOKES = "deepseek-v4.1-flash,gpt-oss:20b,nemotron-3-nano:30b,gemma4:31b"

SPOKE_PROMPT = """A single word is hidden inside the message below.
Nothing has been removed from it; everything you need is in what you see.
 
The message, exactly as sent:
 
```
{message}
```
 
Work out the hidden word. Finish your reply with a line in exactly this form:
ANSWER: <the word>"""

HUB_PROMPT = """Four workers were each asked to decode the same hidden message.
Their answers:

{answers}

State the consensus answer, and note any disagreement in one sentence.
Finish with a line in exactly this form:
ANSWER: <the word>"""


def extract_answer(text):
    """Spokes are told to end with 'ANSWER: <word>'. Pull it out."""
    m = re.search(r"ANSWER:\s*([A-Za-z0-9]+)", text or "", re.IGNORECASE)
    return m.group(1).upper() if m else ""


def log_message(log, frm, to, kind, model, content,
                thinking="", tokens_in=0, tokens_out=0, latency_ms=0.0):
    """One row in the inter-agent message log."""
    log.append({"seq": len(log) + 1, "frm": frm, "to": to, "kind": kind,
                "model": model, "content": content, "thinking": thinking,
                "tokens_in": tokens_in, "tokens_out": tokens_out,
                "latency_ms": latency_ms})


def run_spoke(name, model, prompt, secret, tracer, log):
    """One spoke: ask, score, log, trace. Returns a result row."""
    print(f"\n[{name}] {model} ...", flush=True)
    try:
        out = ask(prompt, model=model)
    except Exception as e:
        print(f"    FAILED: {type(e).__name__}: {e}")
        return {"agent": name, "model": model, "answer": "", "correct": False,
                "tokens": 0, "latency_ms": 0.0, "error": str(e)}

    answer = extract_answer(out["content"])
    correct = answer == secret
    print(f"    answer={answer or '(none)'}  "
          f"{'CORRECT' if correct else 'WRONG'}  "
          f"{out['tokens_in']}in/{out['tokens_out']}out  {out['latency_ms']:.0f}ms")

    log_message(log, name, "HUB", "result", model, out["content"],
                out["thinking"], out["tokens_in"], out["tokens_out"],
                out["latency_ms"])

    tracer.record(name=name, model=model, prompt=prompt, output=out["content"],
                  tokens_in=out["tokens_in"], tokens_out=out["tokens_out"],
                  metadata={"answer": answer, "correct": correct,
                            "thinking": out["thinking"][:2000]})

    return {"agent": name, "model": model, "answer": answer, "correct": correct,
            "tokens": out["tokens_in"] + out["tokens_out"],
            "latency_ms": out["latency_ms"]}


def run_hub(model, results, secret, tracer, log):
    """The hub reads the four answers and calls a consensus."""
    answers = "\n".join(f"- {r['agent']} said: {r['answer'] or '(no answer)'}"
                        for r in results)
    prompt = HUB_PROMPT.format(answers=answers)

    print(f"\n[HUB] {model} aggregating ...", flush=True)
    out = ask(prompt, model=model)
    answer = extract_answer(out["content"])
    correct = answer == secret

    log_message(log, "HUB", "*", "verdict", model, out["content"],
                out["thinking"], out["tokens_in"], out["tokens_out"],
                out["latency_ms"])

    tracer.record(name="hub-aggregate", model=model, prompt=prompt,
                  output=out["content"], tokens_in=out["tokens_in"],
                  tokens_out=out["tokens_out"],
                  metadata={"answer": answer, "correct": correct})

    return {"agent": "HUB", "model": model, "answer": answer, "correct": correct,
            "tokens": out["tokens_in"] + out["tokens_out"],
            "latency_ms": out["latency_ms"]}


def print_scoreboard(rows, secret, log):
    print("\n" + "=" * 74)
    print(f"{'agent':<9}{'model':<26}{'answer':<10}{'ok':<5}{'tokens':<9}{'ms':>7}")
    print("=" * 74)
    for r in rows:
        print(f"{r['agent']:<9}{r['model']:<26}{r['answer'] or '-':<10}"
              f"{'Y' if r['correct'] else 'n':<5}{r['tokens']:<9}"
              f"{r['latency_ms']:>7.0f}")
    print("-" * 74)

    spokes = [r for r in rows if r["agent"] != "HUB"]
    hub = next(r for r in rows if r["agent"] == "HUB")
    print(f"ground truth    : {secret}")
    print(f"spokes correct  : {sum(r['correct'] for r in spokes)}/{len(spokes)}")
    print(f"hub verdict     : {hub['answer'] or '(none)'} "
          f"({'matches' if hub['correct'] else 'DOES NOT match'})")
    print(f"total tokens    : {sum(r['tokens'] for r in rows)}")
    print(f"total latency   : {sum(r['latency_ms'] for r in rows):.0f} ms")
    print(f"messages logged : {len(log)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secret", default="YES")
    ap.add_argument("--carrier", default="this is a secret")
    ap.add_argument("--hub", default="glm-5.3")
    ap.add_argument("--spokes", default=DEFAULT_SPOKES)
    args = ap.parse_args()

    secret = args.secret.upper()
    spoke_models = [s.strip() for s in args.spokes.split(",") if s.strip()]

    # Build the message. Pure python -- no model involved in the encoding.
    message = morse.encode(args.carrier, secret)
    check, gaps = morse.decode(message)

    print(f"secret    : {secret}")
    print(f"carrier   : {args.carrier!r}        <- plain words you supply")
    print(f"encoded   : {message!r}   <- what the agents actually see")
    print(f"gaps      : {gaps}")
    print(f"selfcheck : {check} {'OK' if check == secret else 'MISMATCH'}")
    print(f"hub       : {args.hub}")
    print(f"spokes    : {', '.join(spoke_models)}")

    tracer = make_tracer()
    log = []
    prompt = SPOKE_PROMPT.format(rules=morse.RULES, message=message)

    with tracer.trace("star-run"):
        # hub -> everyone. Routing is not a reasoning task, so no LLM call.
        for i in range(1, len(spoke_models) + 1):
            log_message(log, "HUB", f"spoke{i}", "broadcast", "-", message)

        rows = [run_spoke(f"spoke{i}", m, prompt, secret, tracer, log)
                for i, m in enumerate(spoke_models, 1)]
        rows.append(run_hub(args.hub, rows, secret, tracer, log))

    tracer.flush()

    print_scoreboard(rows, secret, log)

    Path("runs").mkdir(exist_ok=True)
    out_path = Path("runs/step3_star.json")
    out_path.write_text(json.dumps(
        {"secret": secret, "carrier": args.carrier, "encoded": message,
         "results": rows, "messages": log}, indent=2))
    print(f"\nwrote {out_path}")
    print("trace sent to Langfuse -- open your project to see the tree")


if __name__ == "__main__":
    main()