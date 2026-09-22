"""
STEP 2 -- two agents, two models, one task-and-response exchange.

    python step2_two_agents.py --list          # what models can you run?
    python step2_two_agents.py
    python step2_two_agents.py --planner gpt-oss:120b-cloud \
                               --worker deepseek-v4-flash:cloud

The task is deliberately trivial. The thing under test here is the EXCHANGE,
not the puzzle -- who sent what to whom, on which model, at what cost. The
morse puzzle comes next, once this part is boring.

Three calls happen:
    PLANNER -> writes a task instruction
    WORKER  -> does the task
    PLANNER -> reviews the worker's answer

Everything is appended to a transcript. That transcript is the thing you will
compare against Langfuse's trace view in step 3.
"""

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import config
from llm import ask, list_cloud_models

config.require("OLLAMA_API_KEY")


# ---------------------------------------------------------------------------
# An agent is a name + a model + a system prompt. That is all.
# ---------------------------------------------------------------------------

@dataclass
class Agent:
    name: str
    model: str
    system: str

    def act(self, prompt, log, to, kind):
        out = ask(prompt, model=self.model, system=self.system)
        log.add(frm=self.name, to=to, kind=kind, content=out["content"],
                thinking=out["thinking"], model=out["model"],
                tokens_in=out["tokens_in"], tokens_out=out["tokens_out"],
                latency_ms=out["latency_ms"])
        return out["content"]


# ---------------------------------------------------------------------------
# The message log. One row per message between agents.
#
# Worth building by hand: a trace tree tells you which function called which,
# but not who said what to whom. With five agents that difference matters.
# ---------------------------------------------------------------------------

@dataclass
class Transcript:
    messages: list = field(default_factory=list)

    def add(self, **kw):
        kw["seq"] = len(self.messages) + 1
        self.messages.append(kw)

    def show(self):
        for m in self.messages:
            print("\n" + "=" * 70)
            print(f"[{m['seq']}] {m['frm']} -> {m['to']}   ({m['kind']})")
            print(f"     model {m['model']} | "
                  f"tokens {m['tokens_in']}in/{m['tokens_out']}out | "
                  f"{m['latency_ms']:.0f} ms")
            print("=" * 70)
            if m["thinking"]:
                print("--- thinking ---")
                print(m["thinking"][:600])
                print("--- says ---")
            print(m["content"])

    def totals(self):
        return {
            "messages": len(self.messages),
            "tokens_in": sum(m["tokens_in"] for m in self.messages),
            "tokens_out": sum(m["tokens_out"] for m in self.messages),
            "latency_ms": round(sum(m["latency_ms"] for m in self.messages), 1),
            "models": sorted({m["model"] for m in self.messages}),
        }


# ---------------------------------------------------------------------------

TOPIC = "a string where the gaps between words hide a letter"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true",
                    help="list cloud models your key can run, then exit")
    ap.add_argument("--planner", default=config.LLM_MODEL)
    ap.add_argument("--worker", default=config.LLM_MODEL)
    ap.add_argument("--topic", default=TOPIC)
    args = ap.parse_args()

    if args.list:
        print("cloud models available to your key:")
        names = list_cloud_models()
        for n in names:
            print(f"  {n}")
        if not names:
            print("  none returned -- check https://ollama.com/search?c=cloud")
        return

    if args.planner == args.worker:
        print(f"NOTE: both agents on {args.planner}. Pass --planner/--worker "
              f"different tags to see model diversity.\n")

    planner = Agent(
        name="PLANNER",
        model=args.planner,
        system="You coordinate one worker. You are terse. "
               "You never solve the task yourself.",
    )
    worker = Agent(
        name="WORKER",
        model=args.worker,
        system="You carry out exactly the task you are given. "
               "You are concise and concrete.",
    )

    log = Transcript()

    # 1. planner writes the task
    task = planner.act(
        f"Write ONE short instruction, max 30 words, telling a worker how to "
        f"approach this problem: {args.topic}. Output only the instruction.",
        log, to="WORKER", kind="assign")

    # 2. worker does it
    result = worker.act(
        f"Your task: {task}\n\nCarry it out in under 80 words.",
        log, to="PLANNER", kind="result")

    # 3. planner reviews
    planner.act(
        f"You asked: {task}\n\nThe worker replied:\n{result}\n\n"
        f"Reply with ACCEPT or REVISE and one sentence of reason.",
        log, to="WORKER", kind="review")

    log.show()

    print("\n" + "=" * 70)
    print("TOTALS")
    print("=" * 70)
    for k, v in log.totals().items():
        print(f"  {k}: {v}")

    Path("runs").mkdir(exist_ok=True)
    out = Path("runs/step2_transcript.json")
    out.write_text(json.dumps(
        {"totals": log.totals(), "messages": log.messages}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()