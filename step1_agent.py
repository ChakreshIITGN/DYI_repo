"""
STEP 1 -- one agent, one call.

Goal: prove you can get back the three things you said you need.
  1. the answer
  2. the CHAIN OF THOUGHT
  3. TOKEN COUNTS

Nothing else. No Langfuse, no second agent, no network.

    uv pip install httpx python-dotenv
    python step1_one_agent.py
    python step1_one_agent.py --raw     # dump the whole JSON response

Why raw HTTP and not the ollama SDK: you should see the actual request and
response once. Every framework you touch later is a wrapper over exactly this.
"""

import argparse
import json

import httpx

import config

config.require("OLLAMA_API_KEY")


def ask(prompt, system=None, think=True, timeout=180):
    """
    One call to Ollama's NATIVE /api/chat endpoint.

    Native, not /v1/chat/completions, for one reason: the OpenAI-compatible
    shim has no field for a reasoning trace. /api/chat does.

    Returns (content, thinking, tokens_in, tokens_out, raw).
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "stream": False,   # one JSON object back, not a token stream
        "think": think,    # <- this is what produces message.thinking
    }

    r = httpx.post(
        f"{config.OLLAMA_HOST}/api/chat",
        headers={"Authorization": f"Bearer {config.OLLAMA_API_KEY}"},
        json=payload,
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()

    msg = data.get("message", {})
    return (
        msg.get("content", ""),
        msg.get("thinking", ""),          # empty if the model has no trace
        data.get("prompt_eval_count", 0),
        data.get("eval_count", 0),
        data,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", action="store_true", help="print the full JSON")
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--prompt",
                    default="A message hides a letter in its spacing. "
                            "In one sentence: how would you start decoding it?")
    args = ap.parse_args()

    print(f"model : {config.LLM_MODEL}")
    print(f"host  : {config.OLLAMA_HOST}")
    print(f"think : {not args.no_think}\n")

    content, thinking, tin, tout, raw = ask(args.prompt, think=not args.no_think)

    print("-" * 68)
    print("THINKING (the chain of thought)")
    print("-" * 68)
    print(thinking or "(empty -- model returned no trace)")

    print("\n" + "-" * 68)
    print("CONTENT (the answer)")
    print("-" * 68)
    print(content or "(empty)")

    print("\n" + "-" * 68)
    print("TOKENS")
    print("-" * 68)
    print(f"  in  : {tin}")
    print(f"  out : {tout}")
    if tin == 0:
        print("  !! zero -- the server is not reporting counts. Check --raw.")

    if args.raw:
        print("\n" + "-" * 68)
        print("RAW RESPONSE KEYS")
        print("-" * 68)
        for k in raw:
            v = raw[k]
            print(f"  {k}: {json.dumps(v)[:90] if not isinstance(v, dict) else list(v)}")


if __name__ == "__main__":
    main()