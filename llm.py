"""
The agent primitive. This is step 1's function with `model` promoted to a
parameter, so different agents can run on different models.

That is the whole difference between "an LLM call" and "an agent": a name,
a system prompt, a model, and somewhere to put the result.
"""

import time

import httpx

import config


def ask(prompt, model, system=None, think=True, timeout=240):
    """
    One call to Ollama's native /api/chat.

    Returns a dict -- not a bare string -- because the interesting parts are
    the thinking trace and the token counts, and a string throws those away.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    t0 = time.perf_counter()
    r = httpx.post(
        f"{config.OLLAMA_HOST}/api/chat",
        headers={"Authorization": f"Bearer {config.OLLAMA_API_KEY}"},
        json={"model": model, "messages": messages,
              "stream": False, "think": think},
        timeout=timeout,
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    data = r.json()
    msg = data.get("message", {})

    return {
        "content": (msg.get("content") or "").strip(),
        "thinking": (msg.get("thinking") or "").strip(),
        "model": model,
        "tokens_in": data.get("prompt_eval_count", 0),
        "tokens_out": data.get("eval_count", 0),
        "latency_ms": round(latency_ms, 1),
    }


def list_cloud_models():
    """
    Ask your account what it can actually run, instead of hardcoding tags
    that may not exist. Returns a list of names, or [] if the endpoint
    does not answer.
    """
    try:
        r = httpx.get(
            f"{config.OLLAMA_HOST}/api/tags",
            headers={"Authorization": f"Bearer {config.OLLAMA_API_KEY}"},
            timeout=30,
        )
        r.raise_for_status()
        return sorted(m.get("name", "") for m in r.json().get("models", []))
    except Exception as e:
        print(f"  (could not list models: {type(e).__name__}: {e})")
        return []