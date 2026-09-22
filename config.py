"""
The only place configuration lives.

load_dotenv() reads ./.env and pushes the values into os.environ. It does not
replace os -- it feeds it. Everything downstream just does `from config import X`.
"""

import os

from dotenv import load_dotenv

load_dotenv()

OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "https://ollama.com")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-oss:120b-cloud")

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")


_SETTINGS = {
    "OLLAMA_API_KEY": OLLAMA_API_KEY,
    "OLLAMA_HOST": OLLAMA_HOST,
    "LLM_MODEL": LLM_MODEL,
    "LANGFUSE_PUBLIC_KEY": LANGFUSE_PUBLIC_KEY,
    "LANGFUSE_SECRET_KEY": LANGFUSE_SECRET_KEY,
    "LANGFUSE_HOST": LANGFUSE_HOST,
}

def require(*names):
    unknown = [n for n in names if n not in _SETTINGS]
    if unknown:
        raise SystemExit(f"not a setting: {', '.join(unknown)}")
    missing = [n for n in names if not _SETTINGS[n]]
    if missing:
        raise SystemExit(f"missing in .env: {', '.join(missing)}\n"
                         f"Copy .env.example to .env and fill those in.")