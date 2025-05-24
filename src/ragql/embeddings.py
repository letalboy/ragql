"""
ragql.embeddings
~~~~~~~~~~~~~~~~
*  Generates embeddings via Ollama **or** OpenAI.
*  Provides a thin wrapper to call an LLM chat endpoint for final answer generation.
*  All behaviour is driven by `Settings` so the rest of the library never
   reads env-vars directly.
"""

from __future__ import annotations

from typing import Iterable, List

import json
import logging
import os
from dataclasses import asdict

import numpy as np
import requests

from .config import Settings

log = logging.getLogger(__name__)


# Public helper – get_embeddings
def get_embeddings(texts: List[str], cfg: Settings) -> np.ndarray:
    """
    Return an (N, D) float32 ndarray of embeddings.

    * If `cfg.use_ollama` → POST /api/embeddings to Ollama.
    * else → call OpenAI's `text-embedding-3-small`.
    """
    if cfg.use_ollama:
        return _ollama_embed(texts, cfg)
    return _openai_embed(texts, cfg)

# Public helpers – chat completion:

def call_ollama_chat(prompt: str, context: str, cfg: Settings) -> str:
    payload = {
        "model": "mistral:7b-instruct",
        "prompt": _format_prompt(prompt, context),
        "stream": False,
        "options": {"temperature": 0},
    }
    r = requests.post(f"{cfg.ollama_url}/api/generate", json=payload, timeout=120)
    r.raise_for_status()
    js = r.json()
    if "response" not in js:
        raise RuntimeError(f"Ollama chat error → {js.get('error', js)}")
    return js["response"].strip()


def call_openai_chat(prompt: str, context: str, cfg: Settings) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("`openai` package not installed: pip install openai") from exc

    client = OpenAI(api_key=cfg.openai_key)
    rs = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": _format_prompt(prompt, context)}],
        temperature=0,
        max_tokens=256,
    )
    return rs.choices[0].message.content.strip()


# Internal helpers:

def _ollama_embed(texts: Iterable[str], cfg: Settings) -> np.ndarray:
    vecs = []
    for prompt in texts:  # Ollama v0.1.x only supports single-prompt payloads
        r = requests.post(
            f"{cfg.ollama_url}/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": prompt},
            timeout=60,
        )
        r.raise_for_status()
        js = r.json()
        if "embedding" not in js:
            raise RuntimeError(f"Ollama embed error → {js.get('error', js)}")
        vecs.append(js["embedding"])
    return np.array(vecs, dtype="float32")


def _openai_embed(texts: Iterable[str], cfg: Settings) -> np.ndarray:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("`openai` package not installed: pip install openai") from exc

    client = OpenAI(api_key=cfg.openai_key)
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=list(texts),
    )
    return np.array([d.embedding for d in res.data], dtype="float32")


def _format_prompt(question: str, context: str) -> str:
    return (
        "You are LogGPT. Using *only* the context below, answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )
