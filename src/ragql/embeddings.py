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
import logging

import numpy as np
import requests

from .config import Settings

log = logging.getLogger(__name__)


# Public helper – get_embeddings
def get_embeddings(texts: List[str], cfg: Settings) -> np.ndarray:
    """Return an array of embedding vectors for the given texts.

    Chooses the embedding backend based on `cfg.embed_provider`:
    - "ollama": calls the Ollama embeddings endpoint
    - "openai": calls the OpenAI embeddings API

    Args:
        texts (List[str]): A list of strings to embed.
        cfg (Settings): Configuration object, must have `embed_provider`.

    Returns:
        np.ndarray: Float32 array of shape (len(texts), embedding_dim).

    Raises:
        ValueError: If `cfg.embed_provider` is not "ollama" or "openai".
    """
    if cfg.embed_provider == "ollama":
        return _ollama_embed(texts, cfg)
    elif cfg.embed_provider == "openai":
        return _openai_embed(texts, cfg)
    else:
        raise ValueError(
            f"Embed provider: {cfg.embed_provider} is not valid;"
            "it should be either (ollama or openai)"
        )


# Public helpers – chat completion:
def call_ollama_chat(prompt: str, context: str, cfg: Settings) -> str:
    """Generate a chat response using an Ollama model.

    Sends a formatted prompt and context to the Ollama `/api/generate` endpoint
    and returns the model’s reply as a stripped string.

    Args:
        prompt (str): The user’s query or instruction.
        context (str): Supporting context to include in the prompt.
        cfg (Settings): Configuration object containing `ollama_url`.

    Returns:
        str: The model’s text response with leading/trailing whitespace removed.

    Raises:
        requests.HTTPError: If the HTTP request returns a non-2xx status code.
        RuntimeError: If the JSON response does not include a "response" field.
    """

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
    """Generate a chat response using the OpenAI API.

    Uses the `openai` Python package to call the chat completion endpoint
    with a formatted prompt and context, returning the assistant’s reply.

    Args:
        prompt (str): The user’s query or instruction.
        context (str): Supporting context to include in the prompt.
        cfg (Settings): Configuration object containing `openai_key`.

    Returns:
        str: The model’s text response with leading/trailing whitespace removed.

    Raises:
        RuntimeError: If the `openai` package is not installed.
        openai.error.OpenAIError: If the API call fails for any reason.
        RuntimeError: If the API returns no choices or an empty response.
    """

    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "`openai` package not installed: pip install openai"
        ) from exc

    client = OpenAI(api_key=cfg.openai_key)
    rs = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": _format_prompt(prompt, context)}],
        temperature=0,
        max_tokens=256,
    )

    # Extract the content of the first choice
    try:
        content = rs.choices[0].message.content
    except (AttributeError, IndexError) as exc:
        raise RuntimeError("OpenAI chat returned no response choices") from exc

    if not content:
        raise RuntimeError("OpenAI chat returned empty content")

    return content.strip()


# Internal helpers:
def _ollama_embed(texts: Iterable[str], cfg: Settings) -> np.ndarray:
    """Generate embeddings for each text prompt using an Ollama server.

    Sends each prompt in `texts` individually to the Ollama embeddings endpoint
    specified by `cfg.ollama_url` and returns a NumPy array of embedding vectors.

    Args:
        texts (Iterable[str]): A sequence of text strings to embed.
        cfg (Settings): Configuration object containing `ollama_url`
            and the selected `embed_model_name`.

    Returns:
        np.ndarray: An array of shape (len(texts), embedding_dim) with dtype float32.

    Raises:
        requests.HTTPError: If the Ollama API returns a non-2xx status code.
        RuntimeError: If the response JSON does not contain an "embedding" field.
    """
    vecs = []
    for prompt in texts:  # Ollama v0.1.x only supports single-prompt payloads
        r = requests.post(
            f"{cfg.ollama_url}/api/embeddings",
            json={"model": cfg.embed_model_name, "prompt": prompt},
            timeout=60,
        )

        r.raise_for_status()
        js = r.json()

        if "embedding" not in js:
            raise RuntimeError(f"Ollama embed error → {js.get('error', js)}")

        vecs.append(js["embedding"])

    return np.array(vecs, dtype="float32")


def _openai_embed(texts: Iterable[str], cfg: Settings) -> np.ndarray:
    """Generate embeddings for a batch of texts using the OpenAI API.

    Uses the `openai` Python package to call the embeddings endpoint with
    the model specified in `cfg.embed_model_name`.

    Args:
        texts (Iterable[str]): A sequence of text strings to embed.
        cfg (Settings): Configuration object containing `openai_key`
            and the selected `embed_model_name`.

    Returns:
        np.ndarray: An array of shape (len(texts), embedding_dim) with dtype float32.

    Raises:
        RuntimeError: If the `openai` package is not installed.
        openai.error.OpenAIError: If the API call fails for any reason.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "`openai` package not installed: pip install openai"
        ) from exc

    client = OpenAI(api_key=cfg.openai_key)

    res = client.embeddings.create(
        model=cfg.embed_model_name,
        input=list(texts),
    )

    return np.array([d.embedding for d in res.data], dtype="float32")


def _format_prompt(question: str, context: str) -> str:
    """Build and log a prompt for LogGPT using the provided context.

    This helper constructs a prompt that instructs LogGPT to answer the
    given `question` using *only* the supplied `context`. It then prints
    the prompt (for debugging purposes) before returning it.

    Args:
        question (str): The user’s question to be answered.
        context (str): Relevant context that the model may reference.

    Returns:
        str: The fully formatted prompt string ready for the LLM.
    """
    ppt = (
        "You are LogGPT. Using *only* the context below, answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )
    print(ppt)
    return ppt
