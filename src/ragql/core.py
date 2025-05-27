"""
ragql.core
~~~~~~~~~~
High-level façade that ties together loaders, embeddings,
vector storage, and the answering engine.
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable, Iterator

from .config import Settings
from .loaders import REGISTRY
from .loaders import Doc
from .storage import VectorStore, ChunkStore
import logging


class RagQL:
    """
    Orchestrates end-to-end retrieval-augmented Q&A.

    Typical use:
        >>> rq = RagQL(Path("my/logs"), Settings())
        >>> rq.build()                 # embed + index
        >>> answer = rq.query("Why ...?")
    """

    def __init__(self, root: Path, cfg: Settings):
        self.root = root
        self.cfg = cfg
        self.chunks = ChunkStore(cfg.db_path)
        self.vstore = VectorStore(cfg.db_path)
        self.logger = logging.getLogger(__name__)
        self.logger.debug(f"RagQL init: source={root}, verbose={cfg.verbose}")

    # Indexing:

    def _iter_documents(self) -> Iterator[Doc]:
        """
        Walk every file under self.root, hand it off to each loader in REGISTRY,
        and yield back any (doc_id, text) tuples they emit.
        """
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue

            for load in REGISTRY:  # now `load` is Callable[[Path], Iterable[Doc]]
                try:
                    docs = load(path)  # call the loader function directly
                except Exception:
                    # loader didn’t like this file, skip it
                    continue

                if not docs:
                    # loader returned empty or None, skip
                    continue

                # finally, yield each (doc_id, text)
                for doc in docs:
                    yield doc

    def build(self) -> None:
        from .embeddings import get_embeddings

        new_texts, new_ids = [], []
        for doc_id, text in self._iter_documents():
            for idx, chunk in enumerate(self._chunk(text)):
                h = self.chunks.make_hash(doc_id, idx)
                if not self.vstore.has_vector(h):
                    new_ids.append(h)
                    new_texts.append(chunk)
                    self.chunks.add(h, doc_id, idx, chunk, self.cfg.embed_model_name)

        if new_texts:
            vecs = get_embeddings(new_texts, self.cfg)
            self.vstore.add_vectors(new_ids, vecs)

        self.vstore.load_faiss()  # ready for search

    # Querying:

    def query(self, prompt: str, top_k: int = 6) -> str:
        from .embeddings import get_embeddings  # <- lazy import

        vec = get_embeddings([prompt], self.cfg)
        hits = self.vstore.search(vec, top_k)
        context = self.chunks.build_context(hits)

        return self._call_llm(prompt, context)

    # Helpers:

    def _chunk(self, text: str) -> Iterable[str]:
        words = text.split()
        step = self.cfg.chunk_size - self.cfg.chunk_overlap
        for i in range(0, len(words), step):
            yield " ".join(words[i : i + self.cfg.chunk_size])

    def _call_llm(self, prompt: str, context: str) -> str:
        # defer heavy import
        if self.cfg.use_ollama:
            from .embeddings import call_ollama_chat

            return call_ollama_chat(prompt, context, self.cfg)
        else:
            from .embeddings import call_openai_chat

            return call_openai_chat(prompt, context, self.cfg)
