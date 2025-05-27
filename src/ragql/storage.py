"""
ragql.storage
~~~~~~~~~~~~~
SQLite + Faiss helpers for RagQL.
"""

from __future__ import annotations

import sqlite3
import logging
from hashlib import md5
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np

logger = logging.getLogger(__name__)


# SQLite helpers
def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")  # better concurrency
    return conn


# ChunkStore – text & metadata
class ChunkStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = _connect(db_path)
        self._ensure_schema()
        logger = logging.getLogger(__name__)
        logger.debug("ChunkStore init")

    # ---------- public --------------------------------------------------

    @staticmethod
    def make_hash(doc_id: str, idx: int) -> str:
        logger.debug("Creating hash for the document")
        return md5(f"{doc_id}:{idx}".encode()).hexdigest()

    def add(self, h: str, file: str, start: int, text: str) -> None:
        logger.debug("Adding data to the ChunkStoree")
        self.conn.execute(
            "INSERT OR IGNORE INTO chunks VALUES (?,?,?,?)",
            (h, file, start, text),
        )
        self.conn.commit()

    def build_context(self, hits: list[tuple[str, float]], max_len: int = 140) -> str:
        """Turn [(hash, score), …] into a multi-line context string."""
        logger.debug("Building context:")
        cur = self.conn.execute(
            "SELECT file, text FROM chunks WHERE hash IN (%s)"
            % ",".join("?" * len(hits)),
            [h for h, _ in hits],
        )
        rows = {h: (f, t) for (f, t), (h, _) in zip(cur.fetchall(), hits)}
        lines = [
            f"[{rows[h][0]}] {rows[h][1][:max_len]} … (sim {score:.2f})"
            for h, score in hits
        ]
        context = "\n".join(lines)
        logger.debug(context)
        logger.debug("Context built")
        return context

    # ---------- private -------------------------------------------------

    def _ensure_schema(self) -> None:
        logger.debug("Ensuring schema")
        cur = self.conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS chunks("
            "hash TEXT PRIMARY KEY, "
            "file TEXT, "
            "start INT, "
            "text TEXT)"
        )
        self.conn.commit()


# VectorStore – blobs & Faiss index
class VectorStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = _connect(db_path)
        self._ensure_schema()
        self.index: faiss.Index | None = None
        self._hashes: list[str] = []

    # ---------- public --------------------------------------------------

    def has_vector(self, h: str) -> bool:
        condition = (
            self.conn.execute("SELECT 1 FROM vectors WHERE hash=?", (h,)).fetchone()
            is not None
        )

        if condition:
            logger.debug("The vector exists!")
        else:
            logger.debug("The vector doesn't exists!")

        return condition

    def add_vectors(self, ids: list[str], vecs: np.ndarray) -> None:
        """
        ids : list of md5 strings
        vecs: (N, dim) float32
        """
        logger.debug("Trying to add an vector")
        cur = self.conn.cursor()
        for h, v in zip(ids, vecs):
            cur.execute("INSERT OR REPLACE INTO vectors VALUES (?,?)", (h, v.tobytes()))
        self.conn.commit()

    # - - - Faiss --------------------------------------------------------

    def load_faiss(self) -> None:
        logger.debug("Loading faiss")
        cur = self.conn.execute("SELECT hash, vec FROM vectors")
        rows = cur.fetchall()
        if not rows:
            logger.error("Faiss loaded successfully")
            raise RuntimeError("VectorStore is empty")

        mat = np.vstack([np.frombuffer(v, dtype="float32") for _, v in rows])
        faiss.normalize_L2(mat)
        index = faiss.IndexFlatIP(mat.shape[1])
        index.add(x=mat)

        self.index = index
        self._hashes = [h for h, _ in rows]
        logger.debug("Faiss loaded successfully")

    def search(
        self, qvec: np.ndarray, top_k: int = 6
    ) -> list[tuple[str, float]]:  # [(hash, score)]
        logger.debug(f"Searching top_k: {top_k} embeddings")
        logger.debug(f"{qvec}")

        if self.index is None:
            logger.error("Faiss index not loaded; call load_faiss() first")
            raise RuntimeError("Faiss index not loaded; call load_faiss() first")

        qvec = qvec.astype("float32").reshape(1, -1)
        faiss.normalize_L2(qvec)

        distancies, indicies = self.index.search(x=qvec, k=top_k)

        logger.debug(
            f"Results of the embeddings search, distancies: {distancies}, indicies: {indicies}"
        )

        return [
            (self._hashes[i], float(distancies[0][rank]))
            for rank, i in enumerate(indicies[0])
            if i != -1
        ]

    # ---------- private -------------------------------------------------

    def _ensure_schema(self) -> None:
        logger.debug("Ensuring VectorStore schema")
        cur = self.conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS vectors(hash TEXT PRIMARY KEY, vec  BLOB)"
        )
        self.conn.commit()


# Convenience: build everything in one call (optional helper)
def ingest_vectors(
    chunk_store: ChunkStore,
    vec_store: VectorStore,
    docs: Iterable[tuple[str, str]],
    chunker,
    embed_fn,
) -> None:
    """
    High-level helper: feed `(doc_id, text)` pairs → store chunks + vectors.

    • `chunker(text)` yields chunks.
    • `embed_fn(list[str])` returns np.ndarray of embeddings.
    """

    logger.debug("Ingesting vectors")

    new_ids, new_chunks = [], []

    # pass 1 – collect new chunks
    for doc_id, text in docs:
        for idx, chunk in enumerate(chunker(text)):
            h = chunk_store.make_hash(doc_id, idx)
            if not vec_store.has_vector(h):
                new_ids.append(h)
                new_chunks.append(chunk)
                chunk_store.add(h, doc_id, idx, chunk)

    # pass 2 – embed & store
    if new_chunks:
        vecs = embed_fn(new_chunks)
        vec_store.add_vectors(new_ids, vecs)
