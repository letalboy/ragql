#!/usr/bin/env python3
"""
Minimal local-first RAG chat for log-files.

•  Reads every text-like file in a folder (recursive).
•  Splits each file into ~800-token chunks.
•  Generates embeddings via Ollama (or OpenAI if OLLAMA_URL is unset).
•  Stores vectors in RAM (faiss) + a SQLite metadata cache.
•  Opens a tiny REPL:

    $ python rag_chat.py path/to/logs
    >> Is "Acquired Python in a process task!" present?
    << Found in 2 files (see excerpts below) …

Dependencies: pip install faiss-cpu sentencepiece pandas requests tqdm tabulate
"""

import os
import json
import re
import sqlite3
import pathlib
import requests
import textwrap
import hashlib
from hashlib import md5
from tqdm import tqdm
import faiss
import numpy as np
import argparse
from pathlib import Path
from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:
    print("OpenAI library is not installed. Please install it to use OpenAI features.")

# ─── load your .env file in the same folder as rag_chat.py ───
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

# ─── now read it into a Python variable ───
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY in .env")

# ----------------- config ---------------- #
CHUNK_SIZE_TOK  = 800          # ≈ 600–900 chars for code/logs
CHUNK_OVERLAP   = 80
EMBED_MODEL     = "nomic-embed-text"
USE_OPENAI_GEN = True
USE_OPENAI_EMBED = False

OLLAMA_URL      = os.getenv("OLLAMA_URL", "http://localhost:11434")
# OPENAI_KEY      = os.getenv("OPENAI_API_KEY")             # fallback (Loaded automatically by .env)

from pathlib import Path
DB_PATH = Path(__file__).parent.resolve() / ".rag_chat.db"

CONFIG_FILE = "config.json"

def load_config():
    """Load configuration from a JSON file."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {"line_spacing": 1, "response_color": "default"}

def save_config(config):
    """Save configuration to a JSON file."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f)

def config_menu():
    config = load_config()
    while True:
        print("\nConfiguration Menu:")
        print("1. Customize Line Spacing")
        print("2. Customize Response Color")
        print("3. Save and Exit")
        print("4. Exit without Saving")
        choice = input("Choose an option: ")

        if choice == "1":
            spacing = input("Enter line spacing (e.g., 1, 2, 3): ")
            config["line_spacing"] = int(spacing)
        elif choice == "2":
            color = input("Enter response color (e.g., 'red', 'green', 'blue', 'default'): ")
            config["response_color"] = color
        elif choice == "3":
            save_config(config)
            print("Configuration saved.")
            break
        elif choice == "4":
            print("Exiting without saving.")
            break
        else:
            print("Invalid choice. Please try again.")

# ----------------------------------------- #

def iter_files(root: pathlib.Path):
    exts = {".txt", ".log", ".md", ".rst"}
    for p in root.rglob("*"):
        if p.suffix.lower() in exts and p.is_file():
            yield p

def chunk_text(text: str):
    words = text.split()
    step = CHUNK_SIZE_TOK - CHUNK_OVERLAP
    for i in range(0, len(words), step):
        yield " ".join(words[i:i+CHUNK_SIZE_TOK])

def embed(texts: list[str]) -> np.ndarray:
    """
    Return an (N, D) float32 ndarray.

    • If USE_OPENAI_GEN is False and OLLAMA_URL is set → use Ollama embeddings
    • Else → OpenAI text-embedding-3-small
    """
    if OLLAMA_URL and not USE_OPENAI_EMBED:
        vecs = []
        for prompt in texts:
            r = requests.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": prompt},
                timeout=60,
            )
            js = r.json()
            if "embedding" not in js:
                raise RuntimeError(f"Ollama embed error: {js.get('error', js)}")
            vecs.append(js["embedding"])
        return np.array(vecs, dtype="float32")

    # --- OpenAI fallback ---
    client = OpenAI(api_key=OPENAI_KEY)
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return np.array([d.embedding for d in res.data], dtype="float32")

def ensure_db():
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("create table if not exists chunks"
                "(hash text primary key, file text, start int, text text)")
    cur.execute("create table if not exists vectors(hash text primary key, vec blob)")
    conn.commit()
    return conn

# >>> Registry your loaders here:

from loaders import REGISTRY 

def collect_documents(sources: list[Path]):
    for src in sources:
        p = src.resolve()
        candidates = [p]                       # default: just itself
        if p.is_dir():                        # directory → recurse one level
            candidates.extend(p.iterdir())    # or p.rglob("*.db") for deep scan

        matched = False
        for path in candidates:
            for loader in REGISTRY:
                for doc_id, text in loader(path):
                    matched = True
                    yield doc_id, text
        if not matched:
            print(f"⚠️  no loader handled {src}")

def ensure_db():
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS chunks"
                "(hash TEXT PRIMARY KEY, file TEXT, start INT, text TEXT)")
    cur.execute("CREATE TABLE IF NOT EXISTS vectors(hash TEXT PRIMARY KEY, vec BLOB)")
    conn.commit()
    return conn


def build_index(sources: list[pathlib.Path]):
    """
    Build or update the FAISS index from *all* supplied sources.
    Returns (sqlite_connection, faiss_index, [hash list])
    """
    conn = ensure_db()
    cur  = conn.cursor()
    vecs, ids = [], []

    print("🔍 scanning sources …")
    for doc_id, full_text in collect_documents(sources):
        for idx, chunk in enumerate(chunk_text(full_text)):
            h = hashlib.md5(f"{doc_id}:{idx}".encode()).hexdigest()
            cur.execute("SELECT 1 FROM vectors WHERE hash=?", (h,))
            if cur.fetchone():           # embedding cached
                continue
            ids.append(h)
            vecs.append(chunk)
            cur.execute("INSERT OR IGNORE INTO chunks VALUES (?,?,?,?)",
                        (h, doc_id, idx, chunk))

    if vecs:                             # only embed NEW stuff
        print(f"🧠 embedding {len(vecs)} new chunks …")
        emb = embed(vecs)                # (N, D) float32
        for h, v in zip(ids, emb):
            cur.execute("INSERT OR IGNORE INTO vectors VALUES (?,?)",
                        (h, v.tobytes()))
        conn.commit()
        print("✅ cache updated.")

    # --- load *all* vectors into Faiss -------------------------------
    cur.execute("SELECT hash, vec FROM vectors")
    rows = cur.fetchall()
    if not rows:
        raise RuntimeError("No vectors found - did you point to the right data?")

    mat   = np.vstack([np.frombuffer(v, dtype="float32") for _, v in rows])
    faiss.normalize_L2(mat)
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)

    print(f"📈 index ready – {len(rows)} vectors, dim={mat.shape[1]}")
    return conn, index, [h for h, _ in rows]

def search(qvec, index, hashes, topk=6):
    qvec = qvec.astype("float32")
    faiss.normalize_L2(qvec)
    D, I = index.search(qvec, topk)
    return [(hashes[i], float(D[0][k])) for k, i in enumerate(I[0]) if i != -1]

def make_query(prompt: str, conn, index, hashes) -> str:
    # ---------- exact-string shortcut ---------------------------------
    m = re.search(r'`([^`]+)`|"([^"]+)"|\'([^\']+)\'', prompt)
    needle = next((g for g in (m.groups() if m else []) if g), None)

    if not needle and re.match(r'(?i)^is\s+["\']?.+["\']?\s+present\??', prompt.strip()):
        phrase = re.findall(r'"([^"]+)"|\'([^\']+)', prompt)
        needle = phrase[0][0] if phrase else prompt.split(maxsplit=1)[1]

    if needle:
        cur = conn.execute(
            "SELECT file, start, text FROM chunks WHERE text LIKE ?",
            (f"%{needle}%",),
        )
        hits = cur.fetchall()
        if hits:
            # --- build a mini-context for the LLM -----------------------
            ctx_lines = []
            for f, idx, txt in hits[:8]:         # limit to 8 chunks
                excerpt = textwrap.shorten(txt, 300)
                ctx_lines.append(f"[{f}:{idx}] {excerpt}")
            context = "\n".join(ctx_lines)

            q = f"""You are LogGPT.  Using *only* the context below,
            answer the question.

            Context:
            {context}

            Question: {prompt}
            """

            # ---- call Ollama or OpenAI exactly like RAG mode ----------
            if OLLAMA_URL and not USE_OPENAI_GEN:
                js = requests.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model":"mistral:7b-instruct",
                        "prompt":q, "stream":False,
                        "options":{"temperature":0}},
                    timeout=120).json()
                if "response" in js:
                    return js["response"].strip()
                raise RuntimeError(f"Ollama error → {js.get('error', js)}")
            else:
                client = OpenAI(api_key=OPENAI_KEY)
                rs = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"user","content":q}],
                    temperature=0, max_tokens=256)
                return rs.choices[0].message.content.strip()

            # (grep found nothing useful → fall through to normal RA

    # -------------------- RAG mode --------------------  (unchanged)
    qvec  = embed([prompt])
    sims  = search(qvec, index, hashes)
    cur   = conn.cursor()
    ctxs  = []
    for h, d in sims:
        cur.execute("select file,text from chunks where hash=?", (h,))
        f, t = cur.fetchone()
        ctxs.append(f"[{f}] {textwrap.shorten(t, 140)} (sim {d:.2f})")
    context = "\n".join(ctxs)

    # print(context)

    q = f"""You are LogGPT. Answer using *only* the info below.

    Context:
    {context}

    Question: {prompt}
    """

    # ---- choose local Ollama vs. OpenAI ----
    if OLLAMA_URL and not USE_OPENAI_GEN:
        r  = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": "mistral:7b-instruct",
                "prompt": q,
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=120,
        )
        js = r.json()
        if "response" not in js:
            raise RuntimeError(f"Ollama generate error → {js.get('error', js)}")
        return js["response"].strip()
    else:
        client = OpenAI(api_key=OPENAI_KEY)
        rs = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": q}],
            temperature=0,
            max_tokens=256,
        )
        return rs.choices[0].message.content.strip()

def repl(sources: list[pathlib.Path]):
    """Build index from *all* sources and open an interactive prompt."""
    print("Building index …")
    conn, index, hashes = build_index(sources)

    print("Ready. Type your question (or Ctrl-C to exit).")
    while True:
        try:
            q = input(">> ").strip()
            if not q:
                continue
            answer = make_query(q, conn, index, hashes)
            print(textwrap.fill(answer, 100))
        except (EOFError, KeyboardInterrupt):
            break

def add_config_file():
    config_path = Path.cwd() / "rag_chat.json"  # Create config file in the current directory
    default_config = {
        "line_spacing": 1,
        "response_color": "default",
        "allowed_folders": [],  # List of allowed folders
        "rules": {
            "example_rule": "value",  # Add your rules here
            "another_rule": "value"
        }
    }

    if config_path.exists():
        print(f"Configuration file already exists at {config_path}.")
    else:
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=4)
        print(f"Configuration file created at {config_path}.")

def add_folder(folder_path):
    config = load_config()
    if folder_path in config["allowed_folders"]:
        print(f"Folder '{folder_path}' is already in the allowed list.")
    else:
        config["allowed_folders"].append(folder_path)
        save_config(config)
        print(f"Folder '{folder_path}' added to the allowed list.")

def set_openai_key(new_key):
    env_file = Path(__file__).parent / ".env"  # Path to the .env file
    if not env_file.exists():
        print(f"Error: The .env file does not exist at {env_file}.")
        return

    # Read the existing .env file
    with open(env_file, 'r') as f:
        lines = f.readlines()

    # Update or add the OpenAI key
    key_found = False
    for i, line in enumerate(lines):
        if line.startswith("OPENAI_API_KEY="):
            lines[i] = f"OPENAI_API_KEY={new_key}\n"  # Update the existing key
            key_found = True
            break

    if not key_found:
        lines.append(f"OPENAI_API_KEY={new_key}\n")  # Add the key if it doesn't exist

    # Write the updated lines back to the .env file
    with open(env_file, 'w') as f:
        f.writelines(lines)

    print(f"OpenAI API key set to: {new_key}")

def main() -> None:
    ap = argparse.ArgumentParser(description="Modular RAG chat over logs & DBs")
    ap.add_argument(
        "--sources",
        nargs="*",
        help="One or more folders / text files / Data.db files to index",
    )
    ap.add_argument(
        "--remote",
        action="store_true",
        help="Force OpenAI (GPT-4o-mini) even if OLLAMA_URL is set",
    )
    ap.add_argument(
        "--configs",
        action="store_true",
        help="Enter configuration mode to customize chat settings",
    )
    ap.add_argument(
        "command",
        nargs="?",  # Optional command
        help="Command to execute (e.g., 'add', 'add-folder', 'set')",
    )
    ap.add_argument(
        "key_value",  # Optional key value argument for setting the OpenAI key
        nargs="*",  # Accept one or more values
        help="Key to set (e.g., 'openai key sk-*')",
    )
    args = ap.parse_args()

    if args.command == "add":
        add_config_file()  # Call the function to add a config file
        return  # Exit after adding the config

    if args.command == "add-folder" and args.key_value:
        add_folder(args.key_value[0])  # Call the function to add a folder
        return  # Exit after adding the folder

    if args.command == "set" and len(args.key_value) == 3:
        set_openai_key(args.key_value[2])  # Call the function to set the OpenAI key
        return  # Exit after setting the key

    if args.configs:
        config_menu()  # Call the configuration menu
        return  # Exit after configuration

    # expand and normalise all paths only if sources are provided
    if args.sources:
        paths = [pathlib.Path(p).expanduser().resolve() for p in args.sources]
        repl(paths)  # Pass the list to the REPL
    else:
        print("No sources provided. Please provide at least one source path.")


if __name__ == "__main__":
    main()