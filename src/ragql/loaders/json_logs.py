from pathlib import Path
from typing import Iterable, Tuple
import json, gzip

def load(path: Path) -> Iterable[Tuple[str, str]]:
    if path.suffix.lower() not in {".json", ".jsonl", ".gz"}:
        return []

    text = gzip.open(path, "rt").read() if path.suffix == ".gz" else path.read_text()
    try:
        obj  = json.loads(text)
        text = json.dumps(obj, indent=2)     # pretty-print for embedding
    except Exception:
        pass                                  # not valid JSON → keep raw
    yield str(path), text