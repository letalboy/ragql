# src/ragql/loaders/__init__.py  ✅ your current code
from importlib import import_module
from pathlib import Path
from typing import Iterable, Tuple, Protocol

Doc = Tuple[str, str]            # (doc_id, full_text)

class Loader(Protocol):
    def load(self, path: Path) -> Iterable[Doc]: ...

REGISTRY: list[Loader] = []

def _discover() -> None:
    here = Path(__file__).parent
    for p in here.iterdir():
        if p.suffix == ".py" and p.stem not in {"__init__", "__pycache__"}:
            mod = import_module(f"{__package__}.{p.stem}")
            if hasattr(mod, "load"):
                REGISTRY.append(mod.load)

_discover()

__all__ = ["REGISTRY", "Doc", "Loader"]
