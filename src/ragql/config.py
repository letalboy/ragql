from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv
import os

load_dotenv()

@dataclass(slots=True)
class Settings:
    db_path: Path = Path(__file__).parent / ".ragql.db"
    chunk_size: int = 800
    chunk_overlap: int = 80
    openai_key: str | None = os.getenv("OPENAI_API_KEY")
    ollama_url: str | None = os.getenv("OLLAMA_URL")
    use_ollama: bool = bool(os.getenv("OLLAMA_URL"))
