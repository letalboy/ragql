# src/ragql/config.py
from __future__ import annotations
from pathlib import Path
from dataclasses import asdict, dataclass
from dotenv import load_dotenv
import os
import json
from json import JSONDecodeError

load_dotenv()

CONFIG_FILE = "rag_config.json"

@dataclass(slots=True)
class Settings:
    db_path: Path       = Path(__file__).parent / ".ragql.db"
    chunk_size: int     = 800
    chunk_overlap: int  = 80
    openai_key: str     = os.getenv("OPENAI_API_KEY", "")
    ollama_url: str     = os.getenv("OLLAMA_URL", "")
    use_ollama: bool    = bool(os.getenv("OLLAMA_URL"))
    
    # ← insert these two lines:
    line_spacing: int    = 1
    response_color: str  = "default"

    @classmethod
    def load(cls) -> Settings:
        """Read JSON config if present; ignore if missing or broken."""
        cfg = cls()
        p = Path(CONFIG_FILE)
        if p.exists():
            text = p.read_text()
            try:
                data = json.loads(text) if text.strip() else {}
            except JSONDecodeError:
                print(f"Warning: '{CONFIG_FILE}' contains invalid JSON, using defaults.")
                return cfg
            for k, v in data.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        return cfg

    def save(self) -> None:
        # write current settings back to JSON
        with open(CONFIG_FILE, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)

def config_menu() -> None:
    cfg = Settings.load()
    while True:
        print("\nConfiguration Menu:")
        print("1. Customize Line Spacing")
        print("2. Customize Response Color")
        print("3. Save and Exit")
        print("4. Exit without Saving")
        choice = input("Choose an option: ").strip()

        if choice == "1":
            val = input("Enter line spacing (e.g., 1, 2, 3): ").strip()
            try:
                cfg.line_spacing = int(val)
                print(f"Line spacing set to {cfg.line_spacing}.")
            except ValueError:
                print("Invalid number. Please enter an integer.")
        elif choice == "2":
            color = input(
                "Enter response color (e.g., 'red','green','blue','default'): "
            ).strip()
            cfg.response_color = color
            print(f"Response color set to '{cfg.response_color}'.")
        elif choice == "3":
            cfg.save()
            print("Configuration saved.")
            break
        elif choice == "4":
            print("Exiting without saving.")
            break
        else:
            print("Invalid choice. Please try again.")

def add_config_file() -> None:
    default = Settings()
    default.save()
    print(f"Default config written to {CONFIG_FILE}.")


def add_folder(folder: str) -> None:
    cfg = Settings.load()
    folders = getattr(cfg, "allowed_folders", [])
    if folder in folders:
        print(f"Folder '{folder}' already present.")
        return
    folders.append(folder)
    # write back
    data = asdict(cfg)
    data["allowed_folders"] = folders
    with open(CONFIG_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Added '{folder}' to allowed_folders.")


def set_openai_key(new_key: str) -> None:
    # update .env
    env = Path(".env")
    lines = env.read_text().splitlines() if env.exists() else []
    updated = False
    for i, line in enumerate(lines):
        if line.startswith("OPENAI_API_KEY="):
            lines[i] = f"OPENAI_API_KEY={new_key}"
            updated = True
            break
    if not updated:
        lines.append(f"OPENAI_API_KEY={new_key}")
    env.write_text("\n".join(lines))
    print("Updated OPENAI_API_KEY in .env")
