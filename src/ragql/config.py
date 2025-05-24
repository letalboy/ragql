# src/ragql/config.py
from __future__ import annotations
from pathlib import Path
from dataclasses import asdict, dataclass
from dotenv import load_dotenv
import os
import json

load_dotenv()

CONFIG_FILE = "config.json"

@dataclass(slots=True)
class Settings:
    db_path: Path       = Path(__file__).parent / ".ragql.db"
    chunk_size: int     = 800
    chunk_overlap: int  = 80
    openai_key: str     = os.getenv("OPENAI_API_KEY", "")
    ollama_url: str     = os.getenv("OLLAMA_URL", "")
    use_ollama: bool    = bool(os.getenv("OLLAMA_URL"))

    @classmethod
    def load(cls) -> Settings:
        # read the JSON config if it exists
        cfg = cls()
        if Path(CONFIG_FILE).exists():
            data = json.loads(Path(CONFIG_FILE).read_text())
            for k, v in data.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        return cfg

    def save(self) -> None:
        # write current settings back to JSON
        with open(CONFIG_FILE, "w") as f:
            json.dump(asdict(self), f, indent=2)

def config_menu() -> None:
    cfg = Settings.load()
    while True:
        print("\nConfiguration Menu:")
        print("1. Customize Line Spacing")      # example, adjust as needed
        print("2. Customize Response Color")
        print("3. Save and Exit")
        print("4. Exit without Saving")
        choice = input("Choose an option: ")
        if choice == "3":
            cfg.save()
            print("Configuration saved.")
            break
        elif choice == "4":
            print("Exiting without saving.")
            break
        else:
            print("Not implemented yet.")


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
