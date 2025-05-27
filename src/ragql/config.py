# src/ragql/config.py
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field, fields, asdict
import os
import json
import logging
from json import JSONDecodeError

# load_dotenv()

CONFIG_FILE = "rag_config.json"

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Settings:
    db_path: Path = Path(__file__).parent / ".ragql.db"
    chunk_size: int = 800
    chunk_overlap: int = 80

    # keys / endpoints
    openai_key: str = os.getenv("OPENAI_API_KEY", "")
    ollama_url: str = os.getenv("OLLAMA_URL", "")
    use_ollama: bool = False

    # user prefs
    verbose = False
    allowed_folders: list[str] = field(default_factory=list)
    line_spacing: int = 1
    response_color: str = "default"

    # embed model selection: "<provider>::<model-name>"
    embed_model: str = field(
        default_factory=lambda: os.getenv(
            "RAGQL_EMBED_MODEL", "openai::text-embedding-ada-002"
        )
    )

    @property
    def embed_provider(self) -> str:
        """Return the embedding provider (before the '::')."""
        return self.embed_model.split("::", 1)[0]

    @property
    def embed_model_name(self) -> str:
        """Return the embedding model name (after the '::')."""
        parts = self.embed_model.split("::", 1)
        return parts[1] if len(parts) > 1 else parts[0]

    @classmethod
    def load(cls) -> Settings:
        """Read JSON config if present; otherwise fall back to env once."""
        cfg = cls()
        p = Path(CONFIG_FILE)
        if p.exists():
            try:
                text = p.read_text()
                data = json.loads(text) if text.strip() else {}
            except JSONDecodeError:
                print(
                    f"Warning: '{CONFIG_FILE}' contains invalid JSON, using defaults."
                )
                return cfg
            for k, v in data.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
                    print(f"loaded config {k!r} → {v!r}")
        else:
            # first-time run: grab any exisitng env vars

            cfg.openai_key = os.getenv("OPENAI_API_KEY", "")
            cfg.ollama_url = os.getenv("OLLAMA_URL", "")
            cfg.use_ollama = bool(cfg.ollama_url)

        return cfg

    def save(self) -> None:
        """Write current settings back to JSON (including embed_model)."""
        with open(CONFIG_FILE, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)


def config_menu() -> None:
    cfg = Settings.load()
    while True:
        print("\nConfiguration Menu:")
        print("1. Customize Line Spacing")
        print("2. Customize Response Color")
        print("3. Set OpenAI API Key")
        print("4. Save and Exit")
        print("5. Exit without Saving")
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
            key = input("Enter your OpenAI API key: ").strip()
            cfg.openai_key = key
            print("OpenAI API key update in config.")
        elif choice == "4":
            option = input(
                "Enter True to enable verbose by default and False to disable it"
            )
            cfg.verbose = bool(option)
            if option:
                print("Verbose mode set to enabled")
            else:
                print("Verbose mode set to disable")
        elif choice == "5":
            cfg.save()
            print("Configuration saved.")
            break
        elif choice == "6":
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

    if folder in cfg.allowed_folders:
        print(f"Folder '{folder}' already present.")
        return

    cfg.allowed_folders.append(folder)

    # now dumps everything, including allowed_folders
    with open(CONFIG_FILE, "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    print(f"✅ Added '{folder}' to allowed_folders.")


def set_openai_key(new_key: str) -> None:
    """Update the OpenAI key in rag_config.json (no more .env!)."""

    cfg = Settings.load()
    cfg.openai_key = new_key
    cfg.save()

    print("Updated OPENAI_API_KEY in config file.")


def migrate_config() -> None:
    """
    Read the old config file, transform only the
    pieces that need updating, preserve everything else,
    and write back the merged result.
    """
    p = Path(CONFIG_FILE)
    if not p.exists():
        print(f"No config file at {CONFIG_FILE}—nothing to migrate.")
        return

    # 2) Build a Settings instance from it (ignores unknown keys)
    old_cfg = Settings.load()  # your existing loader

    # 3) Create an up-to-date “base” dataclass
    new_cfg = Settings()  # this has all the new defaults

    # 4) Copy over any fields that existed in old_cfg
    #    (this preserves user-set values, including fields you didn’t touch in the new schema)
    for f in fields(Settings):
        name = f.name
        if hasattr(old_cfg, name):
            setattr(new_cfg, name, getattr(old_cfg, name))

    # 5) Now apply your schema changes.
    #    For example, say you renamed `foo` → `bar`, or split `baz` into two:
    # new_cfg.bar = new_cfg.foo; delattr(new_cfg, "foo")
    # new_cfg.new_field = compute_something(old_cfg.some_other_field)
    #
    # (Only mutate the pieces that actually changed in the new version!)

    # 6) Dump the merged result back to JSON
    merged = asdict(new_cfg)
    with open(CONFIG_FILE, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Config migrated successfully—saved to {CONFIG_FILE}")
