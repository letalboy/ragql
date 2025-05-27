# src/ragql/cli.py
import argparse
import pathlib
from pathlib import Path
from .config import (
    Settings,
    config_menu,
    add_config_file,
    add_folder,
    set_openai_key,
)
from .core import RagQL


def main() -> None:
    # Load settings (reads .env + config.json)
    cfg = Settings.load()

    ap = argparse.ArgumentParser(
        prog="ragql",
        description="Modular RAG chat over logs & DBs",
    )
    ap.add_argument(
        "--query",
        "-q",
        metavar="QUESTION",
        help="Run one RAG-powered query and exit",
        type=str,
    )
    ap.add_argument(
        "--sources",
        nargs="*",
        help="One or more folders/text files/Data.db files to index",
    )
    ap.add_argument(
        "--remote",
        action="store_true",
        help="Force OpenAI even if OLLAMA_URL is set",
    )
    ap.add_argument(
        "--configs",
        action="store_true",
        help="Enter configuration mode",
    )
    ap.add_argument(
        "command",
        nargs="?",
        help="Command to execute (e.g., 'add', 'add-folder', 'set')",
    )
    ap.add_argument(
        "key_value",
        nargs="*",
        help="Key/value for setting commands (e.g., 'openai key sk-…')",
    )

    args = ap.parse_args()

    if args.remote:
        cfg.use_ollama = False

    # Subcommands
    if args.command == "add":
        add_config_file()
        return

    # Add folder as source to the config
    if args.command == "add-folder" and args.key_value:
        add_folder(args.key_value[0])
        return

    # Set openai key
    if args.command == "set" and len(args.key_value) >= 3:
        # Expect: set openai key <API_KEY
        if args.key_value[0] == "openai" and args.key_value[1] == "key":
            set_openai_key(args.key_value[2])
        else:
            ap.error("Usage: ragql set openai key <YOUR_KEY>")
        return

    # Enter in the configs menu
    if args.configs:
        config_menu()
        return

    sources = args.sources or cfg.allowed_folders

    # Default: need at least one source
    if not sources:
        ap.print_usage()
        print("Please provide at least one --sources path.")
        return

    # Build index from the first source (expand as needed)
    source = pathlib.Path(sources[0]).expanduser().resolve()
    rq = RagQL(source, cfg)
    rq.build()

    # one-off query via --query
    if args.query:
        # if you want multi-word, either require quotes:
        #   ragql --query "what is status?"
        # or accept nargs='+' and join them:
        #   ap.add_argument("--query","-q", nargs="+", ...)
        answer = rq.query(args.query)
        print(answer)
        return

    # If they passed an inline question, answer and exit
    # (you could detect more than one and loop, but this matches your old style)
    if args.command is None and len(args.sources) > 1:
        # no command, two positional args: sources + question
        question = args.sources[1]
        print(rq.query(question))
        return

    # Otherwise, drop into REPL: legacy “positional question” case
    print("Entering interactive chat (Ctrl-C to exit)")
    try:
        while True:
            q = input(">> ").strip()
            if not q:
                continue
            print(rq.query(q))
    except (KeyboardInterrupt, EOFError):
        print()  # newline
        return


if __name__ == "__main__":
    main()
