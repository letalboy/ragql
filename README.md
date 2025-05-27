# RagQL

## Project Overview

RagQL is a local-first **Retrieval-Augmented Generation (RAG)** system designed for natural language Q\&A over your logs and databases. It provides a modular chat-like interface that can index and query local data sources such as log files and **SQLite `.db`** database files. With RagQL, you can ask questions about the contents of your logs or databases and get answers powered by a large language model, all while keeping your data **private** on your own machine. The system works by generating vector embeddings of your data and using those for context retrieval, then feeding relevant context to an LLM to produce answers. Its modular design makes it easy to extend (e.g. adding new data loaders or swapping components), and it supports dual LLM backends for maximum flexibility.

Importantly, RagQL supports both **local** and **remote** LLM/embedding backends. By default it favors a local setup using [Ollama](https://ollama.com/) (which runs open-source models on your machine) for embedding generation and question answering. This means you can run RagQL completely offline. Alternatively, you can integrate OpenAI’s API for embeddings and/or LLM responses – useful if you prefer OpenAI’s models or if you don’t have a suitable local model. This dual backend support is seamless, allowing you to switch between local and cloud as needed (with a simple flag). RagQL’s chat interface and CLI tools make it easy to interactively query your data or automate queries via scripts.

## Key Features

* **Local-First Operation** – Designed to run fully offline using local models via Ollama. All data stays on your machine, and embeddings can be generated with a local model, ensuring privacy (no log or DB data is sent to the cloud).
* **Dual Backend Support (Ollama & OpenAI)** – Use a local LLM or the OpenAI API as the backend. By default, RagQL will use an Ollama-served model if available, and you can **force** the use of OpenAI with a `--remote` flag. This gives you the choice between offline processing or OpenAI’s latest models on demand.
* **Flexible Data Source Indexing** – Index various data sources, including plain text log files and **SQLite** database files (`.db`). RagQL uses modular loader components to parse content (e.g. reading database tables via pandas or splitting log files into chunks) and then builds a vector index (using FAISS) for efficient similarity search. This modular design makes it easy to add support for new file types or data sources in the future.
* **Interactive Chat Interface** – Launch an interactive REPL to have a multi-turn conversation with your data. The chat interface allows follow-up questions and retains context from previous Q\&A turns (conversation memory), enabling a more natural dialogue when exploring data. You can also enter special commands in this mode to manage configuration or data sources on the fly.
* **One-off Query Mode** – If you prefer not to use the chat interface, you can run RagQL as a one-shot CLI tool. By specifying a question along with target source files/folders in a single command, RagQL will process the query and print the answer, then exit. This is convenient for scripting or quick queries.
* **Configuration System** – Easily manage API keys and default settings through configuration files. RagQL supports a **`.env`** file for sensitive settings (like your OpenAI API key or Ollama server URL) and a **`config.json`** for persistent configuration (such as a list of data sources to index by default, or other preferences). A built-in config mode (`--configs`) allows you to add or remove indexed sources and set keys without manually editing files. These settings persist between runs, so you can “set and forget” your environment and data sources.
* **Lightweight & Extensible** – Built with Python and standard libraries/frameworks (FAISS for embeddings index, `pandas` for data handling, `argparse` for CLI, etc.), the project remains lightweight and hackable. Developers can easily extend RagQL – for example, by adding new loader modules for different file formats, or integrating alternative vector stores – thanks to its clean, modular architecture.

## Installation

**Prerequisites:** You’ll need **Python 3.10+** and [Poetry](https://python-poetry.org/) (for dependency management) installed on your system. If you plan to use the local LLM mode, you should also install [Ollama](https://ollama.com/) and have it running (Ollama is available for macOS, Linux, and Windows; it provides a local API endpoint for running models). For remote mode, you’ll need an OpenAI account and API key.

1. Follow these steps to install RagQL:

  - **Via Poetry (recommended):**
  
    1. Clone the repo and enter it:
       ```bash
       git clone https://github.com/yourusername/ragql.git
       cd ragql
       ```
    
    2. Install dependencies:
    
       ```bash
       poetry install
       ```
  
    3. Configure your `.env` and `config.json` as described below.

  - **Install via PyPI:**
  
    - RAGQL is also published on PyPI, so you can install it directly:
    
        ```bash
        pip install ragql
        ```
    
    - Or, if you’re using Poetry in another project:
    
        ```bash
        poetry add ragql
      ```

2. After installing via PyPI, make sure to create a `.env` file in your working directory:

    ```bash
    # .env
    OPENAI_API_KEY=<your OpenAI key>
    OLLAMA_URL=http://localhost:11434
    ```
    
    Then you can run:
    
    ```bash
    ragql --help
    ```

3. **Configure Environment** – Create a `.env` file in the project root (or wherever you run RagQL) to store configuration like API keys. At minimum you should add:

   * `OPENAI_API_KEY=<your OpenAI key>` (if you plan to use OpenAI for embeddings or answers)
   * `OLLAMA_URL=http://localhost:11434` (or the appropriate URL if your Ollama server is running on a different port/host; default Ollama listens at 11434).

   RagQL will automatically load this `.env` file on startup. You can also configure these via environment variables directly, but using a `.env` is convenient for local development.
4. **(Optional) Configure Default Sources** – By default, RagQL will create a `config.json` to persist settings. You can manually create or edit this file to specify directories or files that should be indexed on startup (and other config options). However, you can also use RagQL’s interactive config commands to set this up after installation (see **Usage** below), so manual editing isn’t required.
5. **Run RagQL** – You’re all set! You can now run the tool via Poetry:

   ```bash
   poetry run ragql --help
   ```

   Or, if the Poetry environment is active, simply:

   ```bash
   ragql --help
   ```

   This will show the help message and verify that the installation was successful. (If RagQL was installed as a package or script, the `ragql` command should be available in your PATH.)

## Usage

RagQL can be used in two primary ways: an **interactive chat mode** and a **one-off query mode**. It also provides a special configuration interface to manage data sources and API keys. Below is an overview of how to use each mode and the available commands:

* **Interactive Chat Mode (REPL):** Simply run `ragql` with no arguments to enter the interactive chat interface. RagQL will load any previously configured data sources (from `config.json`) and you’ll be dropped into a REPL where you can start asking questions. Type your question and press Enter, and the system will retrieve relevant context from your indexed data and generate an answer. You can have a multi-turn conversation this way; ask follow-up questions or new questions as needed. To exit the chat, type a command like `exit` or hit `Ctrl+C`.
  *Example:*

  ```bash
  $ ragql
  RagQL --chat <type your question or 'help' for commands>.
  > **Q:** What were the error messages in yesterday's logs?
  > **A:** ... (answer is generated here) ...
  ```

  In interactive mode, you can also input special commands (instead of a question) to manage configuration without leaving the chat. For example, you can add new sources or set your API key (see **Configuration Mode** below for the command list).

* **One-off Query Mode:** You can ask a one-time question directly from the command line by specifying a source and a query together. Use the `--sources` option to provide one or more file/folder paths to index, followed by the question in quotes. RagQL will index the given sources on the fly, answer the question, and then exit. This is useful for quick queries or integrating RagQL into scripts and automation.
  *Example:*

  ```bash
  $ ragql --sources /var/log/system.log "What errors occurred on Jan 5th?"
  ```

  In the above example, RagQL will index the file `system.log` (splitting it into chunks, creating embeddings, etc.), then use the LLM to answer the question *“What errors occurred on Jan 5th?”*. The answer will be printed to the console. You can specify multiple files or directories after `--sources` if you want to query across them simultaneously.

* **Force Remote LLM (`--remote` flag):** By default, if an Ollama server URL is configured (via `OLLAMA_URL` in .env), RagQL will use the local Ollama backend for both embedding and generation. If you want to override this and use OpenAI’s API even though a local server is available, use the `--remote` option. This flag forces RagQL to use OpenAI for that run. (If no local Ollama is configured or running, RagQL will already default to using OpenAI, so `--remote` is not needed in that case.)
  *Example:*

  ```bash
  $ ragql --remote --sources notes.txt "Summarize the notes database."
  ```

  This command will ensure the OpenAI API is used to generate embeddings and answer the query, instead of any local model.

* **Configuration Mode:** You can launch a special interactive prompt specifically for managing configuration by using the `--configs` flag. For example, run `ragql --configs` to enter configuration mode. In this mode, you can type various commands to add or remove data sources and set API keys or other settings. This is an alternative to manually editing the `config.json` or `.env`. After making changes, you can exit config mode and run RagQL normally to use the updated settings.
  Within **configuration mode**, the following commands are available:

  * `add <path>` – Add a file path to the index configuration. The given file will be included in the data sources that RagQL indexes on startup. Use this to permanently add a single log or data file to your config.
  * `add-folder <directory>` – Recursively add all files from a folder (and subfolders) to the index configuration. This is useful for adding an entire directory of logs or a collection of files in one go.
  * `remove <path>` – Remove a previously added file from the configuration. (Use the exact path as it appears in your config to remove it.) **Note:** Removing a folder that was added will likely require specifying the same folder path that was originally added.
  * `list` – Show all currently configured source files/folders. This helps to review what data sources are set to be indexed by default.
  * `set openai key <API_KEY>` – Store your OpenAI API key in the configuration (and `.env`). This command will update your `.env` file (or config) with the provided OpenAI API key so that you don’t need to set it each time. The next time you run RagQL, it will load this key automatically.
  * `help` – Display a help message listing available commands in config mode.
  * `exit` – Quit the configuration mode (return to your shell).

  *Example config session:*

  ```bash
  $ ragql --configs
  RagQL Config Mode – type a command. (Type 'help' for options, 'exit' to quit.)
  > add /home/user/logs/error.log
  [Config] Added file: /home/user/logs/error.log
  > add-folder /home/user/logs/2025/
  [Config] Added folder: /home/user/logs/2025 (recursively indexed)
  > set openai key sk-XXXXXXXXXXXXXXXXXXX
  [Config] OpenAI API key saved.
  > list
  [Config] Current data sources:
    - /home/user/logs/error.log
    - /home/user/logs/2025 (folder)
  > exit
  ```

  In the above session, we added a log file and a folder of logs to the index, set the OpenAI API key, listed the config, and then exited. Now running `ragql` (interactive mode) will automatically load `/home/user/logs/error.log` and all logs in the 2025 folder into the index for querying.

**Note:** If you run `ragql` without specifying `--sources` and have not configured any default sources, the system will start with an empty index. You can still use `add` commands in the interactive chat to add sources on the fly. Also, the first time you add sources or set a key (either via config mode or interactive commands), a `config.json` file will be created to persist these settings. You can edit or clear this file if needed to reset the configuration.

## Example Commands

Here are some example use cases and the corresponding commands to illustrate how to use RagQL:

* **Index a single log file and ask a question (one-off):**

  ```bash
  ragql --sources ~/logs/system.log "What errors were logged on March 10th?"
  ```

  *Description:* This will index the file `system.log` located in `~/logs/` and answer the question about March 10th errors. After printing the answer, RagQL exits.

* **Index an entire folder of logs and open chat:**

  ```bash
  ragql --sources ~/logs/ 
  ```

  *Description:* This indexes all files under `~/logs/` (including subdirectories) and then drops you into the interactive chat REPL. You can then ask any questions about the combined log data. (Since no question was provided in the command above, it defaults to interactive mode after indexing.)

* **Use OpenAI for query despite having Ollama available:**

  ```bash
  ragql --remote --sources data.db "Give me a summary of the data in this database."
  ```

  *Description:* Even if you have a local model running, this command forces RagQL to use the OpenAI API to handle the embedding and generation. The source `data.db` (a SQLite database file) will be indexed – likely by extracting table content via pandas – and the question will be answered by the OpenAI model.

* **Configure sources and then run chat without specifying them each time:**

  ```bash
  # First, set up config with a folder and file:
  ragql --configs <<EOF
  add-folder ~/project/logs
  add ~/project/data/metrics.db
  exit
  EOF

  # Now simply run interactive chat:
  ragql
  ```

  *Description:* In the above, we used a heredoc (`<<EOF`...`EOF`) to script the config mode. We added an entire log folder and a SQLite database file to the config. After that, running `ragql` (with no arguments) will automatically load those configured sources and enter chat mode. This demonstrates how you can configure once and then reuse that configuration easily.

Feel free to mix and match these options. For instance, you can start interactive mode and then on the first prompt use an `add` command to load a new file, or run an inline query for quick answers. RagQL is flexible and adapts to your workflow.

## Tech Stack

RagQL is built on a stack of modern tools and libraries:

* **Python 3** – The core language used for development. RagQL is written in Python, making it cross-platform and easy to extend or customize.
* **[Poetry](https://python-poetry.org/)** – Used for dependency management and packaging. This ensures a reproducible environment and easy installation of required packages.
* **[Ollama](https://ollama.com/)** – Provides the local LLM backend. Ollama is an open-source tool that allows running large language models on your own hardware via a simple API. RagQL uses Ollama to generate embeddings and responses locally when available (keeping data local).
* **OpenAI API** – The cloud alternative for embeddings and LLM responses. RagQL can call OpenAI’s API (e.g. GPT-4, GPT-3.5, or Ada for embeddings) when a local model is not available or when explicitly requested. This gives access to powerful language models hosted by OpenAI.
* **FAISS** – [Facebook AI Similarity Search](https://github.com/facebookresearch/faiss) is used as the vector store for embeddings. RagQL leverages FAISS to index and search embeddings efficiently, enabling quick retrieval of relevant text chunks from your data.
* **pandas** – [pandas](https://pandas.pydata.org/) is used for data loading and manipulation, particularly for database files. For example, RagQL might use pandas to read tables from a SQLite `.db` and convert them into text or CSV for indexing.
* **python-dotenv** – [python-dotenv](https://github.com/theskumar/python-dotenv) is used to load environment variables from the `.env` file. This simplifies configuration of API keys and URLs without hardcoding them.
* **argparse** – RagQL’s command-line interface is built using Python’s built-in [argparse](https://docs.python.org/3/library/argparse.html) library. This powers the parsing of flags like `--sources`, `--remote`, and subcommands in config mode.

Additionally, RagQL is structured in a modular way (with separate components for CLI, configuration management, data loading, embedding generation, and storage). This design makes it easy for developers to understand and modify the codebase or integrate other tools (for example, swapping FAISS with a different vector database, or adding a new loader for a different file type).

## References

* [OpenAI API Documentation](https://platform.openai.com/docs/) – Official documentation for OpenAI’s API, which RagQL can use for embeddings and chat completion.
* [Ollama Project Website](https://ollama.com/) – Information and download for Ollama, the local LLM engine used for offline mode.
* [FAISS Library (Facebook AI Research)](https://github.com/facebookresearch/faiss) – GitHub repo for FAISS, the vector similarity search library used for embedding indexing.
* [Poetry: Python Package Manager](https://python-poetry.org/) – Documentation for Poetry, used for managing RagQL’s dependencies and environment.
* [Pandas Library](https://pandas.pydata.org/) – Official site for pandas, used in RagQL for data handling (especially with `.db` files).
* [python-dotenv](https://github.com/theskumar/python-dotenv) – GitHub repository for python-dotenv, which RagQL uses to manage environment variables from a file.
* [Python Argparse](https://docs.python.org/3/library/argparse.html) – Documentation for the argparse library used to build the CLI interface.
