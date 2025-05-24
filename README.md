
1. **Initialize the venv:**

```shell
py -m venv .venv
```

2. **Activate the requirements:**

```shell
.\.venv\Scripts\activate
```

3. **Install the requirements:**

```
python -m pip install --upgrade pip wheel setuptools
```

4. **Install the requirements**

```
python -m pip install -r requirements.txt
```

5. **Run inside ollama the `nomic-embed-text` embeddings model just to verify and test if that works:**

```shell
ollama serve               # make sure it’s running
ollama pull nomic-embed-text
curl -X POST http://localhost:11434/api/embeddings \
     -d '{"model":"nomic-embed-text","prompt":"hello"}'
```

