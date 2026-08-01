# LangChain RAG Project

A Retrieval-Augmented Generation (RAG) project using LangChain, supporting OpenAI and local Ollama models.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) — dependency management
- (Optional) [Ollama](https://ollama.com) — for running models locally

## Setup

### 1. Initialize and sync the project
```bash
uv init          # only when creating a brand-new project (skip if cloning)
uv sync          # installs dependencies from pyproject.toml + uv.lock
```

### 2. Activate the environment
```bash
source .venv/bin/activate
```
(Or skip activation and prefix commands with `uv run`, e.g. `uv run main.py`.)

### 3. Add LangChain dependencies
```bash
uv add langchain
uv add langchain-community
uv add langchain-openai
uv add langchain-ollama
uv add python-dotenv
```

### 4. Configure environment variables
Create a `.env` file in the project root:

### 5. (Optional) Pull a local Ollama model
```bash
ollama pull gemma2
ollama list        # verify what's installed
```

## Running
```bash
uv run main.py
```


## Notes
- Manage dependencies with `uv add <package>` — it updates `pyproject.toml` and `uv.lock` automatically. Avoid `pip install` inside this project; it won't be recorded and breaks reproducibility.
- `.env` is gitignored. Never commit API keys.
- To run the Jupyter notebook, make sure the kernel points to `.venv`, not your base conda environment.