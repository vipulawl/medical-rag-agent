# Medical RAG Agent

A local, on-premise medical AI assistant that:
- Ingests your personal medical documents into a vector database
- Answers questions using RAG (Retrieval-Augmented Generation) over your docs
- Autonomously researches PubMed for new publications and enriches the knowledge base on a schedule

All data stays on your machine. No documents or queries leave your device.

---

## Stack

| Component | Tool |
|-----------|------|
| Local LLM | Ollama (`llama3.2:3b`) |
| Embeddings | Ollama (`nomic-embed-text`) |
| Vector DB | ChromaDB (persisted to `./db/`) |
| Orchestration | LlamaIndex |
| Research source | PubMed API (free, no key required) |

---

## Prerequisites

1. **Install Ollama**: https://ollama.com/download
2. **Pull required models**:
   ```bash
   ollama pull llama3.2:3b
   ollama pull nomic-embed-text
   ```
3. **Python 3.11+** (check with `python3 --version`)

---

## Setup

```bash
cd ~/Projects/medical-rag-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env to set your medical condition/search terms
```

---

## Usage

### 1. Ingest your documents

Drop any PDFs, Word docs, or text files into the `documents/` folder, then run:

```bash
python scripts/ingest.py
```

Supported formats: `.pdf`, `.docx`, `.txt`, `.md`

### 2. Chat with your assistant

```bash
python scripts/query.py
```

Flags:
- `--model llama3.2:3b` — override the LLM (default: llama3.2:3b)
- `--top-k 5` — number of document chunks to retrieve per query (default: 5)
- `--no-stream` — disable streaming output

### 3. Fetch new research (manual)

```bash
python scripts/research_agent.py
```

Searches PubMed for your configured condition, downloads abstracts + metadata, and ingests them into the vector DB automatically.

Flags:
- `--query "type 2 diabetes insulin"` — override search query
- `--limit 20` — max results to fetch (default: 10)
- `--days 30` — only fetch papers published in last N days (default: 90)

### 4. Run the research scheduler (background daemon)

```bash
python scripts/scheduler.py
```

Runs the research agent every 24 hours by default. Keeps the knowledge base up to date automatically.

Flags:
- `--interval-hours 12` — how often to run (default: 24)
- `--run-now` — also run immediately on start

---

## Configuration (`.env`)

```
MEDICAL_CONDITION=type 2 diabetes
PUBMED_SEARCH_QUERY=type 2 diabetes treatment insulin resistance
PUBMED_MAX_RESULTS=10
PUBMED_DAYS_BACK=90
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=llama3.2:3b
EMBED_MODEL=nomic-embed-text
```

---

## File Structure

```
medical-rag-agent/
├── documents/          # Drop your medical docs here
├── db/                 # ChromaDB vector store (auto-created)
├── scripts/
│   ├── ingest.py       # Ingest documents into vector DB
│   ├── query.py        # Interactive CLI chat
│   ├── research_agent.py  # PubMed fetcher + auto-ingest
│   └── scheduler.py    # Runs research agent on a schedule
├── src/
│   ├── rag.py          # Core RAG pipeline
│   └── pubmed.py       # PubMed API client
├── .env.example
├── requirements.txt
└── README.md
```

---

## Cost

**Zero API cost.** Everything runs locally via Ollama. PubMed is free and requires no API key.

---

## Disclaimer

This tool is for informational and research purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.
