# RAG Agent Backend

A fully local, production-ready RAG (Retrieval-Augmented Generation) backend — no cloud APIs, no paid services.

---

## Stack

| Layer | Tool |
|---|---|
| API | FastAPI |
| Vector DB | FAISS |
| Embeddings | `BAAI/bge-small-en-v1.5` (sentence-transformers) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | OLLama — `llama3` |
| Pipeline | LangGraph |
| Memory | SQLite |

---

## Folder Structure

```
rag_project/
├── app/
│   ├── api/
│   │   └── v1/
│   │       └── endpoints/
│   │           ├── chat.py          # POST /chat
│   │           ├── ingest.py        # POST /ingest, POST /ingest/file
│   │           └── history.py       # GET /history/{thread_id}
│   ├── core/
│   │   └── config.py                # All settings (loaded from .env)
│   ├── db/
│   │   └── session.py               # SQLite setup and table init
│   ├── rag/
│   │   ├── faiss_store.py           # FAISS vector store
│   │   ├── ollama_client.py         # OLLama HTTP client
│   │   ├── reranker.py              # Cross-encoder reranking
│   │   └── graph.py                 # LangGraph pipeline
│   ├── schemas/
│   │   └── chat.py                  # Pydantic request/response models
│   ├── services/
│   │   └── memory_service.py        # Chat history CRUD (SQLite)
│   └── main.py                      # FastAPI app entry point
├── .env
├── requirements.txt
└── README.md
```

---

## Configuration

All key parameters live in `.env` (or `app/core/config.py`). Here are the current values:

```env
# LLM
OLLAMA_MODEL=llama3

# Embeddings
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# Chunking
CHUNK_SIZE=300
CHUNK_OVERLAP=120

# Retrieval
TOP_K_RESULTS=8

# Reranking
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# SQLite
SQLITE_DB_PATH=chat_memory.db

# FAISS
FAISS_INDEX_PATH=faiss_index
FAISS_EMBEDDING_DIM=384
```

### Why these values?

**Chunking — `chunk_size=300`, `chunk_overlap=120`**
Smaller chunks (300 chars) give more precise retrieval — the retrieved passages are tighter and more relevant to the question. A generous overlap of 120 chars (40% of chunk size) ensures context is not lost at chunk boundaries, which is critical for coherent answers.

**Embedding — `BAAI/bge-small-en-v1.5`**
BGE (BAAI General Embeddings) consistently outperforms `all-MiniLM-L6-v2` on retrieval benchmarks (MTEB). The `small` variant is fast and lightweight while still producing high-quality dense vectors. It's the recommended upgrade for RAG use cases.

**Retrieval — `top_k=8`**
Fetching 8 candidates before reranking gives the cross-encoder enough material to pick the best passages. Because reranking is applied afterward, the LLM only sees the top results — so fetching more at the FAISS stage is cheap and improves recall.

**Reranking — `cross-encoder/ms-marco-MiniLM-L-6-v2`**
FAISS retrieves by approximate vector similarity, which can miss nuanced relevance. The cross-encoder reranker reads the query and each candidate chunk together, scoring them much more accurately. This two-stage retrieve-then-rerank pipeline significantly improves answer quality.

**LLM — `llama3`**
Meta's Llama 3 (8B) outperforms Mistral 7B on instruction-following and open-domain QA. It's the best freely available model on OLLama for RAG tasks.

---

## Quick Setup

### Step 1 — Install OLLama

```bash
# windows
pip install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows — download from https://ollama.ai/download
```

### Step 2 — Pull the Model

```bash
ollama pull llama3
```

### Step 3 — Start OLLama Server

```bash
ollama serve
# Runs on http://localhost:11434
```

### Step 4 — Set Up Python Environment

```bash
python3.11 -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

> On first run, sentence-transformers will download `BAAI/bge-small-en-v1.5` (~130MB) and the reranker model (~80MB) automatically.

### Step 5 — Run the Server

```bash
uvicorn main:app --reload --port 8000
```

| URL | Purpose |
|---|---|
| `http://localhost:8000` | API root |
| `http://localhost:8000/docs` | Swagger UI (interactive docs) |

---

## API Endpoints

### `POST /api/v1/ingest` — Ingest raw text

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "text": "FastAPI is a modern web framework for Python. It is fast and generates automatic docs.",
    "source": "fastapi_guide"
  }'
```

```json
{
  "message": "Document ingested successfully from source: fastapi_guide",
  "chunks_added": 3
}
```

---

### `POST /api/v1/ingest/file` — Upload a file

Supports `.txt`, `.md`, `.pdf`, `.docx`, `.csv`.

```bash
curl -X POST http://localhost:8000/api/v1/ingest/file \
  -F "file=@/path/to/document.pdf" \
  -F "source=my_doc"
```

```json
{
  "message": "File ingested successfully from source: my_doc",
  "filename": "document.pdf",
  "chunks_added": 12
}
```

---

### `POST /api/v1/chat` — Ask a question

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "thread_id": "user-123",
    "message": "What is FastAPI?"
  }'
```

```json
{
  "thread_id": "user-123",
  "answer": "FastAPI is a modern Python web framework that is fast, easy to use, and automatically generates API documentation."
}
```

---

### `GET /api/v1/history/{thread_id}` — Get conversation history

```bash
curl http://localhost:8000/api/v1/history/user-123
```

```json
{
  "thread_id": "user-123",
  "messages": [
    { "role": "user",      "content": "What is FastAPI?",              "created_at": "2024-01-15 10:30:00" },
    { "role": "assistant", "content": "FastAPI is a modern framework...", "created_at": "2024-01-15 10:30:05" }
  ]
}
```

---

## How Data Flows

```
POST /chat  { thread_id, message }
        │
        ▼
┌─────────────────────┐
│  1. load_history    │  ← last 5 messages from SQLite
└────────┬────────────┘
         │
┌────────▼────────────┐
│  2. retrieve_docs   │  ← embed query → FAISS top-8 chunks
└────────┬────────────┘
         │
┌────────▼────────────┐
│  3. rerank          │  ← cross-encoder scores all 8, keeps best
└────────┬────────────┘
         │
┌────────▼────────────┐
│  4. build_prompt    │  ← system + context + history + question
└────────┬────────────┘
         │
┌────────▼────────────┐
│  5. generate        │  ← OLLama llama3 → answer text
└────────┬────────────┘
         │
┌────────▼────────────┐
│  6. save_message    │  ← user + assistant saved to SQLite
└────────┬────────────┘
         │
         ▼
  Return answer to user
```

---

## Other Free OLLama Models

| Model | Pull Command | RAM | Notes |
|---|---|---|---|
| **llama3** ✅ | `ollama pull llama3` | ~5 GB | Current default — best accuracy |
| mistral | `ollama pull mistral` | ~4 GB | Fast, good general QA |
| gemma:7b | `ollama pull gemma:7b` | ~5 GB | Strong instruction following |
| phi3 | `ollama pull phi3` | ~2 GB | Best option for low-RAM machines |
| deepseek-r1:7b | `ollama pull deepseek-r1:7b` | ~5 GB | Best for reasoning-heavy questions |

Change the model by updating `OLLAMA_MODEL` in `.env`.

--

## Troubleshooting

**OLLama not connecting**
```bash
ollama serve          # make sure it's running
ollama list           # confirm llama3 is downloaded
```

**Slow responses**
- Lower `MAX_HISTORY_MESSAGES` in `.env`
- Lower `TOP_K_RESULTS` from 8 to 4 in `.env`
- Switch to `phi3` if RAM is limited

**Empty / bad answers**
- Ingest documents first via `POST /ingest`
- Try `WHISPER_MODEL=small` if using voice features

**Reranker not improving results**
- Make sure `sentence-transformers` is up to date: `pip install -U sentence-transformers`

---

## Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

> OLLama must run separately — on the host or as a sidecar container. Pass `OLLAMA_BASE_URL=http://host.docker.internal:11434` as an env var when running the container on macOS/Windows, or use the host's IP on Linux.
