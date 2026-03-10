# RAG Production App

A production-oriented Retrieval-Augmented Generation (RAG) application for PDF documents.

## Live Demo

Try the deployed app here: [https://rag-ui-xxiq.onrender.com](https://rag-ui-xxiq.onrender.com)

This repository provides:
- A `FastAPI` backend for PDF ingestion and question answering.
- A `Streamlit` frontend for uploading PDFs and asking questions.
- `Inngest` functions to orchestrate ingestion and query workflows.
- `Qdrant` vector storage (local embedded mode or remote cloud mode).
- `Mistral` embeddings and LLM inference.

## What This App Does

1. You upload a PDF from the Streamlit UI.
2. The backend extracts text, chunks it, creates embeddings, and stores vectors in Qdrant.
3. You ask a question.
4. The backend retrieves relevant chunks, sends context to Mistral, and returns a grounded answer with sources.

## Architecture

High-level flow:

1. `streamlit_app.py` sends requests to FastAPI (`/upload`, `/query`).
2. `main.py` emits Inngest events (`rag/ingest_pdf`, `rag/query_pdf_ai`).
3. Inngest functions execute:
	 - Ingestion: load PDF -> chunk -> embed -> upsert to Qdrant.
	 - Query: embed question -> vector search -> LLM answer generation.
4. FastAPI polls Inngest run status and returns final output to the client.

Core modules:
- `main.py`: API, Inngest function registration, polling helper.
- `data_loader.py`: PDF parsing, chunking, embedding.
- `vector_db.py`: Qdrant client and collection management.
- `streamlit_app.py`: end-user UI.
- `custom_types.py`: Pydantic result models.

## Tech Stack

- Python 3.13
- FastAPI + Uvicorn
- Streamlit
- Inngest
- Qdrant
- Mistral API
- LlamaIndex file reader + text splitter

## Prerequisites

- Python `3.13` (see `.python-version`)
- A Mistral API key
- One of the following for vector storage:
	- Local embedded Qdrant (default, no extra service required)
	- Qdrant Cloud / remote Qdrant (`QDRANT_URL` + `QDRANT_API_KEY`)
- Inngest runtime:
	- Local development: Inngest dev server
	- Production: Inngest Cloud

## Environment Variables

Create a `.env` file in the project root.

Required for core functionality:

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `MISTRAL_API_KEY` | Yes | - | API key for embeddings and answer generation. |
| `MISTRAL_EMBED_DIM` | No | `1024` | Embedding vector dimension used for Qdrant collection. |

Qdrant configuration:

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `QDRANT_URL` | No | unset | If set, uses remote Qdrant. |
| `QDRANT_API_KEY` | No | unset | API key for remote Qdrant. |
| `QDRANT_COLLECTION` | No | `docs` | Collection name used for vector storage. |
| `QDRANT_PATH` | No | `qdrant_local_storage/` | Local embedded Qdrant storage path (if `QDRANT_URL` is unset). |

Inngest configuration:

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `INNGEST_API_BASE` | No | `http://127.0.0.1:8288/v1` | Inngest API base URL used by polling helper. |
| `INNGEST_SIGNING_KEY` | No | empty | Used as bearer token when polling Inngest API. |
| `INNGEST_EVENT_KEY` | Depends | unset | Event key for Inngest Cloud event ingestion. |
| `INNGEST_PRODUCTION` | No | `false` | Toggles production mode in Inngest client. |

App/UI configuration:

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `UPLOADS_DIR` | No | `uploads` | Directory where uploaded PDFs are stored. |
| `BACKEND_URL` | No | `http://127.0.0.1:8000` | FastAPI base URL used by Streamlit UI. |

## Local Development Setup

### 1) Install dependencies

Using `pip`:

```bash
python -m venv .venv
# Windows PowerShell
. .venv/Scripts/Activate.ps1
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

Using `uv` (optional):

```bash
uv sync
```

### 2) Configure environment

Create `.env` with at least:

```env
MISTRAL_API_KEY=your_mistral_key
# Optional but recommended for clarity
MISTRAL_EMBED_DIM=1024
QDRANT_COLLECTION=docs
```

For local embedded Qdrant, no extra setup is required.

### 3) Start Inngest dev server

Run Inngest in one terminal:

```bash
npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest
```

### 4) Start FastAPI backend

In another terminal:

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### 5) Start Streamlit UI

In another terminal:

```bash
streamlit run streamlit_app.py
```

Open Streamlit URL shown in terminal (typically `http://localhost:8501`).

## API Endpoints

Base URL: `http://127.0.0.1:8000`

- `GET /health`
	- Returns service health.

- `POST /upload`
	- Accepts multipart form-data with a PDF file.
	- Triggers ingestion and waits for completion.
	- Returns ingestion status and source info.

- `POST /query`
	- JSON body:
		```json
		{ "question": "What is this document about?", "top_k": 5 }
		```
	- Triggers query workflow and returns answer + sources.

Interactive API docs:
- `http://127.0.0.1:8000/docs`

## Example cURL Calls

Upload a PDF:

```bash
curl -X POST "http://127.0.0.1:8000/upload" \
	-F "file=@./sample.pdf"
```

Ask a question:

```bash
curl -X POST "http://127.0.0.1:8000/query" \
	-H "Content-Type: application/json" \
	-d '{"question":"Summarize the main points","top_k":5}'
```

## Deployment

`render.yaml` is included with two services:
- `rag-api`: FastAPI backend
- `rag-ui`: Streamlit frontend

Set sensitive values (`MISTRAL_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, Inngest keys) in the Render dashboard.

For production, use:
- Inngest Cloud (`INNGEST_API_BASE=https://api.inngest.com/v1`)
- Remote Qdrant (recommended)

## Operational Notes

- Local mode defaults to embedded Qdrant storage in `qdrant_local_storage/`.
- If embedding dimension changes, local collection is recreated automatically.
- If using remote Qdrant and dimensions do not match, startup raises an error (collection must be recreated or config corrected).
- `/upload` waits for ingestion completion, so first response can take time for larger PDFs.

## Troubleshooting

- `RuntimeError: MISTRAL_API_KEY is not set`
	- Add `MISTRAL_API_KEY` to `.env` and restart services.

- Inngest polling timeout (`Timed out waiting for Inngest run`)
	- Ensure Inngest dev/cloud is running and reachable.
	- Confirm `INNGEST_API_BASE` and keys are correct.

- Vector dimension mismatch with remote Qdrant
	- Match `MISTRAL_EMBED_DIM` with collection dimension, or recreate collection.

- Streamlit cannot reach backend
	- Verify `BACKEND_URL` and that FastAPI is running.

## Repository Layout

```text
.
|- main.py
|- streamlit_app.py
|- data_loader.py
|- vector_db.py
|- custom_types.py
|- render.yaml
|- requirements.txt
|- pyproject.toml
|- uploads/
|- qdrant_local_storage/
```

## License

Add your preferred license (for example MIT) in a `LICENSE` file.
