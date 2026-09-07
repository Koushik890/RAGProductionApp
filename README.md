# RAG Production App

A production-oriented Retrieval-Augmented Generation (RAG) application for PDF documents.

## Live Demo

Try the deployed app here: [https://rag-ui-xxiq.onrender.com](https://rag-ui-xxiq.onrender.com)

This repository provides:
- A `FastAPI` backend for PDF ingestion and question answering.
- A `Streamlit` frontend for uploading PDFs and asking questions.
- `Inngest` functions to orchestrate ingestion and query workflows.
- `Qdrant` vector storage (local embedded mode or remote cloud mode).
- `OpenRouter` for embeddings and LLM inference (NVIDIA Nemotron models by default).

## What This App Does

1. You upload a PDF from the Streamlit UI.
2. The backend extracts text, chunks it, creates embeddings, and stores vectors in Qdrant.
3. You ask a question.
4. The backend retrieves relevant chunks, sends context to the answer model, and returns a grounded answer with sources.

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
- OpenRouter API (OpenAI-compatible)
- LlamaIndex file reader + text splitter

## Prerequisites

- Python `3.13` (see `.python-version`)
- An OpenRouter API key
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
| `OPENROUTER_API_KEY` | Yes | - | API key for embeddings and answer generation. |
| `OPENROUTER_BASE_URL` | No | `https://openrouter.ai/api/v1` | OpenAI-compatible base URL. |
| `EMBED_MODEL` | No | `nvidia/nemotron-3-embed-1b:free` | Embedding model. |
| `ANSWER_MODEL` | No | `nvidia/nemotron-3-super-120b-a12b:free` | Answer generation model. |
| `EMBED_DIM` | No | probed once | Embedding vector size. Leave unset to detect it from the model. |
| `EMBED_BATCH_SIZE` | No | `64` | Chunks sent per embedding request. |
| `ANSWER_MAX_TOKENS` | No | `4096` | Token budget for the answer, including reasoning tokens. |
| `OPENROUTER_SITE_URL` | No | unset | Optional `HTTP-Referer` for OpenRouter attribution. |
| `OPENROUTER_APP_NAME` | No | unset | Optional `X-Title` for OpenRouter attribution. |

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
| `MAX_UPLOAD_BYTES` | No | `20971520` | Maximum accepted PDF size in bytes. |
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
OPENROUTER_API_KEY=your_openrouter_key
QDRANT_COLLECTION=docs
# Optional: pin the models. These are the defaults.
EMBED_MODEL=nvidia/nemotron-3-embed-1b:free
ANSWER_MODEL=nvidia/nemotron-3-super-120b-a12b:free
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

- `GET /health/deps`
	- Checks the embedding provider and Qdrant, the two dependencies every Inngest step needs.
	- Response: `{ "embeddings": {"ok": true}, "qdrant": {"ok": false, "error": "..."} }`
	- Use it when a run fails: it names the broken dependency instead of leaving
	  every failure looking the same.

- `POST /upload`
	- Accepts multipart form-data with a PDF file.
	- Queues ingestion and returns `202` immediately with an `event_id`.
	- Response: `{ "status": "ingestion_started", "event_id": "...", "source_id": "..." }`

- `POST /query`
	- JSON body:
		```json
		{ "question": "What is this document about?", "top_k": 5 }
		```
	- Queues the query workflow and returns `202` with an `event_id`.
	- Response: `{ "status": "query_started", "event_id": "..." }`

- `GET /status/{event_id}`
	- Polls the Inngest run started by `/upload` or `/query`.
	- Returns one of:
		- `{ "state": "pending" }` - no run created yet
		- `{ "state": "running", "status": "..." }`
		- `{ "state": "completed", "output": { ... } }`
		- `{ "state": "failed", "error": { ... } }`
	- Returns `429` with a `Retry-After` header if the Inngest API rate limits the poll.

Both long-running workflows are polled rather than awaited inside the request.
Holding a request open for the whole ingestion piles up long-lived connections
on the instance and trips upstream rate limiting - that is the `Too Many Requests`
the UI used to show.

Interactive API docs:
- `http://127.0.0.1:8000/docs`

## Example cURL Calls

Upload a PDF:

```bash
curl -X POST "http://127.0.0.1:8000/upload" \
	-F "file=@./sample.pdf"
```

Then poll the run until it finishes:

```bash
curl "http://127.0.0.1:8000/status/<event_id>"
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

Set sensitive values (`OPENROUTER_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, Inngest keys) in the Render dashboard.

For production, use:
- Inngest Cloud (`INNGEST_API_BASE=https://api.inngest.com/v1`)
- Remote Qdrant (recommended)

## Operational Notes

- Local mode defaults to embedded Qdrant storage in `qdrant_local_storage/`.
- The embedding vector size is probed once from `EMBED_MODEL` unless `EMBED_DIM` is set.
- If the collection's vector size does not match the embedding model, it is dropped and
  recreated, and every document must be re-uploaded. Set `QDRANT_ALLOW_RECREATE=false`
  to raise an error instead of dropping a remote collection.
- Changing `EMBED_MODEL` therefore invalidates everything already ingested.
- `/upload` returns immediately; the client polls `/status/{event_id}` until the run ends.
- Free OpenRouter models (`:free` suffix) are limited to 20 requests/minute and
  50/day per account, shared across all free models. Switch to a paid model id to
  remove the daily cap.

## Troubleshooting

- `RuntimeError: OPENROUTER_API_KEY is not set`
	- Add `OPENROUTER_API_KEY` to `.env` and restart services.

- Every ingestion and query fails with the same error
	- Call `GET /health/deps` to see whether the embedding provider or Qdrant is broken.

- Inngest polling timeout (`Timed out waiting for Inngest run`)
	- Ensure Inngest dev/cloud is running and reachable.
	- Confirm `INNGEST_API_BASE` and keys are correct.

- Vector dimension mismatch with remote Qdrant
	- The collection is rebuilt automatically. Re-upload your PDFs afterwards.
	- Set `EMBED_DIM` only if you want to skip the probe and pin the size yourself.

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

This project is licensed under the MIT License. See `LICENSE` for details.
