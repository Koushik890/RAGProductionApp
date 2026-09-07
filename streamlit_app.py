import os
import time

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="RAG Ingest PDF", page_icon="📄", layout="centered")

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
if not BACKEND_URL.startswith(("http://", "https://")):
    BACKEND_URL = f"https://{BACKEND_URL}"

POLL_INTERVAL_S = 3.0
INGEST_TIMEOUT_S = 600.0
QUERY_TIMEOUT_S = 180.0
MAX_TRANSIENT_RETRIES = 4


class BackendError(Exception):
    """A backend call failed in a way worth showing to the user."""


def _describe_http_error(resp: requests.Response) -> str:
    """Build a readable message, whether the body is JSON or a bare proxy string."""
    try:
        payload = resp.json()
    except ValueError:
        body = (resp.text or "").strip()
        return f"HTTP {resp.status_code}: {body[:300]}" if body else f"HTTP {resp.status_code}"

    detail = payload.get("detail") if isinstance(payload, dict) else None
    return f"HTTP {resp.status_code}: {detail or payload}"


def call_backend(method: str, path: str, **kwargs) -> dict:
    """Call the backend, retrying 429 and 5xx with backoff.

    A 429 here normally comes from the platform edge rather than the app, so the
    right response is to wait out Retry-After rather than to hammer it.
    """
    url = f"{BACKEND_URL}{path}"
    delay = 2.0

    for attempt in range(MAX_TRANSIENT_RETRIES):
        try:
            resp = requests.request(method, url, timeout=60, **kwargs)
        except requests.RequestException as exc:
            if attempt == MAX_TRANSIENT_RETRIES - 1:
                raise BackendError(f"Could not reach the backend: {exc}") from exc
            time.sleep(delay)
            delay *= 2
            continue

        if resp.status_code in (429, 502, 503, 504) and attempt < MAX_TRANSIENT_RETRIES - 1:
            retry_after = resp.headers.get("Retry-After")
            try:
                wait = float(retry_after) if retry_after else delay
            except ValueError:
                wait = delay
            time.sleep(min(wait, 30.0))
            delay *= 2
            continue

        if not resp.ok:
            raise BackendError(_describe_http_error(resp))

        return resp.json()

    raise BackendError("Backend is rate limited or unavailable; please try again in a minute.")


def wait_for_run(event_id: str, timeout_s: float, progress_label) -> dict:
    """Poll /status until the Inngest run finishes, then return its output."""
    deadline = time.time() + timeout_s

    while time.time() < deadline:
        status = call_backend("GET", f"/status/{event_id}")
        state = status.get("state")

        if state == "completed":
            return status.get("output") or {}
        if state == "failed":
            raise BackendError(f"Processing failed: {status.get('error') or status.get('status')}")

        progress_label.caption(f"Status: {state or 'pending'}…")
        time.sleep(POLL_INTERVAL_S)

    raise BackendError("Timed out waiting for the backend to finish processing.")


if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = set()
if "ingestion_messages" not in st.session_state:
    st.session_state.ingestion_messages = {}
if "pending_ingestion" not in st.session_state:
    # {"file_key": ..., "event_id": ..., "source_id": ...} while a run is in flight.
    st.session_state.pending_ingestion = None

st.title("Upload a PDF to Ingest")
uploaded = st.file_uploader("Choose a PDF", type=["pdf"], accept_multiple_files=False)

if uploaded is not None:
    file_key = f"{uploaded.name}_{uploaded.size}"
    already_done = file_key in st.session_state.ingested_files
    pending = st.session_state.pending_ingestion

    if already_done:
        message = st.session_state.ingestion_messages.get(file_key)
        st.success(message or f"Already ingested: {uploaded.name}. You can ask questions below!")
    else:
        # The upload is behind an explicit button. Streamlit reruns the whole
        # script on every widget interaction, so firing the POST unconditionally
        # re-uploaded the same file on every rerun and buried the backend.
        resume = bool(pending and pending["file_key"] == file_key)
        start = resume or st.button("Ingest this PDF", type="primary")

        if start:
            label = st.empty()
            try:
                if not resume:
                    with st.spinner("Uploading…"):
                        started = call_backend(
                            "POST",
                            "/upload",
                            files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                        )
                    pending = {
                        "file_key": file_key,
                        "event_id": started["event_id"],
                        "source_id": started.get("source_id", uploaded.name),
                    }
                    st.session_state.pending_ingestion = pending

                with st.spinner("Ingesting — this may take a minute…"):
                    wait_for_run(pending["event_id"], INGEST_TIMEOUT_S, label)
            except BackendError as exc:
                st.session_state.pending_ingestion = None
                label.empty()
                st.error(f"Upload failed: {exc}")
            else:
                label.empty()
                st.session_state.pending_ingestion = None
                st.session_state.ingested_files.add(file_key)
                message = (
                    f"Ingestion complete for: {pending['source_id']}. You can now ask questions!"
                )
                st.session_state.ingestion_messages[file_key] = message
                st.success(message)

    st.caption("You can upload another PDF if you like.")

st.divider()
st.title("Ask a question about your PDFs")

can_query = len(st.session_state.ingested_files) > 0
if not can_query:
    st.info("Upload and finish ingesting at least one PDF to enable questions.")

with st.form("rag_query_form"):
    question = st.text_input("Your question", disabled=not can_query)
    top_k = st.number_input(
        "How many chunks to retrieve",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        disabled=not can_query,
        help="How many text pieces the app reads before answering. 5 is a good default: lower is faster, higher gives more context.",
    )
    st.caption("The app reads this many text pieces before answering. Try 5 for a good balance.")
    submitted = st.form_submit_button("Ask", disabled=not can_query)

if submitted and not can_query:
    st.warning("Please complete PDF upload and ingestion first.")
elif submitted and question.strip():
    label = st.empty()
    try:
        with st.spinner("Generating answer…"):
            started = call_backend(
                "POST",
                "/query",
                json={"question": question.strip(), "top_k": int(top_k)},
            )
            output = wait_for_run(started["event_id"], QUERY_TIMEOUT_S, label)
    except BackendError as exc:
        label.empty()
        st.error(f"Query failed: {exc}")
    else:
        label.empty()
        answer = output.get("answer", "")
        sources = output.get("sources", [])

        st.subheader("Answer")
        st.write(answer or "(No answer)")
        if sources:
            st.caption("Sources")
            for s in sources:
                st.write(f"- {s}")
