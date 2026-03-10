import streamlit as st
from dotenv import load_dotenv
import os
import requests

load_dotenv()

st.set_page_config(page_title="RAG Ingest PDF", page_icon="📄", layout="centered")

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
if not BACKEND_URL.startswith(("http://", "https://")):
    BACKEND_URL = f"https://{BACKEND_URL}"


if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = set()
if "ingestion_messages" not in st.session_state:
    st.session_state.ingestion_messages = {}

st.title("Upload a PDF to Ingest")
uploaded = st.file_uploader("Choose a PDF", type=["pdf"], accept_multiple_files=False)

if uploaded is not None:
    file_key = f"{uploaded.name}_{uploaded.size}"
    if file_key not in st.session_state.ingested_files:
        with st.spinner("Uploading and ingesting — this may take a minute..."):
            resp = requests.post(
                f"{BACKEND_URL}/upload",
                files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                timeout=600,
            )
        if resp.ok:
            data = resp.json()
            st.session_state.ingested_files.add(file_key)
            message = f"Ingestion complete for: {data['source_id']}. You can now ask questions!"
            st.session_state.ingestion_messages[file_key] = message
            st.success(message)
        else:
            st.error(f"Upload failed: {resp.text}")
    else:
        message = st.session_state.ingestion_messages.get(file_key)
        if message:
            st.success(message)
        else:
            st.success(f"Already ingested: {uploaded.name}. You can ask questions below!")
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
    with st.spinner("Generating answer..."):
        resp = requests.post(
            f"{BACKEND_URL}/query",
            json={"question": question.strip(), "top_k": int(top_k)},
            timeout=120,
        )
    if resp.ok:
        output = resp.json()
        answer = output.get("answer", "")
        sources = output.get("sources", [])

        st.subheader("Answer")
        st.write(answer or "(No answer)")
        if sources:
            st.caption("Sources")
            for s in sources:
                st.write(f"- {s}")
    else:
        st.error(f"Query failed: {resp.text}")