import streamlit as st
from dotenv import load_dotenv
import os
import requests

load_dotenv()

st.set_page_config(page_title="RAG Ingest PDF", page_icon="📄", layout="centered")

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
if not BACKEND_URL.startswith(("http://", "https://")):
    BACKEND_URL = f"https://{BACKEND_URL}"


st.title("Upload a PDF to Ingest")
uploaded = st.file_uploader("Choose a PDF", type=["pdf"], accept_multiple_files=False)

if uploaded is not None:
    with st.spinner("Uploading and ingesting — this may take a minute..."):
        resp = requests.post(
            f"{BACKEND_URL}/upload",
            files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
            timeout=600,
        )
    if resp.ok:
        data = resp.json()
        st.success(f"Ingestion complete for: {data['source_id']}. You can now ask questions!")
    else:
        st.error(f"Upload failed: {resp.text}")
    st.caption("You can upload another PDF if you like.")

st.divider()
st.title("Ask a question about your PDFs")


with st.form("rag_query_form"):
    question = st.text_input("Your question")
    top_k = st.number_input(
        "How many chunks to retrieve",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        help="How many text pieces the app reads before answering. 5 is a good default: lower is faster, higher gives more context.",
    )
    st.caption("The app reads this many text pieces before answering. Try 5 for a good balance.")
    submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
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