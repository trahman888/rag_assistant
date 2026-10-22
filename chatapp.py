"""
Chat interface for the Personal Knowledge Assistant
====================================================

RUN
    1. Start Ollama            (ollama serve, or open the Ollama app)
    2. streamlit run chat_app.py
    A browser tab opens at http://localhost:8501

WHAT YOU GET
    * Chat window: your questions on the right, answers on the left, with the
      whole conversation kept on screen.
    * Answers appear word by word while the local LLM writes them.
    * Under each answer, a "Sources" box shows the exact passages it used,
      marked "cited" or "not cited".
    * Sidebar: upload files (saved to uploads/ and stored in the vector
      database), sync the folder, delete documents, clear the chat.

FILES THIS NEEDS (same folder)
    vector_store.py   search(), index_file(), index_folder(), list_documents(), delete_document()
    answer.py         SYSTEM_PROMPT, format_sources(), find_citations(), LLM_MODEL, OLLAMA_URL
    + the extract and chunk files that vector_store.py imports

HOW A QUESTION IS ANSWERED
    question -> search() top-k chunks -> numbered sources + SYSTEM_PROMPT
             -> Ollama (streamed) -> find_citations() -> show answer + sources
    Each question is searched on its own, so ask complete questions
    ("How does AWQ handle outliers?" rather than "and how about that one?").
"""
import json
from pathlib import Path

import requests
import streamlit as st

from answer import LLM_MODEL, OLLAMA_URL, SYSTEM_PROMPT, find_citations, format_sources
from vector_store import (delete_document, get_collection, index_file,
                          index_folder, list_documents, search)

import os

UPLOAD_DIR = Path(__file__).parent / "documents"   # the folder your documents are in

SUPPORTED = ["pdf", "md", "txt"]

st.set_page_config(page_title="Knowledge Assistant", page_icon="📚", layout="centered")
UPLOAD_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------------------------- helpers

def get_groq_key():
    try:
        return st.secrets["GROQ_API_KEY"]      # online (Streamlit Cloud)
    except Exception:
        return os.getenv("GROQ_API_KEY")       # local, if you set it

GROQ_MODEL = "openai/gpt-oss-20b"

def stream_answer(question: str, chunks: list[dict]):
    user_message = f"<sources>\n{format_sources(chunks)}\n</sources>\n\nQuestion: {question}"
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}]

    key = get_groq_key()
    if key:                                    # ---- online: Groq
        from groq import Groq
        stream = Groq(api_key=key).chat.completions.create(
            model=GROQ_MODEL, messages=messages,
            temperature=0.1,
            max_tokens=1024,            # was 400
            reasoning_effort="low",     # new
            stream=True)
        for part in stream:
            yield part.choices[0].delta.content or ""
        return

    with requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": LLM_MODEL,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": user_message}],
            "stream": True,
            "think": False,                    # Qwen3: skip hidden reasoning (ignored by other models)
            "options": {"temperature": 0.1, "num_ctx": 4096, "num_predict": 400},
        },
        stream=True,
        timeout=(10, 120),
    ) as response:
        if response.status_code == 404:
            raise RuntimeError(f"Model '{LLM_MODEL}' not found. Run:  ollama pull {LLM_MODEL}")
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                piece = json.loads(line)
                yield piece.get("message", {}).get("content", "")
                if piece.get("done"):
                    break


def show_sources(sources: list[dict], cited: list[int]) -> None:
    if not sources:
        return
    with st.expander(f"📄 Sources ({len(cited)} cited of {len(sources)})"):
        for n, c in enumerate(sources, start=1):
            page = f", page {c['page']}" if c.get("page") else ""
            mark = "✅ cited" if n in cited else "not cited"
            st.markdown(f"**[{n}] {c['source']}{page}** — {mark}")
            st.caption(c["text"][:600] + ("…" if len(c["text"]) > 600 else ""))


def save_upload(uploaded_file) -> Path:
    """Save into uploads/ without overwriting a different file with the same name."""
    data = uploaded_file.getvalue()
    name = Path(uploaded_file.name).name
    target, n = UPLOAD_DIR / name, 2
    while target.exists() and target.read_bytes() != data:
        target = UPLOAD_DIR / f"{Path(name).stem} ({n}){Path(name).suffix}"
        n += 1
    target.write_bytes(data)
    return target


def ollama_running() -> bool:
    try:
        requests.get(OLLAMA_URL, timeout=2)
        return True
    except requests.RequestException:
        return False


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("📁 Your documents")

    with st.form("upload", clear_on_submit=True):
        files = st.file_uploader("Add PDFs, notes or text files", type=SUPPORTED,
                                 accept_multiple_files=True)
        if st.form_submit_button("Add to knowledge base", type="primary") and files:
            for f in files:
                with st.spinner(f"Reading {f.name}…"):
                    added = index_file(save_upload(f))
                if added:
                    st.success(f"{f.name}: {added} chunks added")
                else:
                    st.info(f"{f.name}: already added (or no text found)")

    if st.button("🔄 Sync uploads folder", use_container_width=True):
        with st.spinner("Checking folder…"):
            index_folder()
        st.rerun()

    docs = list_documents()
    st.caption(f"{len(docs)} documents · {get_collection().count()} chunks")
    for d in docs:
        left, right = st.columns([5, 1])
        left.markdown(f"**{d['source']}**  \n{d['chunks']} chunks")
        if right.button("🗑️", key=f"del-{d['doc_id']}", help=f"Delete {d['source']}"):
            delete_document(d["doc_id"])
            st.rerun()

    st.divider()
    k = st.slider("Sources per answer", 1, 8, 4)
    st.caption(f"LLM: `{GROQ_MODEL}` (Groq)" if get_groq_key() else f"LLM: `{LLM_MODEL}` (Ollama)")
    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------------------------------- chat
st.title("📚 Knowledge Assistant")
st.caption("Ask questions about your documents. Answers cite their sources.")

if "messages" not in st.session_state:
    st.session_state.messages = []      # [{"role", "content", "sources", "cited"}]

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown("Hi! Upload documents in the sidebar, then ask me anything about them."
                    if not docs else
                    f"Hi! I've read **{len(docs)}** document(s). What would you like to know?")

# Re-draw the conversation so far (Streamlit re-runs this script on every message)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            show_sources(msg.get("sources", []), msg.get("cited", []))

# New question
question = st.chat_input("Ask about your documents…", disabled=not docs)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching your documents…"):
                chunks = search(question, k)
            if not chunks:
                answer_text = "I couldn't find anything in your documents yet. Upload some files first."
                st.markdown(answer_text)
            else:
                answer_text = st.write_stream(stream_answer(question, chunks)).strip()
            cited, invalid = find_citations(answer_text, len(chunks))
            if invalid:
                st.warning(f"The answer mentions sources that don't exist: {invalid}")
            show_sources(chunks, cited)
        except requests.ConnectionError:
            answer_text, chunks, cited = "⚠️ Can't reach Ollama. Start it with `ollama serve`.", [], []
            st.error(answer_text)
        except Exception as e:
            answer_text, chunks, cited = f"⚠️ Something went wrong: {e}", [], []
            st.error(answer_text)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer_text, "sources": chunks, "cited": cited})