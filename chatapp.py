"""
Multi-user RAG chat app — login first, then each user gets their own documents.

RUN
    streamlit run chatapp.py

FLOW
    1. Not logged in  -> login / sign-up page (auth.py). Nothing else is shown.
    2. Logged in      -> sync_user() restores that user's files and index
                         (user_store.py), then the chat appears.
    3. Upload         -> saved permanently (Supabase Storage, or the local
                         folder in local mode) + indexed into the user's own
                         vector collection.
    4. Ask            -> search ONLY that user's collection -> LLM -> cited answer.
    5. Log out        -> session cleared; the next person starts at step 1.

LLM
    GROQ_API_KEY set (Streamlit secrets or env)  -> Groq (open-source gpt-oss)
    otherwise                                    -> local Ollama (answer.py settings)

FILES THIS NEEDS
    auth.py, user_store.py, answer.py, vector_store.py (+ the files vector_store imports)
"""
import json
import os

import requests
import streamlit as st

from answer import LLM_MODEL, OLLAMA_URL, SYSTEM_PROMPT, find_citations, format_sources
from auth import current_user, log_out, login_page, supabase_client, supabase_enabled
from user_store import (LocalFiles, SupabaseFiles, add_document, delete_document, list_documents,
                        search, sync_user, user_collection, user_folder)

APP_NAME = "Groundwork"          # change to your app's name
TAGLINE = "Ask questions across your documents and get answers grounded in sources."
GROQ_MODEL = "openai/gpt-oss-20b"

st.set_page_config(page_title=APP_NAME, page_icon="🔎", layout="centered")


# =========================================================================== 1. login gate
user = current_user()
if not user:
    login_page(APP_NAME, TAGLINE)
    st.stop()                    # nothing below runs until someone is logged in


# =========================================================================== 2. this user's data
folder = user_folder(user)
collection = user_collection(user)
files = SupabaseFiles(supabase_client(), user) if supabase_enabled() else LocalFiles(folder)

if not st.session_state.get("synced"):
    with st.spinner("Loading your documents…"):
        try:
            report = sync_user(files, collection, folder)
            if report["indexed"]:
                st.toast(f"Restored {report['indexed']} document(s)")
        except Exception as e:
            st.error(f"Couldn't load your documents: {e}")
    st.session_state["synced"] = True


# =========================================================================== LLM
def _secret(name):
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name)


def stream_answer(question: str, chunks: list[dict]):
    user_message = f"<sources>\n{format_sources(chunks)}\n</sources>\n\nQuestion: {question}"
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_message}]

    key = _secret("GROQ_API_KEY")
    if key:                                                   # online: Groq
        from groq import Groq
        stream = Groq(api_key=key).chat.completions.create(
            model=GROQ_MODEL, messages=messages, temperature=0.1,
            max_tokens=1024, reasoning_effort="low", stream=True)
        for part in stream:
            yield part.choices[0].delta.content or ""
        return

    with requests.post(                                       # local: Ollama
        f"{OLLAMA_URL}/api/chat",
        json={"model": LLM_MODEL, "messages": messages, "stream": True, "think": False,
              "options": {"temperature": 0.1, "num_ctx": 4096, "num_predict": 400}},
        stream=True, timeout=(10, 120),
    ) as r:
        if r.status_code == 404:
            raise RuntimeError(f"Model '{LLM_MODEL}' not found. Run:  ollama pull {LLM_MODEL}")
        r.raise_for_status()
        for line in r.iter_lines():
            if line:
                piece = json.loads(line)
                yield piece.get("message", {}).get("content", "")
                if piece.get("done"):
                    break


def llm_label() -> str:
    return f"`{GROQ_MODEL}` (Groq)" if _secret("GROQ_API_KEY") else f"`{LLM_MODEL}` (Ollama)"


def show_sources(sources: list[dict], cited: list[int]) -> None:
    if not sources:
        return
    with st.expander(f"📄 Sources ({len(cited)} cited of {len(sources)})"):
        for n, c in enumerate(sources, start=1):
            page = f", page {c['page']}" if c.get("page") else ""
            st.markdown(f"**[{n}] {c['source']}{page}** — {'✅ cited' if n in cited else 'not cited'}")
            st.caption(c["text"][:600] + ("…" if len(c["text"]) > 600 else ""))


# =========================================================================== 3. sidebar
with st.sidebar:
    st.markdown(f"👤 **{user['email']}**")
    if st.button("Log out", use_container_width=True):
        log_out()
        st.rerun()
    st.divider()

    st.header("📁 My documents")
    with st.form("upload", clear_on_submit=True):
        uploads = st.file_uploader("Add PDF, Markdown or text files", type=["pdf", "md", "txt"],
                                   accept_multiple_files=True)
        if st.form_submit_button("Add to my library", type="primary", use_container_width=True) and uploads:
            for f in uploads:
                with st.spinner(f"Saving and indexing {f.name}…"):
                    try:
                        added = add_document(files, collection, folder, f.name, f.getvalue())
                        if added:
                            st.success(f"{f.name}: {added} chunks")
                        else:
                            st.info(f"{f.name}: already in your library")
                    except Exception as e:
                        st.error(f"{f.name}: {e}")

    docs = list_documents(collection)
    st.caption(f"{len(docs)} document(s) · {collection.count()} chunks")
    for d in docs:
        left, right = st.columns([5, 1])
        left.markdown(f"**{d['source']}**  \n{d['chunks']} chunks")
        if right.button("🗑️", key=f"del-{d['doc_id']}", help=f"Delete {d['source']}"):
            delete_document(files, collection, folder, d["source"])
            st.rerun()

    if st.button("🔄 Reload my documents", use_container_width=True):
        st.session_state["synced"] = False
        st.rerun()

    st.divider()
    k = st.slider("Sources per answer", 1, 8, 4)
    st.caption(f"LLM: {llm_label()}")
    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()


# =========================================================================== 4. chat
st.title(f"🔎 {APP_NAME}")
st.caption(TAGLINE)

messages = st.session_state.setdefault("messages", [])
if not messages:
    with st.chat_message("assistant"):
        st.markdown(f"Hi! You have **{len(docs)}** document(s). What would you like to know?" if docs
                    else "Hi! Upload some documents in the sidebar, then ask me anything about them.")

for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            show_sources(msg.get("sources", []), msg.get("cited", []))

question = st.chat_input("Ask about your documents…", disabled=not docs)
if question:
    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        chunks, cited = [], []
        try:
            with st.spinner("Searching your documents…"):
                chunks = search(collection, question, k)
            text = st.write_stream(stream_answer(question, chunks)).strip()
            cited, invalid = find_citations(text, len(chunks))
            if invalid:
                st.warning(f"The answer mentions sources that don't exist: {invalid}")
            show_sources(chunks, cited)
        except requests.ConnectionError:
            text = "⚠️ Can't reach Ollama. Start it with `ollama serve`, or add GROQ_API_KEY."
            st.error(text)
        except Exception as e:
            text = f"⚠️ Something went wrong: {e}"
            st.error(text)
    messages.append({"role": "assistant", "content": text, "sources": chunks, "cited": cited})