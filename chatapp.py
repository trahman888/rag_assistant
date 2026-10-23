"""
RAG chat app — admin-managed knowledge base, chat for everyone.

RUN
    streamlit run chatapp.py

WHO SEES WHAT
    Not logged in   -> login / sign-up page only.
    Normal user     -> chat only (no sidebar). Answers come from the shared
                       knowledge base. They can attach files in the chat box
                       for extra context; those files are used for THIS chat
                       only and are never saved.
    Admin           -> the same chat + a sidebar to upload / delete / reload
                       the shared knowledge base. Admin emails: ADMIN_EMAILS
                       secret, default in auth.py.

LLM
    GROQ_API_KEY set  -> Groq (open-source gpt-oss)
    otherwise         -> local Ollama (answer.py settings)

FILES THIS NEEDS
    auth.py, knowledge_base.py, answer.py, vector_store.py (+ files it imports)
"""
import json
import os

import requests
import streamlit as st

from answer import LLM_MODEL, OLLAMA_URL, SYSTEM_PROMPT, find_citations
from auth import (CHECKING, current_user, is_admin, log_out, login_page, supabase_client,
                  supabase_enabled)
from knowledge_base import (LocalFiles, SupabaseFiles, add_document, delete_document, ensure_synced,
                            list_documents, read_attachment, search, shared_collection)

APP_NAME = "Personal Assistant"    # title on the login page, chat page and browser tab
TAGLINE = "Ask questions across the knowledge base and get answers grounded in sources."
GROQ_MODEL = "openai/gpt-oss-20b"
TOP_K = 4                        # sources per answer
ATTACH_MODE = "multiple"         # "multiple" = pick files, "directory" = pick a whole folder
MAX_ATTACHED_FILES = 10          # per chat, to keep memory use small
MAX_FILE_MB = 20

st.set_page_config(page_title=APP_NAME, page_icon="🔎", layout="centered")


# =========================================================================== 1. login gate
user = current_user()            # also restores a saved login after a page refresh
if user == CHECKING:             # first instant after a refresh: the browser is sending the saved login
    st.caption("Checking your login…")
    if st.button("Continue to login page"):
        st.rerun()
    st.stop()
if not user:
    login_page(APP_NAME, TAGLINE)
    st.stop()

admin = is_admin(user)
files = SupabaseFiles(supabase_client()) if supabase_enabled() else LocalFiles()
collection = shared_collection()

try:
    with st.spinner("Preparing the knowledge base…"):
        ensure_synced(files, collection)        # no-op unless the server just restarted
except Exception as e:
    st.error(f"Couldn't load the knowledge base: {e}")

st.session_state.setdefault("messages", [])
st.session_state.setdefault("attachments", [])     # chunks of files attached in this chat
st.session_state.setdefault("attached_names", [])


def new_chat():
    st.session_state["messages"] = []
    st.session_state["attachments"] = []
    st.session_state["attached_names"] = []


# =========================================================================== LLM
def _secret(name):
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name)


def sources_for_llm(chunks: list[dict]) -> str:
    """Numbered sources; the user's own attachments are labelled so the LLM knows
    which document "my file" / "this PDF" refers to."""
    blocks = []
    for n, c in enumerate(chunks, start=1):
        page = f", page {c['page']}" if c.get("page") else ""
        label = " (file attached by the user)" if c.get("attached") else " (knowledge base)"
        blocks.append(f"[{n}] {c['source']}{page}{label}\n{c['text']}")
    return "\n\n".join(blocks)


def stream_answer(question: str, chunks: list[dict]):
    note = ""
    if any(c.get("attached") for c in chunks):
        note = ("The user attached their own file(s), marked '(file attached by the user)'. "
                "When the question refers to 'my file', 'this PDF', 'the document' or similar, "
                "answer from those sources.\n\n")
    user_message = f"{note}<sources>\n{sources_for_llm(chunks)}\n</sources>\n\nQuestion: {question}"
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


def show_sources(sources: list[dict], cited: list[int]) -> None:
    if not sources:
        return
    with st.expander(f"📄 Sources ({len(cited)} cited of {len(sources)})"):
        for n, c in enumerate(sources, start=1):
            page = f", page {c['page']}" if c.get("page") else ""
            where = " · 📎 your attachment" if c.get("attached") else ""
            st.markdown(f"**[{n}] {c['source']}{page}**{where} — {'✅ cited' if n in cited else 'not cited'}")
            st.caption(c["text"][:600] + ("…" if len(c["text"]) > 600 else ""))


# =========================================================================== 2. admin sidebar
docs = list_documents(collection)

if admin:                         # normal users never call st.sidebar, so they don't get one
    with st.sidebar:
        st.header("🛠️ Knowledge base")
        st.caption("Admin only. Files added here are saved permanently and used for everyone's answers.")

        with st.form("upload", clear_on_submit=True):
            uploads = st.file_uploader("Add PDF, Markdown or text files", type=["pdf", "md", "txt"],
                                       accept_multiple_files=True)
            if st.form_submit_button("Add to knowledge base", type="primary", use_container_width=True) and uploads:
                for f in uploads:
                    with st.spinner(f"Saving and indexing {f.name}…"):
                        try:
                            added = add_document(files, collection, f.name, f.getvalue())
                            if added:
                                st.success(f"{f.name}: {added} chunks")
                            else:
                                st.info(f"{f.name}: already in the knowledge base")
                        except Exception as e:
                            st.error(f"{f.name}: {e}")
                docs = list_documents(collection)

        st.caption(f"{len(docs)} document(s) · {collection.count()} chunks")
        for d in docs:
            left, right = st.columns([5, 1])
            left.markdown(f"**{d['source']}**  \n{d['chunks']} chunks")
            if right.button("🗑️", key=f"del-{d['doc_id']}", help=f"Delete {d['source']}"):
                delete_document(files, collection, d["source"])
                st.rerun()

        if st.button("🔄 Reload from storage", use_container_width=True):
            ensure_synced(files, collection, force=True)
            st.rerun()

        st.divider()
        llm = f"`{GROQ_MODEL}` (Groq)" if _secret("GROQ_API_KEY") else f"`{LLM_MODEL}` (Ollama)"
        st.caption(f"LLM: {llm}")


# =========================================================================== 3. header
st.title(f"🔎 {APP_NAME}")
who_col, b1, b2 = st.columns([4, 1, 1], vertical_alignment="center")
who_col.caption(f"👤 {user['email']}" + (" · **admin**" if admin else ""))
if b1.button("New chat", use_container_width=True):
    new_chat()
    st.rerun()
if b2.button("Log out", use_container_width=True):
    log_out()
    st.rerun()

if st.session_state["attached_names"]:
    a, b = st.columns([4, 1], vertical_alignment="center")
    a.caption("📎 In this chat only, not saved: " + ", ".join(st.session_state["attached_names"]))
    if b.button("Remove", help="Forget the attached files"):
        st.session_state["attachments"], st.session_state["attached_names"] = [], []
        st.rerun()


# =========================================================================== 4. chat
messages = st.session_state["messages"]
if not messages:
    with st.chat_message("assistant"):
        if docs:
            st.markdown(f"Hi! I can answer questions from **{len(docs)}** document(s) in the knowledge base. "
                        "You can also attach your own files with 📎 for this chat.")
        else:
            st.markdown("Hi! The knowledge base is empty right now. "
                        + ("Add documents in the sidebar." if admin else "You can attach your own files with 📎 and ask about them."))

for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            show_sources(msg.get("sources", []), msg.get("cited", []))

submission = st.chat_input("Ask a question, or attach files with 📎…", accept_file=ATTACH_MODE,
                           file_type=["pdf", "md", "txt"], max_upload_size=MAX_FILE_MB)

if submission:
    question = (submission.text or "").strip()
    new_files = submission.files or []

    # ---- 4a. read any attached files (memory only)
    read_ok, problems = [], []
    for f in new_files:
        if len(st.session_state["attached_names"]) >= MAX_ATTACHED_FILES:
            problems.append(f"Only {MAX_ATTACHED_FILES} files per chat — skipped {f.name}")
            continue
        try:
            with st.spinner(f"Reading {f.name}…"):
                chunks = read_attachment(f.name, f.getvalue())
            st.session_state["attachments"] += chunks
            st.session_state["attached_names"].append(chunks[0]["source"])
            read_ok.append(chunks[0]["source"])
        except Exception as e:
            problems.append(str(e))

    user_text = question or "*(attached files)*"
    if new_files:
        user_text += "\n\n📎 " + ", ".join(f.name for f in new_files)
    messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        for p in problems:
            st.warning(p)
        chunks, cited = [], []

        if not question:                                   # files only, no question yet
            text = (f"I've read **{', '.join(read_ok)}** for this chat. What would you like to know?"
                    if read_ok else "I couldn't read those files.")
            st.markdown(text)
        else:
            try:
                with st.spinner("Searching…"):
                    chunks = search(collection, st.session_state["attachments"], question, TOP_K)
                if not chunks:
                    text = "There's nothing to search yet: the knowledge base is empty and no files are attached."
                    st.markdown(text)
                else:
                    text = st.write_stream(stream_answer(question, chunks)).strip()
                    cited, invalid = find_citations(text, len(chunks))
                    if invalid:
                        st.warning(f"The answer mentions sources that don't exist: {invalid}")
                    show_sources(chunks, cited)
            except requests.ConnectionError:
                text = "⚠️ Can't reach the language model right now. Please try again shortly."
                st.error(text)
            except Exception as e:
                text = f"⚠️ Something went wrong: {e}"
                st.error(text)

    messages.append({"role": "assistant", "content": text, "sources": chunks, "cited": cited})
    if read_ok:
        st.rerun()                # redraw so the "📎 In this chat only" banner appears right away