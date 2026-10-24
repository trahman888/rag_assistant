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
import re
import os

import requests
import streamlit as st

from answer import LLM_MODEL, OLLAMA_URL, SYSTEM_PROMPT, find_citations
from auth import (current_user, display_name, is_admin, log_out, login_page, supabase_client,
                  supabase_enabled, update_name)
from privacy import PROMPT_NOTE as PRIVACY_NOTE
from privacy import Masker
from router import CHAT, chat_system_prompt, route
from images import load_image
from knowledge_base import (SHARED_FOLDER, LocalFiles, SupabaseFiles, add_document, delete_document,
                            ensure_synced, list_documents, read_attachment, search, shared_collection)

APP_NAME = "Personal Assistant"    # title on the login page, chat page and browser tab
TAGLINE = "Ask questions across the knowledge base and get answers grounded in sources."
GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_VISION_MODEL = "qwen/qwen3.8-27b"   # reads images (admin uploads only)
LOCAL_VISION_MODEL = "qwen2.5vl:3b"      # Ollama vision model for local mode: ollama pull qwen2.5vl:3b
TOP_K = 4                        # sources per answer
ATTACH_MODE = "multiple"         # "multiple" = pick files, "directory" = pick a whole folder
MAX_ATTACHED_FILES = 10          # per chat, to keep memory use small
MAX_FILE_MB = 20

st.set_page_config(page_title=APP_NAME, page_icon="🔎", layout="centered")


# =========================================================================== 1. login gate
user = current_user()            # also restores the login after a page refresh
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


def llm_stream(messages: list[dict], max_tokens: int = 1024, temperature: float = 0.1):
    """EVERY LLM call goes through here, so private details are always masked:
    emails, phones, card numbers… become <EMAIL_1>, <PHONE_1>… before sending,
    and the real values are put back into the reply before the user sees it."""
    masker = Masker()
    safe = masker.mask_messages(messages)
    if masker.hidden:                                          # explain the placeholders to the LLM
        safe[0] = {**safe[0], "content": safe[0]["content"] + "\n\n" + PRIVACY_NOTE}
    st.session_state["_hidden_count"] = masker.hidden
    yield from masker.unmask_stream(_raw_llm_stream(safe, max_tokens, temperature))


def _raw_llm_stream(messages: list[dict], max_tokens: int, temperature: float):
    """Send (already masked) messages to Groq or Ollama and yield the reply piece by piece."""
    key = _secret("GROQ_API_KEY")
    if key:                                                   # online: Groq
        from groq import Groq
        stream = Groq(api_key=key).chat.completions.create(
            model=GROQ_MODEL, messages=messages, temperature=temperature,
            max_tokens=max_tokens, reasoning_effort="low", stream=True)
        for part in stream:
            yield part.choices[0].delta.content or ""
        return

    with requests.post(                                       # local: Ollama
        f"{OLLAMA_URL}/api/chat",
        json={"model": LLM_MODEL, "messages": messages, "stream": True, "think": False,
              "options": {"temperature": temperature, "num_ctx": 4096, "num_predict": min(max_tokens, 400)}},
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


def llm_complete(messages: list[dict]) -> str:
    """Same LLM, whole reply at once (used by the router)."""
    return "".join(llm_stream(messages, max_tokens=300, temperature=0.0)).strip()


def stream_answer(question: str, chunks: list[dict]):
    """RAG answer: question + numbered sources -> cited answer."""
    note = ""
    if any(c.get("attached") for c in chunks):
        note = ("The user attached their own file(s), marked '(file attached by the user)'. "
                "When the question refers to 'my file', 'this PDF', 'the document' or similar, "
                "answer from those sources.\n\n")
    if any(c.get("image") for c in chunks):
        note += ("Sources that start with '[Figure on page N]' describe images from the documents. "
                 "The app shows those images to the user right under your answer, so when the user asks "
                 "to see a chart, diagram or picture, briefly say what it shows and cite it; never say you "
                 "can't display images.\n\n")
    user_message = f"{note}<sources>\n{sources_for_llm(chunks)}\n</sources>\n\nQuestion: {question}"
    return llm_stream([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_message}])


def stream_chat_reply(question: str, history: list[dict], doc_names: list[str]):
    """Small-talk reply: no documents, a little recent conversation for context."""
    system = chat_system_prompt(APP_NAME, doc_names, st.session_state["attached_names"], display_name(user))
    recent = [{"role": m["role"], "content": m["content"]} for m in history[-6:]]
    return llm_stream([{"role": "system", "content": system}, *recent,
                       {"role": "user", "content": question}], max_tokens=300, temperature=0.5)


def vision_label() -> str:
    return f"{GROQ_VISION_MODEL} on Groq" if _secret("GROQ_API_KEY") else f"{LOCAL_VISION_MODEL} on Ollama"


def describe_image(png: bytes, prompt: str) -> str:
    """Ask a vision model to describe one image (used when the admin uploads a PDF).
    The prompt includes text from the PDF page, so it is privacy-masked like everything else."""
    import base64
    import re as _re

    masker = Masker()
    safe_prompt = masker.mask(prompt)
    b64 = base64.b64encode(png).decode()

    key = _secret("GROQ_API_KEY")
    if key:                                                   # online: Groq vision model
        from groq import Groq
        resp = Groq(api_key=key).chat.completions.create(
            model=GROQ_VISION_MODEL, temperature=0.1, max_tokens=1024,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": safe_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]}])
        text = resp.choices[0].message.content or ""
    else:                                                     # local: Ollama vision model
        r = requests.post(f"{OLLAMA_URL}/api/chat", timeout=(10, 300), json={
            "model": LOCAL_VISION_MODEL, "stream": False, "think": False,
            "messages": [{"role": "user", "content": safe_prompt, "images": [b64]}],
            "options": {"temperature": 0.1, "num_ctx": 4096}})
        if r.status_code == 404:
            raise RuntimeError(f"Vision model '{LOCAL_VISION_MODEL}' not found. Run:  ollama pull {LOCAL_VISION_MODEL}")
        r.raise_for_status()
        text = r.json()["message"]["content"]
    text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.S).strip()   # drop any visible reasoning
    return masker.unmask(text)


def md_safe(text: str) -> str:
    """Streamlit reads text between two $ signs as a math formula ("$8 to $2" breaks),
    so dollar signs are escaped for display. The stored text keeps the plain $."""
    return text.replace("$", "\\$")


def show_stream(pieces) -> str:
    """Stream a reply onto the page (with $ made safe) and return the plain text."""
    collected = []

    def safe_pieces():
        for p in pieces:
            collected.append(p)
            yield md_safe(p)

    st.write_stream(safe_pieces())
    return "".join(collected).strip()


def show_privacy_note(hidden: int) -> None:
    if hidden:
        st.caption(f"🔒 {hidden} private detail{'s' if hidden != 1 else ''} (emails, phone numbers…) "
                   "hidden from the AI and restored here.")


_ASKS_FOR_PICTURE = re.compile(r"\b(show|see|display|view|look)\b|\b(image|images|picture|pictures|photo|photos|"
                               r"figure|figures|fig|chart|charts|graph|graphs|plot|diagram|diagrams|table|screenshot)\b",
                               re.IGNORECASE)
MAX_INLINE_FIGURES = 3


def figures_to_show(question: str, sources: list[dict], cited: list[int]) -> list[dict]:
    """Which stored images to show right under the answer:
    every figure the answer cited, or, if the user asked to see a picture
    ("show me the chart…") and none was cited, the best-matching figure."""
    figures = [sources[n - 1] for n in cited if sources[n - 1].get("image")]
    if not figures and _ASKS_FOR_PICTURE.search(question or ""):
        figures = [s for s in sources if s.get("image")][:1]
    unique = list({f["image"]: f for f in figures}.values())
    return unique[:MAX_INLINE_FIGURES]


def show_figures(figures: list[dict]) -> None:
    """Draw figures directly under the answer (not hidden in the Sources box)."""
    for f in figures:
        picture = load_image(files, f["image"], SHARED_FOLDER)
        if picture:
            page = f", page {f['page']}" if f.get("page") else ""
            label = f"{f['caption']} — " if f.get("caption") else "🖼️ Figure from "
            st.image(picture, caption=f"{label}{f['source']}{page}", width=520)


def show_sources(sources: list[dict], cited: list[int]) -> None:
    if not sources:
        return
    with st.expander(f"📄 Sources ({len(cited)} cited of {len(sources)})"):
        for n, c in enumerate(sources, start=1):
            page = f", page {c['page']}" if c.get("page") else ""
            where = " · 📎 your attachment" if c.get("attached") else (" · 🖼️ figure" if c.get("image") else "")
            st.markdown(f"**[{n}] {c['source']}{page}**{where} — {'✅ cited' if n in cited else 'not cited'}")
            if c.get("image"):                                     # show the picture itself
                picture = load_image(files, c["image"], SHARED_FOLDER)
                if picture:
                    st.image(picture, width=420)
            st.caption(c["text"][:600] + ("…" if len(c["text"]) > 600 else ""))


# =========================================================================== delete confirmation
@st.dialog("Delete document?", icon="⚠️")
def confirm_delete(name: str, n_chunks: int) -> None:
    """Pop-up shown before the admin deletes a knowledge-base document."""
    st.markdown(f"**{name}** ({n_chunks} chunks)")
    st.warning("This permanently removes the file from storage and from the knowledge base. "
               "Nobody will get answers from it any more, and this can't be undone.")
    cancel_col, delete_col = st.columns(2)
    if cancel_col.button("Cancel", use_container_width=True):
        st.rerun()
    if delete_col.button("Delete", type="primary", use_container_width=True):
        with st.spinner("Deleting…"):
            delete_document(files, collection, name)
        st.session_state["_deleted_note"] = name
        st.rerun()


# =========================================================================== 2. admin sidebar
docs = list_documents(collection)

if admin:                         # normal users never call st.sidebar, so they don't get one
    with st.sidebar:
        st.header("🛠️ Knowledge base")
        st.caption("Admin only. Files added here are saved permanently and used for everyone's answers.")

        with st.form("upload", clear_on_submit=True):
            uploads = st.file_uploader("Add PDF, Markdown or text files", type=["pdf", "md", "txt"],
                                       accept_multiple_files=True)
            read_images = st.checkbox("Also read images inside PDFs", value=True,
                                      help="Charts, diagrams and photos are described by a vision AI "
                                           f"({vision_label()}) so they can be searched. Slower uploads.")
            if st.form_submit_button("Add to knowledge base", type="primary", use_container_width=True) and uploads:
                for f in uploads:
                    bar = st.progress(0.0, text=f"Reading {f.name}…")

                    def progress(done, total, name=f.name, bar=bar):
                        bar.progress(done / total, text=f"{name}: describing image {done} of {total}…")

                    image_errors = []
                    try:
                        added, figures = add_document(files, collection, f.name, f.getvalue(),
                                                      describe_fn=describe_image if read_images else None,
                                                      progress=progress, on_error=image_errors.append)
                        bar.empty()
                        if added:
                            st.success(f"{f.name}: {added} chunks" + (f" + {figures} figures" if figures else ""))
                            if image_errors:          # tell the admin instead of silently skipping
                                st.warning(f"{len(image_errors)} image(s) couldn't be read by the vision model "
                                           f"({vision_label()}). First error — {image_errors[0][:300]}")
                        else:
                            st.info(f"{f.name}: already in the knowledge base. To re-read its figures, "
                                    "delete it with 🗑️ and upload it again.")
                    except Exception as e:
                        bar.empty()
                        st.error(f"{f.name}: {e}")
                docs = list_documents(collection)

        deleted = st.session_state.pop("_deleted_note", None)
        if deleted:
            st.toast(f"Deleted {deleted}", icon="🗑️")      # small pop-up that disappears by itself
        st.caption(f"{len(docs)} document(s) · {collection.count()} chunks")
        for d in docs:
            left, right = st.columns([5, 1])
            figs = f" · {d['figures']} figures" if d.get("figures") else ""
            left.markdown(f"**{d['source']}**  \n{d['chunks'] - d.get('figures', 0)} chunks{figs}")
            if right.button("🗑️", key=f"del-{d['doc_id']}", help=f"Delete {d['source']}"):
                confirm_delete(d["source"], d["chunks"])        # opens a "are you sure?" window

        if st.button("🔄 Reload from storage", use_container_width=True):
            ensure_synced(files, collection, force=True)
            st.rerun()

        st.divider()
        llm = f"`{GROQ_MODEL}` (Groq)" if _secret("GROQ_API_KEY") else f"`{LLM_MODEL}` (Ollama)"
        st.caption(f"LLM: {llm}")


# =========================================================================== 3. header
st.title(f"🔎 {APP_NAME}")
who_col, b1, b2 = st.columns([4, 1, 1], vertical_alignment="center")
with who_col.popover(f"👤 {display_name(user)}" + (" · admin" if admin else "")):
    st.caption(f"Signed in as {user['email']}")
    with st.form("rename", clear_on_submit=True, border=False):
        new_name = st.text_input("Your name", placeholder=display_name(user), max_chars=60)
        if st.form_submit_button("Save name"):
            try:
                update_name(new_name)
                st.rerun()
            except Exception as e:
                st.error(str(e))
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
first_name = display_name(user).split()[0]
if not messages:
    with st.chat_message("assistant"):
        st.markdown(f"Hi {first_name}! How can I help you today?")
        if admin and not docs:                 # only the admin is told the knowledge base is empty
            st.caption("The knowledge base is empty. Add documents in the sidebar.")

for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(md_safe(msg["content"]))
        if msg["role"] == "assistant":
            show_figures(msg.get("figures", []))
            show_privacy_note(msg.get("hidden", 0))
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
        st.markdown(md_safe(user_text))

    with st.chat_message("assistant"):
        for p in problems:
            st.warning(p)
        chunks, cited, hidden, figures = [], [], 0, []

        if not question:                                   # files only, no question yet
            text = (f"I've read **{', '.join(read_ok)}** for this chat. What would you like to know?"
                    if read_ok else "I couldn't read those files.")
            st.markdown(text)
        else:
            try:
                # ---- 4b. decide first: small talk, or a question for the documents?
                with st.spinner("Thinking…"):
                    kind = route(question, bool(st.session_state["attachments"]), llm_complete)

                st.session_state["_hidden_count"] = 0
                if kind == CHAT:                           # small talk: no search, no sources
                    text = show_stream(stream_chat_reply(question, messages[:-1], [d["source"] for d in docs]))
                    hidden = st.session_state["_hidden_count"]
                    show_privacy_note(hidden)
                else:                                      # ---- 4c. search + cited answer
                    with st.spinner("Searching…"):
                        chunks = search(collection, st.session_state["attachments"], question, TOP_K)
                    if not chunks:
                        text = "There's nothing to search yet: the knowledge base is empty and no files are attached."
                        st.markdown(text)
                    else:
                        text = show_stream(stream_answer(question, chunks))
                        hidden = st.session_state["_hidden_count"]
                        cited, invalid = find_citations(text, len(chunks))
                        if invalid:
                            st.warning(f"The answer mentions sources that don't exist: {invalid}")
                        show_privacy_note(hidden)
                        figures = figures_to_show(question, chunks, cited)
                        show_figures(figures)
                        show_sources(chunks, cited)
            except requests.ConnectionError:
                text = "⚠️ Can't reach the language model right now. Please try again shortly."
                st.error(text)
            except Exception as e:
                text = f"⚠️ Something went wrong: {e}"
                st.error(text)

    messages.append({"role": "assistant", "content": text, "sources": chunks, "cited": cited, "hidden": hidden,
                     "figures": figures})
    if read_ok:
        st.rerun()                # redraw so the "📎 In this chat only" banner appears right away
