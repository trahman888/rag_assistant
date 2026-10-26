"""
RAG chat app — every user has their own private document library.

RUN
    streamlit run chatapp.py

WHO SEES WHAT
    Not logged in   -> login / sign-up page only.
    Logged in       -> chat + a "My documents" sidebar. Each user uploads,
                       deletes and searches ONLY their own documents; nobody
                       else can see or search them (separate storage folder,
                       separate search index — see knowledge_base.Library).
                       Limits per user: knowledge_base.MAX_DOCUMENTS etc.
                       Files attached in the chat box (📎) are used for THIS
                       chat only and are never saved.

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
from auth import (current_user, display_name, log_out, login_page, supabase_client, supabase_enabled,
                  update_name)
from privacy import PROMPT_NOTE as PRIVACY_NOTE
from privacy import Masker
from router import CHAT, chat_system_prompt, plan
import time as _time
import uuid
from images import load_image
from knowledge_base import (MAX_DOCUMENTS, MAX_FIGURES_PER_PDF, LimitError, Library, read_attachment)
from knowledge_base import MAX_FILE_MB as MAX_LIBRARY_FILE_MB

APP_NAME = "Personal Assistant"    # title on the login page, chat page and browser tab
TAGLINE = "Upload your own documents and get answers grounded in sources. Private to you."
GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_VISION_MODEL = "qwen/qwen3.8-27b"   # describes figures in uploaded PDFs
LOCAL_VISION_MODEL = "qwen2.5vl:3b"      # Ollama vision model for local mode: ollama pull qwen2.5vl:3b
TOP_K = 4                        # default Top K; each user can change theirs in the sidebar
ATTACH_MODE = "multiple"         # "multiple" = pick files, "directory" = pick a whole folder
MAX_ATTACHED_FILES = 10          # per chat, to keep memory use small
MAX_FILE_MB = 20

st.set_page_config(page_title=APP_NAME, page_icon="🔎", layout="centered")


# =========================================================================== 1. login gate
user = current_user()            # also restores the login after a page refresh
if not user:
    login_page(APP_NAME, TAGLINE)
    st.stop()

# This user's private library: their own storage folder + their own search index.
library = Library(user, supabase_client() if supabase_enabled() else None)
files = library.files
settings = library.settings()            # this user's own choices (Top K)

try:
    with st.spinner("Preparing your documents…"):
        library.sync()                   # no-op unless the server just restarted
except Exception as e:
    st.error(f"Couldn't load your documents: {e}")

if st.session_state.get("_chat_owner") != user["id"]:      # new browser session (e.g. a refresh) or new user
    st.session_state["messages"] = library.load_chat()       # bring back their last conversation
    st.session_state["attachments"], st.session_state["attached_names"] = [], []
    st.session_state["_chat_owner"] = user["id"]
    st.session_state["chat_id"] = uuid.uuid4().hex[:8]
st.session_state.setdefault("attachments", [])     # chunks of files attached in this chat
st.session_state.setdefault("attached_names", [])


def save_chat():
    try:
        library.save_chat(st.session_state["messages"])
    except Exception as e:
        st.toast(f"Couldn't save this conversation: {e}", icon="⚠️")


def new_chat():
    library.clear_chat()
    st.session_state["chat_id"] = uuid.uuid4().hex[:8]          # fresh keys for the 👍/👎 widgets
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
        label = " (file attached by the user)" if c.get("attached") else " (their library)"
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
    return "".join(llm_stream(messages, max_tokens=600, temperature=0.0)).strip()


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


VISION_MAX_TOKENS = 300      # a 2–6 sentence description needs ~150–250; Groq's free tier caps output per minute
VISION_RETRIES = 5


def _groq_vision(key: str, prompt: str, b64: str, on_wait=None) -> str:
    """One Groq vision request, staying inside the free tier's per-minute limits:
      * asks for at most VISION_MAX_TOKENS (a request asking for more than the
        per-minute output limit is refused outright);
      * turns the model's hidden "thinking" off (it would use up that limit);
      * on "429 rate limit", waits as long as Groq says, then tries again."""
    import time

    from groq import BadRequestError, Groq, RateLimitError

    client = Groq(api_key=key)
    extra = {"reasoning_effort": "none"}                     # supported by qwen/qwen3.8-27b
    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ]}]
    for attempt in range(1, VISION_RETRIES + 1):
        try:
            resp = client.chat.completions.create(model=GROQ_VISION_MODEL, temperature=0.1,
                                                  max_tokens=VISION_MAX_TOKENS, messages=messages, **extra)
            return resp.choices[0].message.content or ""
        except BadRequestError as e:
            if extra and "reasoning" in str(e).lower():     # a model without this option: retry without it
                extra = {}
                continue
            raise
        except RateLimitError as e:
            if attempt == VISION_RETRIES:
                raise
            wait = _retry_after_seconds(e)
            if on_wait:
                on_wait(wait)
            time.sleep(wait)
    raise RuntimeError("vision model kept refusing requests")


def _retry_after_seconds(error) -> float:
    """How long Groq asks us to wait: the retry-after header, or 'try again in 12.3s' in the message."""
    import re as _re
    try:
        header = error.response.headers.get("retry-after")
        if header:
            return min(65.0, float(header) + 1)
    except Exception:
        pass
    m = _re.search(r"try again in\s+(?:(\d+)m)?([\d.]+)s", str(error))
    if m:
        return min(65.0, int(m.group(1) or 0) * 60 + float(m.group(2)) + 1)
    return 20.0


def describe_image(png: bytes, prompt: str, on_wait=None) -> str:
    """Ask a vision model to describe one image (used when a user uploads a PDF).
    The prompt includes text from the PDF page, so it is privacy-masked like everything else."""
    import base64
    import re as _re

    masker = Masker()
    safe_prompt = masker.mask(prompt)
    b64 = base64.b64encode(png).decode()

    key = _secret("GROQ_API_KEY")
    if key:                                                   # online: Groq vision model
        text = _groq_vision(key, safe_prompt, b64, on_wait)
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
        picture = load_image(files, f["image"], library.folder)
        if picture:
            page = f", page {f['page']}" if f.get("page") else ""
            label = f"{f['caption']} — " if f.get("caption") else "🖼️ Figure from "
            st.image(picture, caption=f"{label}{f['source']}{page}", width=520)


def show_sources(sources: list[dict], cited: list[int], queries: list[str] | None = None) -> None:
    if not sources:
        return
    with st.expander(f"📄 Sources ({len(cited)} cited of {len(sources)})"):
        if queries:
            st.caption("🔎 Searched for: " + " · ".join(f"“{q}”" for q in queries))
        for n, c in enumerate(sources, start=1):
            page = f", page {c['page']}" if c.get("page") else ""
            where = " · 📎 your attachment" if c.get("attached") else (" · 🖼️ figure" if c.get("image") else "")
            st.markdown(f"**[{n}] {c['source']}{page}**{where} — {'✅ cited' if n in cited else 'not cited'}")
            if c.get("image"):                                     # show the picture itself
                picture = load_image(files, c["image"], library.folder)
                if picture:
                    st.image(picture, width=420)
            st.caption(c["text"][:600] + ("…" if len(c["text"]) > 600 else ""))


# =========================================================================== 👍 / 👎 feedback
def _record_feedback(i: int, key: str) -> None:
    """Called when the user clicks 👍 or 👎 under answer number i."""
    messages = st.session_state["messages"]
    msg, value = messages[i], st.session_state.get(key)
    msg["feedback"] = value
    if value is not None:
        question = messages[i - 1]["content"] if i > 0 and messages[i - 1]["role"] == "user" else ""
        try:
            library.add_feedback({
                "answer_id": msg.get("time") or f"{st.session_state['chat_id']}-{i}",
                "time": _time.strftime("%Y-%m-%dT%H:%M:%S"), "rating": "up" if value == 1 else "down",
                "question": question, "answer": msg["content"][:2000], "queries": msg.get("queries", []),
                "sources": [{"source": c["source"], "page": c.get("page"), "cited": n in msg.get("cited", []),
                             "text": "(attached file, not saved)" if c.get("attached") else c["text"][:400]}
                            for n, c in enumerate(msg.get("sources", []), start=1)]})
        except Exception as e:
            st.toast(f"Couldn't save your rating: {e}", icon="⚠️")
    save_chat()


def feedback_widget(i: int) -> None:
    msg = st.session_state["messages"][i]
    if not msg.get("sources"):                 # only document answers are rated, not small talk
        return
    key = f"fb-{st.session_state['chat_id']}-{i}"
    st.feedback("thumbs", key=key, default=msg.get("feedback"), on_change=_record_feedback, args=(i, key))


# =========================================================================== delete confirmation
@st.dialog("Delete document?", icon="⚠️")
def confirm_delete(name: str, n_chunks: int) -> None:
    """Pop-up shown before a user deletes one of their documents."""
    st.markdown(f"**{name}** ({n_chunks} chunks)")
    st.warning("This permanently removes the file and its figures from your library. "
               "You won't get answers from it any more, and this can't be undone.")
    cancel_col, delete_col = st.columns(2)
    if cancel_col.button("Cancel", use_container_width=True):
        st.rerun()
    if delete_col.button("Delete", type="primary", use_container_width=True):
        with st.spinner("Deleting…"):
            library.delete(name)
        st.session_state["_deleted_note"] = name
        st.rerun()


# =========================================================================== 2. "My documents" sidebar
docs = library.documents()

with st.sidebar:
    view = st.segmented_control("View", ["💬 Chat", "📊 Evaluation"], default="💬 Chat", key="view",
                                label_visibility="collapsed") or "💬 Chat"
    st.header("📁 My documents")
    st.caption("Private to you: other users can't see or search these files.")
    st.caption(f"Limits: {MAX_DOCUMENTS} documents · {MAX_LIBRARY_FILE_MB} MB per file · "
               f"{MAX_FIGURES_PER_PDF} figures read per PDF")

    with st.form("upload", clear_on_submit=True):
        uploads = st.file_uploader("Add PDF, Markdown or text files", type=["pdf", "md", "txt"],
                                   accept_multiple_files=True)
        read_images = st.checkbox("Also read images inside PDFs", value=True,
                                  help="Charts, diagrams and photos are described by a vision AI "
                                       f"({vision_label()}) so they can be searched. Slower uploads.")
        if st.form_submit_button("Add to my documents", type="primary", use_container_width=True) and uploads:
            for f in uploads:
                bar = st.progress(0.0, text=f"Reading {f.name}…")

                state = {"done": 0, "total": 1}

                def progress(done, total, name=f.name, bar=bar, state=state):
                    state.update(done=done, total=total)
                    bar.progress(done / total, text=f"{name}: describing image {done} of {total}…")

                def waiting(seconds, name=f.name, bar=bar, state=state):
                    bar.progress(state["done"] / state["total"],
                                 text=f"{name}: Groq free-tier limit reached, waiting {seconds:.0f} s before "
                                      f"image {state['done']} of {state['total']}…")

                image_errors = []
                try:
                    added, figures = library.add(f.name, f.getvalue(),
                                                 describe_fn=(lambda png, prompt, w=waiting: describe_image(png, prompt, on_wait=w))
                                                 if read_images else None,
                                                 progress=progress, on_error=image_errors.append)
                    bar.empty()
                    if added:
                        st.success(f"{f.name}: {added} chunks" + (f" + {figures} figures" if figures else ""))
                        if image_errors:          # say so instead of silently skipping
                            st.warning(f"{len(image_errors)} image(s) couldn't be read by the vision model "
                                       f"({vision_label()}). First error — {image_errors[0][:300]}")
                    else:
                        st.info(f"{f.name}: already in your documents. To re-read its figures, "
                                "delete it with 🗑️ and upload it again.")
                except LimitError as e:
                    bar.empty()
                    st.warning(str(e))
                except Exception as e:
                    bar.empty()
                    st.error(f"{f.name}: {e}")
            docs = library.documents()

    deleted = st.session_state.pop("_deleted_note", None)
    if deleted:
        st.toast(f"Deleted {deleted}", icon="🗑️")      # small pop-up that disappears by itself
    st.caption(f"{len(docs)} of {MAX_DOCUMENTS} documents")
    for d in docs:
        left, right = st.columns([5, 1])
        figs = f" · {d['figures']} figures" if d.get("figures") else ""
        left.markdown(f"**{d['source']}**  \n{d['chunks'] - d.get('figures', 0)} chunks{figs}")
        if right.button("🗑️", key=f"del-{d['doc_id']}", help=f"Delete {d['source']}"):
            confirm_delete(d["source"], d["chunks"])        # opens a "are you sure?" window

    st.divider()
    st.subheader("⚙️ Answer settings")
    current_k = settings["top_k"]
    new_k = st.slider("Top K (passages retrieved per answer)", min_value=2, max_value=10, value=current_k,
                      help="Top K = how many of the best-matching passages are retrieved and given to the AI "
                           "for each answer. The AI then cites only the ones it actually uses. Higher K gives "
                           "fuller answers but uses more of the AI's free quota. Only affects your answers.")
    st.caption(f"Currently retrieving the top **{current_k}** passages per question.")
    if new_k != current_k:
        try:
            library.save_setting("top_k", new_k)
            settings["top_k"] = new_k
            st.toast(f"Top K set to {new_k}", icon="⚙️")
            st.rerun()
        except Exception as e:
            st.error(f"Couldn't save the setting: {e}")

    if st.button("🔄 Reload from storage", use_container_width=True):
        library.sync(force=True)
        st.rerun()

    st.divider()
    llm = f"`{GROQ_MODEL}` (Groq)" if _secret("GROQ_API_KEY") else f"`{LLM_MODEL}` (Ollama)"
    st.caption(f"LLM: {llm}")


# =========================================================================== evaluation page
if view == "📊 Evaluation":
    import eval_page

    def _answer_for_eval(question, passages):
        return "".join(stream_answer(question, passages))

    eval_page.render(library, settings, llm_complete,
                     plan_fn=lambda q: plan(q, False, llm_complete)[1], answer_fn=_answer_for_eval)
    st.stop()


# =========================================================================== 3. header
st.title(f"🔎 {APP_NAME}")
who_col, b1, b2 = st.columns([4, 1, 1], vertical_alignment="center")
with who_col.popover(f"👤 {display_name(user)}"):
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
        if not docs:
            st.caption("Your library is empty. Add documents in the sidebar (they stay private to you), "
                       "or attach a file here with 📎 for this chat only.")

for i, msg in enumerate(messages):
    with st.chat_message(msg["role"]):
        st.markdown(md_safe(msg["content"]))
        if msg["role"] == "assistant":
            show_figures(msg.get("figures", []))
            show_privacy_note(msg.get("hidden", 0))
            show_sources(msg.get("sources", []), msg.get("cited", []), msg.get("queries"))
            feedback_widget(i)

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

    answer_box = st.chat_message("assistant")
    with answer_box:
        for p in problems:
            st.warning(p)
        chunks, cited, hidden, figures, search_queries = [], [], 0, [], []

        if not question:                                   # files only, no question yet
            text = (f"I've read **{', '.join(read_ok)}** for this chat. What would you like to know?"
                    if read_ok else "I couldn't read those files.")
            st.markdown(text)
        else:
            try:
                # ---- 4b. decide first: small talk, or a question for the documents?
                with st.spinner("Thinking…"):
                    # one LLM call: is this small talk? and if not, which search phrases to use
                    kind, search_queries = plan(question, bool(st.session_state["attachments"]),
                                                llm_complete, history=messages[:-1])

                st.session_state["_hidden_count"] = 0
                if kind == CHAT:                           # small talk: no search, no sources
                    text = show_stream(stream_chat_reply(question, messages[:-1], [d["source"] for d in docs]))
                    hidden = st.session_state["_hidden_count"]
                    show_privacy_note(hidden)
                else:                                      # ---- 4c. search + cited answer
                    with st.spinner("Searching…"):
                        chunks = library.search(question, st.session_state["attachments"],
                                                settings["top_k"], queries=search_queries)
                    if not chunks:
                        text = "There's nothing to search yet: your library is empty and no files are attached."
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
                        show_sources(chunks, cited, search_queries)
            except requests.ConnectionError:
                text = "⚠️ Can't reach the language model right now. Please try again shortly."
                st.error(text)
            except Exception as e:
                text = f"⚠️ Something went wrong: {e}"
                st.error(text)

    messages.append({"role": "assistant", "content": text, "sources": chunks, "cited": cited, "hidden": hidden,
                     "figures": figures, "queries": search_queries, "time": _time.strftime("%Y%m%d%H%M%S") + uuid.uuid4().hex[:4]})
    save_chat()
    with answer_box:
        feedback_widget(len(messages) - 1)                   # so a page refresh brings the conversation back
    if read_ok:
        st.rerun()                # redraw so the "📎 In this chat only" banner appears right away
