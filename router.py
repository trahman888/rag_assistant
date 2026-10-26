"""
Decide what to do with a message BEFORE searching: chat, or search the documents?

WHY
    Searching the knowledge base for "hello" or "thanks!" finds random passages,
    and the answer then quotes them. Real assistants first work out what kind of
    message it is. This is called "query routing".

TWO STEPS (fast first, smart second)
    1. Rules (instant, free): obvious small talk — greetings, thanks, goodbyes,
       "how are you", "who are you", "what can you do" — goes straight to CHAT.
    2. LLM (one short call): anything else is shown to the LLM, which returns the
       route plus rewritten search queries as JSON. If that call fails, we fall
       back to SEARCH with the question as typed.

    Messages that mention an attached file ("my pdf", "this document", ...) skip
    both steps and always SEARCH.

ROUTES
    "chat"    -> a short friendly reply, no documents, no sources box
    "search"  -> the normal RAG answer with citations

QUERY REWRITING (plan())
    The same LLM call that routes the message also rewrites it into 2–3 search
    phrases, so it costs no extra call:
      * a complete, standalone version of the question — follow-ups like
        "and what about figure 2?" are rewritten using the recent conversation;
      * the key technical terms and names (good for keyword search);
      * for figure requests, a description of what the figure would show.
    knowledge_base.search() searches all of them plus the original question.
"""
import json
import re

CHAT, SEARCH = "chat", "search"

_SMALL_TALK = re.compile(r"""^(
      hi | hello | hey | hiya | yo | howdy | greetings | salam | assalamu?\s*alaikum
    | good\s+(morning|afternoon|evening|night|day)
    | how\s+are\s+(you|u)(\s+doing)?(\s+today)? | how('?s|\s+is)\s+it\s+going | what'?s\s+up | sup
    | (i'?m\s+)?(fine|good|great|ok(ay)?|not\s+bad)(,?\s+thanks?( you)?)?
    | (thanks?|thank\s+(you|u))(\s+(so\s+much|a\s+lot|very\s+much|again))? | thx | ty | cheers | appreciate\s+it
    | (ok(ay)?|cool|nice|great|awesome|got\s+it|perfect|sounds\s+good)
    | bye | goodbye | see\s+(you|ya)(\s+later)? | good\s*bye | take\s+care
    | who\s+are\s+you | what\s+are\s+you | what\s+is\s+your\s+name | what('?s|\s+is)\s+your\s+name
    | what\s+can\s+you\s+do | how\s+can\s+you\s+help(\s+me)? | help
)$""", re.IGNORECASE | re.VERBOSE)

_ABOUT_ATTACHMENT = re.compile(r"\b(my|this|the|attached|uploaded)\s+(pdf|file|document|doc|notes?|paper|report)s?\b",
                               re.IGNORECASE)

def _clean(text: str) -> str:
    text = re.sub(r"[!?.,:;~\s]+$", "", text.strip())      # trailing punctuation / emoji-ish spacing
    text = re.sub(r"[\U0001F300-\U0001FAFF☀-➿]", "", text)   # emoji
    return re.sub(r"\s+", " ", text).strip()


def quick_route(text: str, has_attachments: bool) -> str | None:
    """Rule-based step. Returns CHAT / SEARCH when sure, None when the LLM should decide."""
    t = _clean(text)
    if not t:
        return CHAT
    if has_attachments and _ABOUT_ATTACHMENT.search(t):
        return SEARCH
    # "hi there, how are you?" -> ["hi there", "how are you"]; every part must be small talk
    parts = [p.strip() for p in re.split(r"[,.!?;]+", t) if p.strip()]
    parts = [re.sub(r"\s+(there|all|everyone|guys|friend|buddy|bot|assistant)$", "", p, flags=re.I) for p in parts]
    if parts and all(_SMALL_TALK.match(p) for p in parts):
        return CHAT
    return None


PLAN_PROMPT = """You prepare messages for a question-answering assistant that searches a library of documents.

1. Decide the route:
   "chat"   = small talk or about the conversation itself: greetings, thanks, feelings, jokes,
              compliments, goodbyes, questions about the assistant.
   "search" = anything that needs information from the documents: factual questions, requests
              to explain, summarise, compare, list, find or SHOW something (a figure, chart,
              table, picture), or anything about a topic, method, person, number or file.
   When unsure, choose "search".

2. If "search", write 2 or 3 short search queries (each under 15 words) that would find the
   answer in the documents:
   - first: the question rewritten as a complete, standalone question. Resolve words like
     "it", "that", "this figure", "the second one" using the recent conversation.
   - second: the key terms only — technical words, acronyms, names, numbers.
   - if the user wants a figure/chart/diagram/table: a phrase describing what it would show.
   Don't answer the question; don't invent facts or names that aren't in the conversation.

Reply with JSON only, no other text:
{"route": "search", "queries": ["...", "..."]}   or   {"route": "chat", "queries": []}"""


def _history_text(history: list[dict] | None) -> str:
    if not history:
        return "(none)"
    lines = []
    for m in history[-4:]:
        content = re.sub(r"\s+", " ", m.get("content", ""))[:300]
        lines.append(f"{m.get('role', 'user')}: {content}")
    return "\n".join(lines)


def _parse_plan(reply: str) -> tuple[str, list[str]] | None:
    reply = re.sub(r"<think>.*?</think>", "", reply or "", flags=re.S)
    m = re.search(r"\{.*\}", reply, re.S)
    if m:
        try:
            data = json.loads(m.group(0))
            route_ = CHAT if str(data.get("route", "")).lower().strip() == "chat" else SEARCH
            queries = [str(q).strip() for q in (data.get("queries") or []) if str(q).strip()]
            return route_, [q[:200] for q in queries[:3]]
        except (ValueError, AttributeError):
            pass
    up = (reply or "").upper()                   # the model ignored JSON: fall back to a plain word
    if re.search(r"\bCHAT\b", up) and "SEARCH" not in up:
        return CHAT, []
    return None


def plan(text: str, has_attachments: bool, llm_complete, history: list[dict] | None = None) -> tuple[str, list[str]]:
    """Route AND rewrite in one step. Returns (route, extra_search_queries).
    llm_complete(messages) -> str is the app's non-streaming LLM call."""
    quick = quick_route(text, has_attachments)
    if quick == CHAT:
        return CHAT, []                          # obvious small talk: no LLM call at all
    try:
        reply = llm_complete([
            {"role": "system", "content": PLAN_PROMPT},
            {"role": "user", "content": f"Recent conversation:\n{_history_text(history)}\n\nNew message: {text}"},
        ])
        parsed = _parse_plan(reply)
    except Exception:
        parsed = None
    if parsed is None:
        return SEARCH, []                        # router failed: search with the question as typed
    route_, queries = parsed
    if quick == SEARCH:                          # e.g. "summarise my pdf" with an attachment
        route_ = SEARCH
    return route_, queries


def route(text: str, has_attachments: bool, llm_complete) -> str:
    """Route only (kept for compatibility); see plan()."""
    return plan(text, has_attachments, llm_complete)[0]


def chat_system_prompt(app_name: str, doc_names: list[str], attached: list[str], user_name: str = "") -> str:
    docs = ", ".join(doc_names[:15]) + (" …" if len(doc_names) > 15 else "") if doc_names else "none yet"
    files = ", ".join(attached) if attached else "none"
    return f"""You are {app_name}, a friendly assistant that answers questions from a document library.

Right now the user is making small talk, not asking about the documents.
Reply naturally and briefly (1–3 sentences), in the user's language.
If it fits, mention what you can help with: questions about the documents in the library,
or about files they attach with the 📎 button. Don't say how many documents there are or
list their names unless the user asks.
Do not answer factual questions from your own knowledge and do not invent document content.

The user's name: {user_name or "unknown"} (use their first name now and then, naturally).
Documents in the library: {docs}
Files the user attached in this chat: {files}"""
