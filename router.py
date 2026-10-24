"""
Decide what to do with a message BEFORE searching: chat, or search the documents?

WHY
    Searching the knowledge base for "hello" or "thanks!" finds random passages,
    and the answer then quotes them. Real assistants first work out what kind of
    message it is. This is called "query routing".

TWO STEPS (fast first, smart second)
    1. Rules (instant, free): obvious small talk — greetings, thanks, goodbyes,
       "how are you", "who are you", "what can you do" — goes straight to CHAT.
    2. LLM router (one short extra LLM call): anything else is shown to the LLM
       with a tiny prompt that must answer CHAT or SEARCH. If that call fails,
       we fall back to SEARCH, which is the old behaviour.

    Messages that mention an attached file ("my pdf", "this document", ...) skip
    both steps and always SEARCH.

ROUTES
    "chat"    -> a short friendly reply, no documents, no sources box
    "search"  -> the normal RAG answer with citations
"""
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

ROUTER_PROMPT = """You route messages for a document question-answering assistant.

Reply CHAT if the message is small talk or about the conversation itself: greetings,
thanks, feelings, jokes, compliments, goodbyes, or questions about the assistant.

Reply SEARCH if answering it needs information from documents: any factual question,
request to explain, summarise, compare, list or find something, or anything about a
topic, person, method, number or file.

When unsure, reply SEARCH.
Reply with exactly one word: CHAT or SEARCH."""


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


def route(text: str, has_attachments: bool, llm_complete) -> str:
    """Full decision. llm_complete(messages) -> str is the app's non-streaming LLM call."""
    quick = quick_route(text, has_attachments)
    if quick:
        return quick
    try:
        reply = llm_complete([{"role": "system", "content": ROUTER_PROMPT},
                              {"role": "user", "content": text}])
        return CHAT if re.search(r"\bCHAT\b", reply.upper()) and "SEARCH" not in reply.upper() else SEARCH
    except Exception:
        return SEARCH


def chat_system_prompt(app_name: str, doc_names: list[str], attached: list[str], user_name: str = "") -> str:
    docs = ", ".join(doc_names[:15]) + (" …" if len(doc_names) > 15 else "") if doc_names else "none yet"
    files = ", ".join(attached) if attached else "none"
    return f"""You are {app_name}, a friendly assistant that answers questions from a document library.

Right now the user is making small talk, not asking about the documents.
Reply naturally and briefly (1–3 sentences), in the user's language.
If it fits, mention what you can help with: questions about the documents in the library,
or about files they attach with the 📎 button.
Do not answer factual questions from your own knowledge and do not invent document content.

The user's name: {user_name or "unknown"} (use their first name now and then, naturally).
Documents in the library: {docs}
Files the user attached in this chat: {files}"""
