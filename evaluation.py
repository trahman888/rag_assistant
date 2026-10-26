"""
Measuring answer quality, with numbers instead of impressions.

THE TEST SET (saved per user in _eval.json)
    Each test case is a question plus how to recognise the right passage:
        question       "How does the method keep accuracy at 4 bits?"
        expected_text  a short phrase the correct passage contains, e.g.
                       "protects the salient weight channels"   (best)
        document/page  optional: the file (and page) the answer is in
        in_documents   False for questions the documents DON'T answer; the
                       right behaviour is then "I couldn't find this".
    Cases come from three places: written by hand, generated from your own
    chunks by the LLM (generate_cases), or copied from answers rated 👎.

WHAT IS MEASURED (evaluate)
    Retrieval, for each search mode:
        keyword  BM25 only
        vector   embeddings only
        hybrid   both, fused with Reciprocal Rank Fusion (what the app uses)
        rewrite  hybrid + the LLM's rewritten queries (exactly what the app does)
      Hit rate@K  share of questions where a correct passage is in the top K
      MRR         Mean Reciprocal Rank: 1 if the correct passage is first,
                  1/2 if second, 1/3 if third… 0 if missing. Rewards ranking
                  the right passage high, not just somewhere in the top K.
    Answers (optional, one LLM call per question, using the "rewrite" mode):
      Cited correctly     the answer cites at least one correct passage
      Correct "not found" for questions the documents don't answer
"""
import json
import re
import time
from datetime import datetime, timezone

MODES = {
    "keyword": "Keyword only (BM25)",
    "vector": "Vector only (embeddings)",
    "hybrid": "Hybrid (vector + BM25)",
    "rewrite": "Hybrid + query rewriting",
}
EVAL_FILE = "_eval.json"
RUNS_FILE = "_eval_runs.json"
MAX_RUNS = 30
NOT_FOUND_HINTS = ("couldn't find", "could not find", "not in the documents", "no information", "don't contain",
                   "do not contain", "not mentioned")


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def clean_case(c: dict) -> dict | None:
    """Tidy one row from the editor; None if it's empty or can't be checked."""
    q = str(c.get("question") or "").strip()
    if not q:
        return None
    page = c.get("page")
    try:
        page = int(page) if page not in (None, "") and page == page else None    # NaN != NaN
    except (TypeError, ValueError):
        page = None
    in_docs = c.get("in_documents")
    case = {"question": q, "expected_text": str(c.get("expected_text") or "").strip(),
            "document": str(c.get("document") or "").strip(), "page": page,
            "in_documents": True if in_docs is None or in_docs != in_docs else bool(in_docs),
            "origin": str(c.get("origin") or "manual")}
    return case


def checkable(case: dict) -> bool:
    return (not case["in_documents"]) or bool(case["expected_text"] or case["document"])


def is_relevant(hit: dict, case: dict) -> bool:
    """Is this retrieved passage a correct one for the test case?"""
    if case["document"] and hit.get("source") != case["document"]:
        return False
    if case["expected_text"]:
        return _norm(case["expected_text"]) in _norm(hit.get("text", ""))
    if case["page"]:
        return hit.get("page") == case["page"]
    return bool(case["document"])


def first_relevant_rank(hits: list[dict], case: dict) -> int | None:
    for rank, h in enumerate(hits, start=1):
        if is_relevant(h, case):
            return rank
    return None


def _with_retry(fn, *args, tries: int = 4):
    """Groq's free tier limits tokens per minute; a burst of test questions can hit it."""
    for attempt in range(tries):
        try:
            return fn(*args)
        except Exception as e:
            if "ratelimit" not in type(e).__name__.lower() and "429" not in str(e) or attempt == tries - 1:
                raise
            time.sleep(10 * (attempt + 1))


def evaluate(library, cases: list[dict], modes: list[str], k: int, plan_fn=None, answer_fn=None,
             progress=None) -> dict:
    """Run every test case through every search mode (and optionally the full answer).
    plan_fn(question) -> rewritten queries;  answer_fn(question, passages) -> answer text."""
    from answer import find_citations

    cases = [c for c in cases if checkable(c)]
    findable = [c for c in cases if c["in_documents"]]
    steps = len(findable) * len(modes) + (len(cases) if answer_fn else 0)
    done = 0

    def tick(label):
        nonlocal done
        done += 1
        if progress:
            progress(done / max(steps, 1), label)

    rewrites: dict[str, list[str]] = {}
    report = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "k": k,
              "n_cases": len(findable), "modes": {}, "answers": None}

    for mode in modes:
        rows, hits_n, rr_sum = [], 0, 0.0
        for c in findable:
            queries = None
            if mode == "rewrite":
                if c["question"] not in rewrites:
                    try:
                        rewrites[c["question"]] = _with_retry(plan_fn, c["question"]) if plan_fn else []
                    except Exception:
                        rewrites[c["question"]] = []
                queries = rewrites[c["question"]]
            hits = library.search(c["question"], k=k, queries=queries,
                                  mode="hybrid" if mode == "rewrite" else mode)[:k]
            rank = first_relevant_rank(hits, c)
            hits_n += rank is not None
            rr_sum += 1 / rank if rank else 0.0
            rows.append({"question": c["question"], "rank": rank, "top": [h["source"] + (f" p{h['page']}" if h.get("page") else "")
                                                                          for h in hits[:3]]})
            tick(f"{MODES[mode]}: {c['question'][:60]}")
        n = max(len(findable), 1)
        report["modes"][mode] = {"hit_rate": hits_n / n, "mrr": rr_sum / n, "rows": rows}

    if answer_fn:
        rows = []
        cited_ok = cited_n = nf_ok = nf_n = errors = 0
        for c in cases:
            queries = rewrites.get(c["question"])
            if queries is None and plan_fn:
                try:
                    queries = _with_retry(plan_fn, c["question"])
                except Exception:
                    queries = []
            hits = library.search(c["question"], k=k, queries=queries)
            try:
                text = _with_retry(answer_fn, c["question"], hits) if hits else "I couldn't find this in your documents."
            except Exception as e:
                errors += 1
                rows.append({"question": c["question"], "verdict": f"⚠️ error: {str(e)[:80]}", "answer": ""})
                tick("answer")
                continue
            said_not_found = any(h in text.lower() for h in NOT_FOUND_HINTS)
            if c["in_documents"]:
                cited, _ = find_citations(text, len(hits))
                good = any(is_relevant(hits[n - 1], c) for n in cited)
                cited_n += 1
                cited_ok += good
                verdict = "✅ cited a correct passage" if good else ("❌ said not found" if said_not_found
                                                                  else "❌ no correct passage cited")
            else:
                nf_n += 1
                nf_ok += said_not_found
                verdict = "✅ said not found" if said_not_found else "❌ answered anyway"
            rows.append({"question": c["question"], "verdict": verdict, "answer": text[:300]})
            tick("answer")
        report["answers"] = {"cited_rate": cited_ok / cited_n if cited_n else None, "n_cited": cited_n,
                             "not_found_rate": nf_ok / nf_n if nf_n else None, "n_not_found": nf_n,
                             "errors": errors, "rows": rows}
    return report


# --------------------------------------------------------------------------- generating test questions
GENERATE_PROMPT = """You write test questions for a search system.

Read the passage and write ONE question that a real user might ask and that this passage answers.
- Use DIFFERENT words from the passage where you can (paraphrase), the way a person would ask.
- The question must make sense on its own, without seeing the passage.
- Then copy "evidence": 4–12 words copied EXACTLY, character for character, from the passage,
  that contain the answer.

Reply with JSON only: {"question": "...", "evidence": "..."}"""


def generate_cases(library, llm_complete, n: int = 5, seed: int | None = None) -> list[dict]:
    """Ask the LLM to write test questions from random chunks of the user's own documents.
    Only keeps a question if its evidence really appears in the chunk."""
    out = []
    for chunk in library.chunk_samples(n * 2, seed):
        if len(out) >= n:
            break
        try:
            reply = _with_retry(llm_complete, [{"role": "system", "content": GENERATE_PROMPT},
                                               {"role": "user", "content": f"Passage:\n{chunk['text']}"}])
            m = re.search(r"\{.*\}", re.sub(r"<think>.*?</think>", "", reply, flags=re.S), re.S)
            data = json.loads(m.group(0)) if m else {}
        except Exception:
            continue
        q, ev = str(data.get("question", "")).strip(), str(data.get("evidence", "")).strip()
        if q and ev and _norm(ev) and _norm(ev) in _norm(chunk["text"]):
            out.append({"question": q, "expected_text": ev, "document": chunk["source"], "page": chunk["page"],
                        "in_documents": True, "origin": "generated"})
    return out
