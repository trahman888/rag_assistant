"""
STEP 4 — Answer questions with an open-source LLM, with citations
==================================================================

GOAL
    Take the chunks from step 3, give them to a local open-source LLM, and get
    back an answer where every claim cites its source: [1], [2] ...

SETUP (once)
    1. Install Ollama from https://ollama.com  (runs open-source LLMs locally)
    2. Download a model:      ollama pull llama3.1:8b
       Other good choices:    qwen2.5:7b, mistral:7b, gemma2:9b
       Low on memory (8 GB)?  llama3.2:3b  (faster, but weaker at citing)
    3. Ollama runs a local server at http://localhost:11434 — we call it over HTTP.
       No API key, nothing leaves your computer.

RUN
    python step4_answer.py "What are the three main approaches to reducing LLM inference costs?"
    python step4_answer.py        (interactive: keeps asking)

HOW IT WORKS
    1. search(question)  -> top 5 chunks             (step 3)
    2. Number the chunks as sources:
           [1] paper.pdf, page 2
           <chunk text>
           [2] notes.md
           <chunk text>
    3. A system prompt sets the rules: use ONLY these sources, cite [n] after
       every claim, and say "I couldn't find this in your documents." when the
       sources don't contain the answer. That last rule stops the model from
       filling gaps with things it "knows".
    4. Send it to Ollama's /api/chat endpoint and read the reply.
    5. Find every [n] in the reply. Valid numbers map back to a file and page;
       a number with no matching source is a made-up citation, and we flag it.

TIPS FOR SMALL LOCAL MODELS
    * temperature 0.1 keeps answers close to the sources.
    * num_ctx: Ollama's default context window is small. 5 chunks + prompt is
      ~1,500 tokens, so 8192 leaves plenty of room. If answers ignore sources,
      a too-small context window is a common cause.
    * 7–9B models cite reasonably well; 3B models often forget citations.
"""
import os
import re
import sys
import json

import requests

from vector_store import search

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:4b")
TOP_K = 5
NOT_FOUND = "I couldn't find this in your documents."

SYSTEM_PROMPT = f"""You answer questions using ONLY the numbered sources given to you.

Rules:
1. Use only information from the sources. Do not use outside knowledge.
2. After every sentence that states a fact, write the number of the source that supports it, like [1] or [2][3].
3. If the sources do not contain the answer, reply exactly: {NOT_FOUND}
4. If the sources answer only part of the question, answer that part and say what is missing.
5. Keep the answer short and clear."""


def format_sources(chunks: list[dict]) -> str:
    blocks = []
    for n, c in enumerate(chunks, start=1):
        page = f", page {c['page']}" if c["page"] else ""
        blocks.append(f"[{n}] {c['source']}{page}\n{c['text']}")
    return "\n\n".join(blocks)


def call_llm(system: str, user: str) -> str:
    """Stream the answer from Ollama so you see progress, and never wait forever."""
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": LLM_MODEL,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": True,          # receive the answer piece by piece
            "think": False,          # no long hidden reasoning (Qwen3)
            "options": {
                "temperature": 0.1,
                "num_ctx": 4096,     # enough for 5 chunks + question
                "num_predict": 400,  # stop after ~300 words
            },
        },
        stream=True,
        timeout=(10, 120),           # 10 s to connect, 120 s max silence between pieces
    )
    if response.status_code == 404:
        raise SystemExit(f"Model '{LLM_MODEL}' not found. Run:  ollama pull {LLM_MODEL}")
    response.raise_for_status()

    parts = []
    for line in response.iter_lines():
        if not line:
            continue
        piece = json.loads(line)
        text = piece.get("message", {}).get("content", "")
        print(text, end="", flush=True)   # show it as it's written
        parts.append(text)
        if piece.get("done"):
            break
    print()
    return "".join(parts).strip()


def find_citations(answer: str, n_sources: int) -> tuple[list[int], list[int]]:
    """Return (valid, invalid) citation numbers found in the answer: [1], [2][3], [1, 4]."""
    numbers = [int(n) for group in re.findall(r"\[([\d,\s]+)\]", answer)
               for n in re.findall(r"\d+", group)]
    numbers = list(dict.fromkeys(numbers))                      # unique, keep order
    return ([n for n in numbers if 1 <= n <= n_sources],
            [n for n in numbers if not 1 <= n <= n_sources])


def ask(question: str, k: int = TOP_K, search_fn=search) -> dict:
    """The full RAG pipeline: retrieve -> prompt -> generate -> check citations."""
    chunks = search_fn(question, k)
    if not chunks:
        return {"question": question, "answer": NOT_FOUND, "sources": [], "cited": [], "invalid": []}
    user_message = f"<sources>\n{format_sources(chunks)}\n</sources>\n\nQuestion: {question}"
    answer = call_llm(SYSTEM_PROMPT, user_message)
    cited, invalid = find_citations(answer, len(chunks))
    return {"question": question, "answer": answer, "sources": chunks, "cited": cited, "invalid": invalid}


def print_result(result: dict) -> None:
    print(f"\n{result['answer']}\n")
    if result["invalid"]:
        print(f"WARNING: cited sources that don't exist: {result['invalid']}")
    print("Sources:")
    for n, c in enumerate(result["sources"], start=1):
        page = f", page {c['page']}" if c["page"] else ""
        mark = "*" if n in result["cited"] else " "
        print(f" {mark}[{n}] {c['source']}{page}")
    print("(* = cited in the answer)")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        requests.get(OLLAMA_URL, timeout=3)
    except requests.ConnectionError:
        raise SystemExit("Ollama isn't running. Start the Ollama app (or run: ollama serve).")

    if len(sys.argv) > 1:
        print_result(ask(" ".join(sys.argv[1:])))
    else:
        while question := input("\nQuestion (Enter to quit): ").strip():
            print_result(ask(question))