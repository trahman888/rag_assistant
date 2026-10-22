"""
STEP 2 — Split the text into chunks that keep their source
==========================================================

GOAL
    Cut every page into small overlapping pieces ("chunks"), each one carrying
    its file name and page number so it can be cited later.

RUN
    python step2_chunk.py

WHY CHUNK AT ALL?
    * Precision: one vector for a whole 30-page paper is an average of every
      idea in it, so it matches nothing well. A ~150-word chunk can match the
      exact passage that answers a question.
    * Budget: the LLM only receives the top 5 chunks, not whole files. Local
      models have small context windows, so this matters even more.

THE TRADE-OFF
    Too small -> a chunk loses its context ("this halves memory" — what does?).
    Too big   -> the match gets fuzzy and wastes the LLM's context.
    ~800 characters with ~150 characters of overlap is a sensible start.
    Step 7 measures other sizes instead of guessing.

HOW THIS CHUNKER WORKS
    1. Split the page into sentences.
    2. Add sentences to the current chunk until the next one would make it
       longer than CHUNK_SIZE.
    3. Start the next chunk with the last few sentences of the previous one
       (the OVERLAP), so an idea that sits on a boundary appears complete in
       at least one chunk.
    4. Chunks never cross a page boundary -> every chunk has exactly one page
       to cite.
"""
import re

from extract import extract_file, list_files

CHUNK_SIZE = 800      # max characters per chunk
CHUNK_OVERLAP = 150   # characters repeated at the start of the next chunk

# A sentence ends with . ! or ? followed by a space and a capital letter or digit.
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])")


def split_sentences(text: str) -> list[str]:
    sentences = []
    for paragraph in text.split("\n\n"):
        for s in SENTENCE_END.split(paragraph.strip()):
            s = s.strip()
            # A "sentence" longer than a chunk (tables, code, no punctuation): cut by words.
            while len(s) > CHUNK_SIZE:
                cut = s.rfind(" ", 0, CHUNK_SIZE)
                cut = cut if cut > 0 else CHUNK_SIZE
                sentences.append(s[:cut])
                s = s[cut:].strip()
            if s:
                sentences.append(s)
    return sentences


def chunk_pages(pages: list[dict], size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """pages from step 1 -> list of {"text", "source", "page", "chunk_no"}."""
    chunks = []
    for page in pages:
        texts = []                      # the chunk texts for THIS page
        current: list[str] = []         # sentences in the chunk being built
        for sentence in split_sentences(page["text"]):
            if current and len(" ".join(current + [sentence])) > size:
                texts.append(" ".join(current))
                # carry the last sentence(s) into the next chunk (the overlap)
                tail = []
                while current and len(" ".join([current[-1]] + tail)) <= overlap:
                    tail.insert(0, current.pop())
                current = tail
            current.append(sentence)
        if current:
            texts.append(" ".join(current))

        # every chunk inherits the page's source metadata
        for text in texts:
            chunks.append({"text": text, "source": page["source"], "page": page["page"],
                           "chunk_no": len(chunks)})
    return chunks


# ---------------------------------------------------------------------------
# Run this file directly to see the chunks
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for path in list_files():
        chunks = chunk_pages(extract_file(path))
        lengths = [len(c["text"]) for c in chunks]
        print("=" * 70)
        print(f"{path.name}: {len(chunks)} chunks "
              f"(min {min(lengths)}, avg {sum(lengths) // len(lengths)}, max {max(lengths)} chars)")
        for c in chunks[:2]:  # show the first two so you can judge them
            where = f"page {c['page']}" if c["page"] else "whole file"
            print(f"\n  --- chunk {c['chunk_no']} | {c['source']} | {where}")
            print(f"  {c['text']}")
    print("\nAsk yourself: could someone understand each chunk on its own?")