"""
STEP 1 — Extract and inspect text from your documents
======================================================

GOAL
    Read every file in the documents/ folder and turn it into clean text that
    remembers WHERE it came from (file name + page number). That "where" is
    what later becomes the citation: "[2] report.pdf, page 4".

RUN
    1. Put a few files (PDF, .md or .txt) into a folder called  documents/
    2. python extract.py

WHAT HAPPENS
    For each file we produce a list of "pages". A page is a small dict:
        {"source": "report.pdf", "page": 3, "text": "..."}
    - PDF  -> one entry per PDF page (pypdf reads the text layer).
    - .md / .txt -> one entry for the whole file (page = None).

    PDF text comes out exactly as it was laid out on paper: a hard line break
    at the end of every printed line and words split with hyphens
    ("quanti-\\nzation"). clean_text() repairs that, because both search and
    the LLM work better on normal sentences.

WHAT TO LOOK FOR IN THE OUTPUT
    * A page with 0 characters = a scanned image. It has no text layer and
      needs OCR first (e.g. the free tool `ocrmypdf`). We can't search it.
    * Tables usually come out as a jumble of cells — note it, it affects answers.
    * Headings often run straight into the paragraph after them. PDFs don't
      store "this is a heading", only positioned letters.
"""
import re
from pathlib import Path

from pypdf import PdfReader

DOCUMENTS_DIR = Path("./documents")
SUPPORTED = {".pdf", ".md", ".txt"}


def clean_text(text: str) -> str:
    """Fix the usual extraction damage while keeping paragraph breaks."""
    text = text.replace("­", "")                  # invisible soft hyphens
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)       # "quanti-\nzation" -> "quantization"
    text = re.sub(r"[ \t]+", " ", text)                # many spaces -> one space
    text = re.sub(r" *\n *", "\n", text)               # spaces around line breaks
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)       # single line break = wrapped line -> join
    text = re.sub(r"\n{3,}", "\n\n", text)             # keep at most one blank line
    return text.strip()


def extract_file(path: Path) -> list[dict]:
    """Return a list of {"source", "page", "text"} for one file."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(path)
        return [
            {"source": path.name, "page": number, "text": clean_text(page.extract_text() or "")}
            for number, page in enumerate(reader.pages, start=1)
        ]
    if path.suffix.lower() in (".md", ".txt"):
        text = path.read_text(encoding="utf-8", errors="replace")
        return [{"source": path.name, "page": None, "text": clean_text(text)}]
    raise ValueError(f"Unsupported file type: {path.suffix}")


def list_files(folder: Path = DOCUMENTS_DIR) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in SUPPORTED)


# ---------------------------------------------------------------------------
# Run this file directly to inspect what was extracted
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    files = list_files() if DOCUMENTS_DIR.exists() else []
    if not files:
        raise SystemExit("Put some .pdf / .md / .txt files in the documents/ folder first.")

    for path in files:
        pages = extract_file(path)
        total = sum(len(p["text"]) for p in pages)
        print("=" * 70)
        print(f"{path.name}: {len(pages)} page(s), {total:,} characters")
        for p in pages:
            where = f"page {p['page']}" if p["page"] else "whole file"
            warning = "   <-- EMPTY: scanned image? needs OCR" if len(p["text"]) < 50 else ""
            print(f"  {where:<11} {len(p['text']):>6,} chars{warning}")
            print(f"     \"{p['text'][:120]}...\"")