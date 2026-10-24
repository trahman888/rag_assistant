"""
Images inside PDFs: extract -> describe with a vision model -> make them searchable.

WHY
    A PDF's text layer skips its pictures, so charts, diagrams and photos were
    invisible to search. Here each meaningful image gets a written description
    from a vision AI, and that description is stored as a normal chunk. Search,
    routing, privacy masking and citations then work for figures automatically,
    and the Sources box can show the picture itself.

PIPELINE (admin upload, runs once per PDF)
    1. extract_images()   PyMuPDF pulls every embedded image out of the PDF.
                          Tiny ones (logos, icons, lines) and repeats are skipped,
                          at most MAX_IMAGES per document.
    2. describe_fn(...)   a vision model writes a description of each image
                          (supplied by chatapp.py: Groq online, Ollama locally).
                          Decorative images are dropped.
    3. stored             the PNG and an index.json with all descriptions go to
                          storage under _images/<doc_id>/, next to the PDF.
    4. indexed            each description becomes a chunk tagged with its page
                          and image path, under the PDF's doc_id, so deleting the
                          PDF deletes its figures too.

AFTER A SERVER RESTART
    restore_image_chunks() reads index.json back, so figures become searchable
    again WITHOUT calling the vision model a second time.

LIMITS
    Only embedded pictures are found. Charts drawn as vector graphics (common in
    papers made with LaTeX) are not images inside the PDF, so they're skipped.
    Images are sent to the vision model as-is: text masking can't hide what's
    printed inside a picture.
"""
import hashlib
import io
import json
from pathlib import Path

from PIL import Image

MIN_SIDE = 120          # px: smaller than this is an icon, bullet or line
MIN_AREA = 40_000       # px²: e.g. 200 × 200
MAX_IMAGES = 20         # per document, to protect the vision model's free quota
MAX_SEND_SIDE = 1024    # px: images are shrunk to this before being described
IMAGES_DIR = "_images"  # storage sub-folder, next to the PDFs

DESCRIBE_PROMPT = """This image is from page {page} of the document "{source}".
Text on the same page starts: "{context}"

Describe the image so it can be found by search and used to answer questions:
- what kind of image it is (chart, diagram, table, photo, screenshot, ...)
- everything it shows, including ALL visible text, labels, axis names, numbers and units
- the main trend, comparison or takeaway
Use 2–6 plain sentences. Don't guess beyond what is visible.
If the image is purely decorative (logo, border, background, icon), reply with exactly: DECORATIVE"""


# =========================================================================== 1. extract
def extract_images(pdf_path: Path) -> list[dict]:
    """Embedded images worth describing: [{"page", "png", "width", "height", "context"}]."""
    import pymupdf

    found, seen = [], set()
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            page_text = " ".join(page.get_text().split())[:300]
            for info in page.get_images(full=True):
                xref = info[0]
                try:
                    raw = doc.extract_image(xref)
                except Exception:
                    continue
                w, h = raw.get("width", 0), raw.get("height", 0)
                if min(w, h) < MIN_SIDE or w * h < MIN_AREA:
                    continue                                   # icon / logo / rule line
                digest = hashlib.sha1(raw["image"]).hexdigest()
                if digest in seen:
                    continue                                   # same picture repeated (e.g. on every page)
                seen.add(digest)
                try:
                    png = _to_png(raw["image"])
                except Exception:
                    continue                                   # format Pillow can't read
                found.append({"page": page.number + 1, "png": png, "width": w, "height": h,
                              "context": page_text})
                if len(found) >= MAX_IMAGES:
                    return found
    return found


def _to_png(data: bytes) -> bytes:
    """Normalise to RGB PNG no larger than MAX_SEND_SIDE (smaller uploads, any input format)."""
    img = Image.open(io.BytesIO(data))
    if img.mode not in ("RGB", "L"):
        background = Image.new("RGB", img.size, "white")
        img = img.convert("RGBA")
        background.paste(img, mask=img.split()[-1])
        img = background
    img.thumbnail((MAX_SEND_SIDE, MAX_SEND_SIDE))
    out = io.BytesIO()
    img.convert("RGB").save(out, format="PNG", optimize=True)
    return out.getvalue()


# =========================================================================== 2–4. describe, store, index
def add_pdf_images(files, collection, pdf_path: Path, doc_id: str, source: str,
                   describe_fn, progress=None) -> int:
    """Run the whole pipeline for one PDF. Returns how many figures were added.
    describe_fn(png_bytes, prompt) -> str ; progress(done, total) is optional."""
    images = extract_images(pdf_path)
    records = []
    for i, im in enumerate(images, start=1):
        if progress:
            progress(i, len(images))
        prompt = DESCRIBE_PROMPT.format(page=im["page"], source=source, context=im["context"] or "(no text)")
        try:
            text = (describe_fn(im["png"], prompt) or "").strip()
        except Exception as e:                      # one failed image shouldn't stop the upload
            print(f"[images] couldn't describe image {i} of {source}: {e}")
            continue
        if not text or "DECORATIVE" in text.upper()[:40]:
            continue
        name = f"p{im['page']}_{i}.png"
        files.upload(f"{IMAGES_DIR}/{doc_id}/{name}", im["png"])
        records.append({"file": name, "page": im["page"], "description": text})

    if records:
        files.upload(f"{IMAGES_DIR}/{doc_id}/index.json", json.dumps(records, indent=2).encode())
        _index(collection, doc_id, source, records)
    return len(records)


def _index(collection, doc_id: str, source: str, records: list[dict]) -> None:
    from vector_store import embed

    texts = [f"[Figure on page {r['page']}] {r['description']}" for r in records]
    collection.upsert(
        ids=[f"{doc_id}-img-{r['file']}" for r in records],
        embeddings=embed(texts),
        documents=texts,
        metadatas=[{"doc_id": doc_id, "source": source, "page": r["page"], "kind": "image",
                    "image": f"{IMAGES_DIR}/{doc_id}/{r['file']}", "path": ""} for r in records],
    )


def restore_image_chunks(files, collection, doc_id: str, source: str) -> int:
    """After a restart: re-add figure chunks from the saved index.json (no vision calls)."""
    try:
        records = json.loads(files.download(f"{IMAGES_DIR}/{doc_id}/index.json"))
    except Exception:
        return 0                                    # this PDF has no figures (or none were saved)
    _index(collection, doc_id, source, records)
    return len(records)


def delete_pdf_images(files, doc_id: str) -> None:
    """Remove a document's stored figures (called when the admin deletes the PDF)."""
    try:
        names = files.list_folder(f"{IMAGES_DIR}/{doc_id}")
    except Exception:
        return
    for name in names:
        try:
            files.delete(f"{IMAGES_DIR}/{doc_id}/{name}")
        except Exception:
            pass


def load_image(files, image_path: str, cache_dir: Path) -> bytes | None:
    """Image bytes for the Sources box: local copy if we have one, else download and keep it."""
    local = cache_dir / image_path
    if local.exists():
        return local.read_bytes()
    try:
        data = files.download(image_path)
    except Exception:
        return None
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(data)
    return data
