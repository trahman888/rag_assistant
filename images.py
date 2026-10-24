"""
Images inside PDFs: extract -> describe with a vision model -> make them searchable.

WHY
    A PDF's text layer skips its pictures, so charts, diagrams and photos were
    invisible to search. Here each meaningful image gets a written description
    from a vision AI, and that description is stored as a normal chunk. Search,
    routing, privacy masking and citations then work for figures automatically,
    and the Sources box can show the picture itself.

PIPELINE (admin upload, runs once per PDF)
    1. extract_images()   PyMuPDF finds each figure's AREA on the page (placed
                          pictures + drawn charts/diagrams/tables), merges pieces
                          that belong together, adds nearby axis labels, finds
                          its "Figure N: …" caption, and renders exactly that
                          area. Logos, rules, whole-page scans and repeats are
                          skipped; at most MAX_IMAGES per document.
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
    Figure areas are found with layout rules, so unusual layouts can still be
    cut imperfectly: two figures placed almost touching may merge into one, and
    a short line of body text right next to a figure may be included with it.
    Tune MERGE_GAP_PT / LABEL_GAP_PT above if your documents need it.
    Scanned pages (one picture per page) are skipped here; they need OCR.
    Images are sent to the vision model as-is: text masking can't hide what's
    printed inside a picture.
"""
import hashlib
import io
import json
import re
from pathlib import Path

from PIL import Image

MIN_SIDE_PT = 50        # points (1/72 inch): narrower or shorter regions are logos, icons, rules
MIN_AREA_PT = 9_000     # points²: about 95 × 95
FULL_PAGE = 0.80        # a region covering more than this share of the page is a whole page, not a figure
MERGE_GAP_PT = 10       # pieces closer than this are one figure (tiles, panels, bars + axes)
LABEL_GAP_PT = 14       # short text this close to a figure (axis labels, values) is part of it
RENDER_DPI = 150
MAX_IMAGES = 20         # per document, to protect the vision model's free quota
MAX_SEND_SIDE = 1024    # px: images are shrunk to this before being described
IMAGES_DIR = "_images"  # storage sub-folder, next to the PDFs

_CAPTION = re.compile(r"^\s*(figure|fig\.?|table|chart|diagram|image|plate|exhibit)\s*\d+", re.IGNORECASE)

DESCRIBE_PROMPT = """This image is from page {page} of the document "{source}".
Its caption in the document: "{caption}"
Text on the same page starts: "{context}"

Describe the image so it can be found by search and used to answer questions:
- what kind of image it is (chart, diagram, table, photo, screenshot, ...)
- everything it shows, including ALL visible text, labels, axis names, numbers and units
- the main trend, comparison or takeaway
Use 2–6 plain sentences. Don't guess beyond what is visible.
If the image is purely decorative (logo, border, background, icon), reply with exactly: DECORATIVE"""


# =========================================================================== 1. find figures on the page
def extract_images(pdf_path: Path) -> list[dict]:
    """Figures as they LOOK on the page: [{"page", "png", "width", "height", "context", "caption"}].

    Why not just take the images stored inside the PDF? Because what's stored
    often doesn't match what you see:
      * one figure may be stored as several pieces (tiles, layers, panels);
      * a scanned or exported page may be one big picture of the whole page;
      * charts drawn with lines and shapes (most research papers) aren't
        stored as images at all.
    So instead we find WHERE figures are on each page (embedded pictures plus
    drawn graphics), merge pieces that touch into one figure, pull in their
    axis labels, skip whole-page captures, and render exactly that area."""
    import pymupdf

    found, seen = [], set()
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            page_text = " ".join(page.get_text().split())[:300]
            for region, caption in figure_regions(page):
                pix = page.get_pixmap(clip=region, dpi=RENDER_DPI)
                png = _to_png(pix.tobytes("png"))
                digest = hashlib.sha1(png).hexdigest()
                if digest in seen:
                    continue                                   # same figure repeated (e.g. on every page)
                seen.add(digest)
                found.append({"page": page.number + 1, "png": png, "width": pix.width, "height": pix.height,
                              "context": page_text, "caption": caption})
                if len(found) >= MAX_IMAGES:
                    return found
    return found


def figure_regions(page) -> list[tuple]:
    """(rectangle, caption) for each figure on one page, top to bottom."""
    import pymupdf

    page_rect = page.rect
    boxes = [pymupdf.Rect(i["bbox"]) & page_rect for i in page.get_image_info()]   # placed pictures
    try:
        boxes += [r & page_rect for r in page.cluster_drawings()]                # drawn charts, diagrams, tables
    except Exception:
        pass
    boxes = [b for b in boxes if not b.is_empty and b.width > 2 and b.height > 2]
    regions = _merge(boxes, MERGE_GAP_PT)

    lines = _text_lines(page)
    results = []
    for r in regions:
        if r.get_area() > FULL_PAGE * page_rect.get_area():
            continue                                   # a whole-page picture (scan / slide), not a figure
        if min(r.width, r.height) < MIN_SIDE_PT or r.get_area() < MIN_AREA_PT:
            continue                                   # logo, icon, horizontal rule
        caption = ""
        for rect, text in lines:                       # pull in short labels around the figure
            if _CAPTION.match(text):
                continue
            near = pymupdf.Rect(r.x0 - LABEL_GAP_PT, r.y0 - LABEL_GAP_PT, r.x1 + LABEL_GAP_PT, r.y1 + LABEL_GAP_PT)
            if len(text) <= 40 and near.intersects(rect):
                r |= rect
        caption = _find_caption(r, lines)
        results.append((r + (-4, -4, 4, 4)) & page_rect)        # a little white margin
        results[-1] = (results[-1], caption)
    return sorted(results, key=lambda rc: (rc[0].y0, rc[0].x0))


def _merge(boxes: list, gap: float) -> list:
    """Repeatedly join rectangles that overlap or are within `gap` points of each other."""
    import pymupdf

    boxes = [pymupdf.Rect(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        out = []
        while boxes:
            cur = boxes.pop()
            i = 0
            while i < len(boxes):
                grown = pymupdf.Rect(cur.x0 - gap, cur.y0 - gap, cur.x1 + gap, cur.y1 + gap)
                if grown.intersects(boxes[i]):
                    cur |= boxes.pop(i)
                    changed = True
                else:
                    i += 1
            out.append(cur)
        boxes = out
    return boxes


def _text_lines(page) -> list[tuple]:
    import pymupdf

    out = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = "".join(span["text"] for span in line["spans"]).strip()
            if text:
                out.append((pymupdf.Rect(line["bbox"]), text))
    return out


def _find_caption(region, lines) -> str:
    """The 'Figure 3: …' / 'Table 1: …' line just below (or, for tables, just above) the region."""
    best, best_dist = "", 60.0
    for rect, text in lines:
        if not _CAPTION.match(text):
            continue
        overlaps = min(rect.x1, region.x1) - max(rect.x0, region.x0) > 0
        if not overlaps:
            continue
        dist = rect.y0 - region.y1 if rect.y0 >= region.y1 - 2 else region.y0 - rect.y1
        if 0 <= dist < best_dist:
            best, best_dist = text, dist
    return best[:200]


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
        prompt = DESCRIBE_PROMPT.format(page=im["page"], source=source, context=im["context"] or "(no text)",
                                        caption=im.get("caption") or "none")
        try:
            text = (describe_fn(im["png"], prompt) or "").strip()
        except Exception as e:                      # one failed image shouldn't stop the upload
            print(f"[images] couldn't describe image {i} of {source}: {e}")
            continue
        if not text or "DECORATIVE" in text.upper()[:40]:
            continue
        name = f"p{im['page']}_{i}.png"
        files.upload(f"{IMAGES_DIR}/{doc_id}/{name}", im["png"])
        records.append({"file": name, "page": im["page"], "caption": im.get("caption", ""), "description": text})

    if records:
        files.upload(f"{IMAGES_DIR}/{doc_id}/index.json", json.dumps(records, indent=2).encode())
        _index(collection, doc_id, source, records)
    return len(records)


def _index(collection, doc_id: str, source: str, records: list[dict]) -> None:
    from vector_store import embed

    texts = [f"[Figure on page {r['page']}]" + (f" {r['caption'].rstrip('. ')}." if r.get("caption") else "")
             + f" {r['description']}" for r in records]
    collection.upsert(
        ids=[f"{doc_id}-img-{r['file']}" for r in records],
        embeddings=embed(texts),
        documents=texts,
        metadatas=[{"doc_id": doc_id, "source": source, "page": r["page"], "kind": "image",
                    "image": f"{IMAGES_DIR}/{doc_id}/{r['file']}", "caption": r.get("caption", ""),
                    "path": ""} for r in records],
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
