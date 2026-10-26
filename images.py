"""
Images inside PDFs: extract -> describe with a vision model -> make them searchable.

WHY
    A PDF's text layer skips its pictures, so charts, diagrams and photos were
    invisible to search. Here each meaningful image gets a written description
    from a vision AI, and that description is stored as a normal chunk. Search,
    routing, privacy masking and citations then work for figures automatically,
    and the Sources box can show the picture itself.

PIPELINE (when a user uploads a PDF, runs once per PDF)
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
    Tables drawn only with thin lines (most research-paper tables) are skipped:
    their contents are already searchable as text. Boxed grid tables are kept.
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
LABEL_GAP_PT = 14       # small text this close to a figure (axis labels, values) is part of it
LABEL_FONT_RATIO = 0.8  # "small" = font under 80% of the page's body text (labels are small; paragraphs and captions aren't)
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
def extract_images(pdf_path: Path, max_images: int = MAX_IMAGES) -> list[dict]:
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
    if Path(pdf_path).suffix.lower() in (".docx", ".pptx"):     # Word / PowerPoint: their embedded pictures
        from extractors import office_images
        return office_images(pdf_path, max_images)

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
                if len(found) >= max_images:
                    return found
    return found


def figure_regions(page) -> list[tuple]:
    """(rectangle, caption) for each figure on one page, top to bottom."""
    import pymupdf

    page_rect = page.rect
    boxes = [pymupdf.Rect(i["bbox"]) & page_rect for i in page.get_image_info()]   # placed pictures
    try:
        drawings = page.get_drawings()
        for cluster in page.cluster_drawings(drawings=drawings):                 # drawn charts, diagrams
            # (a perfectly flat line has zero height, which counts as "empty", so pad it before testing)
            inside = [pymupdf.Rect(d["rect"]) for d in drawings
                      if (pymupdf.Rect(d["rect"]) + (-1, -1, 1, 1)).intersects(cluster)]
            if inside and all(min(r.width, r.height) < 2.5 for r in inside):
                continue          # only thin rules: a text table or separator, already searchable as text
            boxes.append(cluster & page_rect)
    except Exception:
        pass
    boxes = [b for b in boxes if not b.is_empty and b.width > 2 and b.height > 2]
    regions = _merge(boxes, MERGE_GAP_PT)

    lines = _text_lines(page)
    body_size = _body_font_size(lines)
    results = []
    for r in regions:
        if r.get_area() > FULL_PAGE * page_rect.get_area():
            continue                                   # a whole-page picture (scan / slide), not a figure
        if min(r.width, r.height) < MIN_SIDE_PT or r.get_area() < MIN_AREA_PT:
            continue                                   # logo, icon, horizontal rule
        near = pymupdf.Rect(r.x0 - LABEL_GAP_PT, r.y0 - LABEL_GAP_PT, r.x1 + LABEL_GAP_PT, r.y1 + LABEL_GAP_PT)
        for rect, text, size in lines:                 # pull in the figure's own small labels only,
            if size < LABEL_FONT_RATIO * body_size and near.intersects(rect):   # never paragraphs or captions
                r |= rect
        caption, cap_rect = _find_caption(r, lines)
        crop = (r + (-4, -4, 4, 4)) & page_rect        # a little white margin
        if cap_rect is not None:
            if cap_rect.y0 > r.y0 + 0.6 * r.height:    # caption below: never include any of it
                crop.y1 = min(crop.y1, cap_rect.y0 - 1)
            elif cap_rect.y1 <= r.y0 + 2:              # caption above (tables): reach up to it, so the
                crop.y0 = cap_rect.y1 + 1              # header row printed above the table's lines is kept
        results.append((crop, caption))
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
    """(rectangle, text, font size) for every text line on the page."""
    import pymupdf

    out = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            spans = [sp for sp in line["spans"] if sp["text"].strip()]
            text = "".join(sp["text"] for sp in line["spans"]).strip()
            if text and spans:
                size = max(sp["size"] for sp in spans)
                out.append((pymupdf.Rect(line["bbox"]), text, size))
    return out


def _body_font_size(lines) -> float:
    """The page's normal text size: the size used by the most characters."""
    counts = {}
    for _, text, size in lines:
        counts[round(size, 1)] = counts.get(round(size, 1), 0) + len(text)
    return max(counts, key=counts.get) if counts else 10.0


def _find_caption(region, lines) -> tuple:
    """The 'Figure 3: …' / 'Table 1: …' line belonging to the region: just below it
    (even if the drawn area slightly overlaps it), or, for tables, just above it.
    Returns (caption text, caption rectangle or None)."""
    best, best_rect, best_dist = "", None, 60.0
    for rect, text, _size in lines:
        if not _CAPTION.match(text):
            continue
        if min(rect.x1, region.x1) - max(rect.x0, region.x0) <= 0:
            continue                                   # not under/over the figure
        if rect.y0 >= region.y1 - 25 and rect.y0 > region.y0 + 0.6 * region.height:
            dist = max(0.0, rect.y0 - region.y1)       # below, or overlapping the bottom edge
        elif rect.y1 <= region.y0 + 2:
            dist = region.y0 - rect.y1 + 5             # above (table captions); slightly less preferred
        else:
            continue
        if dist < best_dist:
            best, best_rect, best_dist = text, rect, dist
    return best[:200], best_rect


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
                   describe_fn, progress=None, on_error=None, max_images: int = MAX_IMAGES) -> int:
    """Run the whole pipeline for one PDF. Returns how many figures were added.
    describe_fn(png_bytes, prompt) -> str ; progress(done, total) is optional."""
    images = extract_images(pdf_path, max_images)
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
            if on_error:
                on_error(f"page {im['page']}: {e}")
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
    from embedder import embed_passages

    unit = "slide" if source.lower().endswith(".pptx") else "page"
    texts = [f"[Figure on {unit} {r['page']}]" + (f" {r['caption'].rstrip('. ')}." if r.get("caption") else "")
             + f" {r['description']}" for r in records]
    collection.upsert(
        ids=[f"{doc_id}-img-{r['file']}" for r in records],
        embeddings=embed_passages(texts),
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
    """Remove a document's stored figures (called when the user deletes the PDF)."""
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
