"""
One shared knowledge base (admin-managed) + temporary chat attachments.

SHARED KNOWLEDGE BASE  — only the admin can change it
    Permanent:  Supabase Storage bucket "documents", folder "shared/"
                (local mode: user_data/shared/uploads/)
    Searchable: Chroma collection "shared_kb_<model>" on the server's disk,
                one collection per embedding model (see embedder.py)

    The server's disk is wiped when Streamlit Cloud restarts, so the first
    visitor after a restart (admin or not) triggers ensure_synced(): it
    downloads shared/ and rebuilds the index once for the whole app.
    Everyone after that shares the same ready index. The same thing happens
    automatically after switching to a new embedding model.

HOW SEARCH WORKS (search())
    1. Several queries: the user's question plus the 2–3 search phrases the
       LLM rewrote it into (router.py). Each query is searched separately.
    2. Hybrid retrieval, for every query:
         * meaning search  — vectors from the embedding model (Chroma)
         * keyword search  — BM25 over the same chunks: exact terms such as
                             "RFQ", "FP4", "Wan2.2" or a person's name
    3. Reciprocal Rank Fusion merges all those ranked lists into one: a chunk
       that ranks well in several lists rises to the top. RRF only uses ranks,
       so meaning scores and keyword scores never have to be compared directly.
    4. Figures: "Figure 3" by number is looked up directly; "show me the …
       figure/chart/diagram" makes sure the best-matching figure is included.

CHAT ATTACHMENTS  — any user, this chat only, never saved
    Files attached in the chat box are read, chunked and embedded in memory
    and kept in that user's session (st.session_state). They come first in
    search results, and disappear on logout, "New chat", or when the tab closes.

USES FROM vector_store.py
    DB_DIR, file_id, extract_file, chunk_pages
"""
import math
import re
import tempfile
import threading
from collections import Counter
from pathlib import Path

import chromadb
import numpy as np

from embedder import collection_suffix, embed_passages, embed_query
from vector_store import DB_DIR, chunk_pages, extract_file, file_id

BUCKET = "documents"
SHARED_PREFIX = "shared"
SUPPORTED = {".pdf", ".md", ".txt"}
SHARED_FOLDER = Path(__file__).parent / "user_data" / "shared" / "uploads"


# =========================================================================== names
def safe_name(name: str) -> str:
    name = Path(name).name
    stem, suffix = Path(name).stem, Path(name).suffix.lower()
    stem = re.sub(r"[^A-Za-z0-9._() -]+", "_", stem).strip(" .") or "document"
    return f"{stem[:100]}{suffix}"


# =========================================================================== shared vector db
_client = None
_client_lock = threading.Lock()


def shared_collection():
    """Open the vector database once per server. The lock stops two visitors who
    arrive at the same moment from opening it twice at once (which Chroma rejects).
    The collection name includes the embedding model, so vectors from different
    models are never mixed."""
    global _client
    with _client_lock:
        if _client is None:
            _client = chromadb.PersistentClient(path=str(DB_DIR))
        return _client.get_or_create_collection(
            f"shared_kb_{collection_suffix()}", metadata={"hnsw:space": "cosine"}, embedding_function=None)


def list_documents(collection) -> list[dict]:
    docs = {}
    for m in collection.get(include=["metadatas"])["metadatas"]:
        d = docs.setdefault(m["doc_id"], {"doc_id": m["doc_id"], "source": m["source"], "chunks": 0, "figures": 0})
        d["chunks"] += 1
        if m.get("kind") == "image":
            d["figures"] += 1
    return sorted(docs.values(), key=lambda d: d["source"].lower())


def index_document(path: Path, collection) -> int:
    """Extract -> chunk -> embed (embedder.py) -> store one file. Returns chunks added (0 = already there)."""
    doc_id = file_id(path)
    if collection.get(where={"doc_id": doc_id}, limit=1)["ids"]:
        return 0
    chunks = chunk_pages(extract_file(path))
    if not chunks:
        return 0
    collection.add(
        ids=[f"{doc_id}-{c['chunk_no']}" for c in chunks],
        embeddings=embed_passages([c["text"] for c in chunks]),
        documents=[c["text"] for c in chunks],
        metadatas=[{"doc_id": doc_id, "source": Path(path).name, "page": c["page"] or 0,
                    "path": str(Path(path).resolve())} for c in chunks],
    )
    keyword_index_changed()
    return len(chunks)


# =========================================================================== permanent file storage
class SupabaseFiles:
    """Files in Supabase Storage under 'shared/'. Supabase's security rules let
    every logged-in user READ them, but only the admin can add/replace/delete."""

    def __init__(self, client):
        self.bucket = client.storage.from_(BUCKET)

    def names(self) -> list[str]:
        items = self.bucket.list(SHARED_PREFIX, {"limit": 1000})
        return sorted(i["name"] for i in items if i.get("name") and not i["name"].startswith(".")
                      and Path(i["name"]).suffix.lower() in SUPPORTED)

    def upload(self, name: str, data: bytes) -> None:
        content_type = {"pdf": "application/pdf", "md": "text/markdown", "png": "image/png",
                        "json": "application/json"}.get(name.rsplit(".", 1)[-1].lower(), "text/plain")
        self.bucket.upload(f"{SHARED_PREFIX}/{name}", data, {"content-type": content_type, "upsert": "true"})

    def download(self, name: str) -> bytes:
        return self.bucket.download(f"{SHARED_PREFIX}/{name}")

    def delete(self, name: str) -> None:
        self.bucket.remove([f"{SHARED_PREFIX}/{name}"])

    def list_folder(self, sub: str) -> list[str]:
        """File names inside a sub-folder, e.g. '_images/<doc_id>'."""
        items = self.bucket.list(f"{SHARED_PREFIX}/{sub}", {"limit": 1000})
        return [i["name"] for i in items if i.get("name") and not i["name"].startswith(".")]


class LocalFiles:
    """Local mode: the shared folder on this computer is the permanent store."""

    def __init__(self):
        SHARED_FOLDER.mkdir(parents=True, exist_ok=True)

    def names(self) -> list[str]:
        return sorted(p.name for p in SHARED_FOLDER.iterdir() if p.suffix.lower() in SUPPORTED)

    def upload(self, name: str, data: bytes) -> None:
        target = SHARED_FOLDER / name
        target.parent.mkdir(parents=True, exist_ok=True)      # e.g. _images/<doc_id>/
        target.write_bytes(data)

    def download(self, name: str) -> bytes:
        return (SHARED_FOLDER / name).read_bytes()

    def delete(self, name: str) -> None:
        (SHARED_FOLDER / name).unlink(missing_ok=True)

    def list_folder(self, sub: str) -> list[str]:
        folder = SHARED_FOLDER / sub
        return [p.name for p in folder.iterdir() if p.is_file()] if folder.is_dir() else []


# =========================================================================== admin operations
def add_document(files, collection, name: str, data: bytes, describe_fn=None, progress=None,
                 on_error=None) -> tuple[int, int]:
    """Save + index one document. For PDFs, describe_fn (a vision model) also turns
    the PDF's images into searchable figure chunks. Returns (text_chunks, figures)."""
    from images import add_pdf_images, delete_pdf_images

    SHARED_FOLDER.mkdir(parents=True, exist_ok=True)
    name = safe_name(name)
    path = SHARED_FOLDER / name
    if path.exists() and path.read_bytes() != data:          # new version of an existing file
        old_id = file_id(path)
        collection.delete(where={"doc_id": old_id})
        delete_pdf_images(files, old_id)
        keyword_index_changed()
    files.upload(name, data)
    path.write_bytes(data)
    chunks = index_document(path, collection)
    figures = 0
    if chunks and describe_fn and path.suffix.lower() == ".pdf":
        figures = add_pdf_images(files, collection, path, file_id(path), name, describe_fn, progress, on_error)
        keyword_index_changed()
    return chunks, figures


def delete_document(files, collection, name: str) -> None:
    """Remove a document everywhere: storage, its figures, and all its chunks."""
    from images import delete_pdf_images

    doc_ids = {m["doc_id"] for m in collection.get(where={"source": name}, include=["metadatas"])["metadatas"]}
    files.delete(name)
    path = SHARED_FOLDER / name
    if path.exists():
        doc_ids.add(file_id(path))
        path.unlink()
    for doc_id in doc_ids:
        delete_pdf_images(files, doc_id)
        collection.delete(where={"doc_id": doc_id})
    collection.delete(where={"source": name})
    keyword_index_changed()


# =========================================================================== sync after restart
_sync_lock = threading.Lock()
_synced = False


def ensure_synced(files, collection, force: bool = False) -> dict | None:
    """Rebuild the shared index from permanent storage — once per server process
    (or again when the admin presses Reload). Safe if several users arrive at once."""
    global _synced
    with _sync_lock:
        if _synced and not force:
            return None
        SHARED_FOLDER.mkdir(parents=True, exist_ok=True)
        report = {"downloaded": 0, "indexed": 0, "removed": 0}
        wanted = set(files.names())
        present_ids = set()
        for name in sorted(wanted):
            path = SHARED_FOLDER / name
            if not path.exists():
                path.write_bytes(files.download(name))
                report["downloaded"] += 1
            doc_id = file_id(path)
            present_ids.add(doc_id)
            if index_document(path, collection):
                report["indexed"] += 1
                if path.suffix.lower() == ".pdf":         # bring back its figures (no vision calls)
                    from images import restore_image_chunks
                    restore_image_chunks(files, collection, doc_id, name)
        for d in list_documents(collection):
            if d["doc_id"] not in present_ids:
                collection.delete(where={"doc_id": d["doc_id"]})
                report["removed"] += 1
        for p in SHARED_FOLDER.iterdir():
            if p.is_file() and p.suffix.lower() in SUPPORTED and p.name not in wanted:
                p.unlink()
        keyword_index_changed()
        _synced = True
        return report


# =========================================================================== temporary attachments
def read_attachment(name: str, data: bytes) -> list[dict]:
    """Parse + chunk + embed one attached file entirely in memory/temp. Not saved anywhere."""
    name = safe_name(name)
    if Path(name).suffix.lower() not in SUPPORTED:
        raise ValueError(f"{name}: only PDF, Markdown and text files are supported")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / name
        path.write_bytes(data)
        chunks = chunk_pages(extract_file(path))
    if not chunks:
        raise ValueError(f"{name}: no readable text (scanned PDF?)")
    vectors = np.array(embed_passages([c["text"] for c in chunks]), dtype=np.float32)
    return [{"text": c["text"], "source": name, "page": c["page"], "attached": True, "vector": v}
            for c, v in zip(chunks, vectors)]


# =========================================================================== keyword search (BM25)
_STOPWORDS = set("""a an and are as at be been but by can could did do does for from had has have how i if in
into is it its me my of on or our over show tell than that the their them then there these they this to
under up was we were what when where which who why will with would you your about please give find see""".split())


def tokenize(text: str) -> list[str]:
    """Lower-case words and numbers; keeps terms like 'fp4', 'wan2', 'bf16', 'qwen3'."""
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS and len(w) > 1]


class BM25:
    """Classic keyword ranking: a chunk scores higher for containing the query's
    words, especially RARE words (idf), with diminishing returns for repeats (k1)
    and a correction for long chunks (b)."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.tfs = [Counter(d) for d in docs]
        self.lengths = [len(d) for d in docs]
        self.avg = (sum(self.lengths) / len(docs)) if docs else 1.0
        df = Counter(w for d in docs for w in set(d))
        n = len(docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query: list[str]) -> list[float]:
        words = [w for w in set(query) if w in self.idf]
        out = []
        for tf, length in zip(self.tfs, self.lengths):
            s = 0.0
            for w in words:
                f = tf.get(w, 0)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / self.avg))
            out.append(s)
        return out


_kw_lock = threading.Lock()
_kw_cache = {"version": -1}
_kw_version = 0


def keyword_index_changed() -> None:
    """Call after any add/delete so the keyword index is rebuilt on the next search."""
    global _kw_version
    _kw_version += 1


def _keyword_index(collection):
    """All chunks + a BM25 index over them, built once and reused until the data changes."""
    with _kw_lock:
        if _kw_cache["version"] != (_kw_version, collection.count()):
            got = collection.get(include=["documents", "metadatas"])
            _kw_cache.update(version=(_kw_version, collection.count()), ids=got["ids"], docs=got["documents"],
                             metas=got["metadatas"], bm25=BM25([tokenize(d) for d in got["documents"]]),
                             pos={cid: i for i, cid in enumerate(got["ids"])})
        return _kw_cache


# =========================================================================== hybrid search
RRF_K = 60                 # standard Reciprocal Rank Fusion constant
POOL = 20                  # candidates taken from each ranked list before fusing
ATTACH_WHOLE_LIMIT = 8     # attachments this small (in chunks) are sent to the LLM in full
ATTACH_TOP = 6             # otherwise: this many best-matching attachment chunks
KB_WITH_ATTACH = 2         # knowledge-base chunks allowed alongside attachments

_FIGURE_REF = re.compile(r"\b(fig(?:ure)?s?\.?|tables?)\s*((?:\d+(?:\s*(?:,|and|&|or|-)\s*)?)+)", re.IGNORECASE)
_WANTS_PICTURE = re.compile(r"\b(figures?|figs?|charts?|graphs?|plots?|diagrams?|images?|pictures?|photos?|"
                            r"illustrations?|visuali[sz]ations?|screenshots?)\b", re.IGNORECASE)


def _hit(text: str, meta: dict, score: float) -> dict:
    return {"text": text, "source": meta["source"], "page": meta.get("page") or None, "score": score,
            "attached": False, "image": meta.get("image"), "caption": meta.get("caption", "")}


def hybrid_rank(collection, queries: list[str]) -> list[tuple[str, float]]:
    """Meaning search + keyword search for every query, fused with RRF.
    Returns [(chunk_id, fused_score)] best first."""
    n = collection.count()
    if not n:
        return []
    kw = _keyword_index(collection)
    fused: dict[str, float] = {}

    def add(ranked_ids):
        for rank, cid in enumerate(ranked_ids, start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)

    for q in queries:
        dense = collection.query(query_embeddings=[embed_query(q)], n_results=min(POOL, n))
        add(dense["ids"][0])
        s = kw["bm25"].scores(tokenize(q))
        top = sorted((i for i in range(len(s)) if s[i] > 0), key=lambda i: -s[i])[:POOL]
        add([kw["ids"][i] for i in top])
    return sorted(fused.items(), key=lambda kv: -kv[1])


def named_figures(collection, texts: list[str]) -> list[dict]:
    """Figures/tables named by number ("Figure 3", "figs 1 and 2", "Table 1"), matched on captions."""
    refs = {(("table" if kind.lower().startswith("tab") else "figure"), num)
            for t in texts for kind, nums in _FIGURE_REF.findall(t or "") for num in re.findall(r"\d+", nums)}
    if not refs:
        return []
    got = collection.get(where={"kind": "image"}, include=["documents", "metadatas"])
    out = []
    for text, m in zip(got["documents"], got["metadatas"]):
        cap = (m.get("caption") or "").strip()
        hit = re.match(r"(fig(?:ure)?\.?|table)\s*(\d+)", cap, re.IGNORECASE)
        if hit and (("table" if hit.group(1).lower().startswith("tab") else "figure"), hit.group(2)) in refs:
            out.append(_hit(text, m, 1.0))
    return out


def best_figure(collection, queries: list[str], ranked: list[tuple[str, float]]) -> list[dict]:
    """The best-matching figure: the highest-ranked figure in the fused results,
    or, if none made it there, the closest figure by meaning."""
    kw = _keyword_index(collection)
    for cid, score in ranked:
        i = kw["pos"].get(cid)
        if i is not None and kw["metas"][i].get("kind") == "image":
            return [_hit(kw["docs"][i], kw["metas"][i], score)]
    try:
        r = collection.query(query_embeddings=[embed_query(queries[0])], n_results=1, where={"kind": "image"})
        return [_hit(r["documents"][0][0], r["metadatas"][0][0], 0.0)] if r["ids"][0] else []
    except Exception:
        return []


def search(collection, attachments: list[dict], question: str, k: int = 4,
           queries: list[str] | None = None) -> list[dict]:
    """Find the passages to answer from.

    queries: extra search phrases from query rewriting (router.py). The original
    question is always searched too, so a poor rewrite can't make things worse.

    No attachments -> the k best knowledge-base chunks (hybrid, multi-query).
    Attachments    -> the user's own files come FIRST, like in ChatGPT: a small
                      attachment is included in full, a large one contributes its
                      best-matching chunks, and at most KB_WITH_ATTACH knowledge-
                      base chunks are added, only if they match better.
    """
    all_queries = list(dict.fromkeys([question] + [q for q in (queries or []) if q and q.strip()]))[:4]

    # figures: by number ("Figure 3"), or "show me the … figure/chart/diagram"
    figures = named_figures(collection, all_queries)
    ranked = hybrid_rank(collection, all_queries) if collection.count() else []
    if not figures and _WANTS_PICTURE.search(question or "") and ranked:
        figures = best_figure(collection, all_queries, ranked)
    taken = {f["image"] for f in figures}

    kw = _keyword_index(collection) if ranked else None
    kb = []
    for cid, score in ranked:
        i = kw["pos"].get(cid)
        if i is None or kw["metas"][i].get("image") in taken:
            continue
        kb.append(_hit(kw["docs"][i], kw["metas"][i], score))

    if not attachments:
        return (figures + kb)[:max(k, len(figures))]

    qvecs = np.array([embed_query(q) for q in all_queries], dtype=np.float32)
    scores = (np.stack([a["vector"] for a in attachments]) @ qvecs.T).max(axis=1)   # best match over all queries
    if len(attachments) <= ATTACH_WHOLE_LIMIT:
        chosen = range(len(attachments))                                # whole file(s), in reading order
    else:
        chosen = sorted(np.argsort(-scores)[:ATTACH_TOP])               # best chunks, kept in reading order
    att = [{"text": attachments[i]["text"], "source": attachments[i]["source"], "page": attachments[i]["page"],
            "score": float(scores[i]), "attached": True} for i in chosen]

    # knowledge-base extras only when they match the question better than the attachment does
    best_att = float(scores.max())
    extra = []
    if collection.count():
        r = collection.query(query_embeddings=[qvecs[0].tolist()], n_results=min(KB_WITH_ATTACH, collection.count()))
        extra = [_hit(t, m, 1 - d) for t, m, d in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])
                 if 1 - d > best_att and m.get("image") not in taken]
    return att + figures + extra
