"""
One shared knowledge base (admin-managed) + temporary chat attachments.

SHARED KNOWLEDGE BASE  — only the admin can change it
    Permanent:  Supabase Storage bucket "documents", folder "shared/"
                (local mode: user_data/shared/uploads/)
    Searchable: Chroma collection "shared_kb" on the server's disk

    The server's disk is wiped when Streamlit Cloud restarts, so the first
    visitor after a restart (admin or not) triggers ensure_synced(): it
    downloads shared/ and rebuilds the index once for the whole app.
    Everyone after that shares the same ready index.

CHAT ATTACHMENTS  — any user, this chat only, never saved
    Files attached in the chat box are read, chunked and embedded in memory
    and kept in that user's session (st.session_state). They are searched
    together with the shared knowledge base, and disappear on logout,
    "New chat", or when the browser tab closes. Nothing is written to
    Supabase or to the vector database.

USES FROM vector_store.py
    DB_DIR, embed, file_id, index_file, extract_file, chunk_pages
"""
import hashlib
import re
import tempfile
import threading
from pathlib import Path

import chromadb
import numpy as np

from vector_store import DB_DIR, chunk_pages, embed, extract_file, file_id, index_file

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
    arrive at the same moment from opening it twice at once (which Chroma rejects)."""
    global _client
    with _client_lock:
        if _client is None:
            _client = chromadb.PersistentClient(path=str(DB_DIR))
        return _client.get_or_create_collection(
            "shared_kb", metadata={"hnsw:space": "cosine"}, embedding_function=None)


def list_documents(collection) -> list[dict]:
    docs = {}
    for m in collection.get(include=["metadatas"])["metadatas"]:
        d = docs.setdefault(m["doc_id"], {"doc_id": m["doc_id"], "source": m["source"], "chunks": 0, "figures": 0})
        d["chunks"] += 1
        if m.get("kind") == "image":
            d["figures"] += 1
    return sorted(docs.values(), key=lambda d: d["source"].lower())


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
def add_document(files, collection, name: str, data: bytes, describe_fn=None, progress=None) -> tuple[int, int]:
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
    files.upload(name, data)
    path.write_bytes(data)
    chunks = index_file(path, collection)
    figures = 0
    if chunks and describe_fn and path.suffix.lower() == ".pdf":
        figures = add_pdf_images(files, collection, path, file_id(path), name, describe_fn, progress)
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
            if index_file(path, collection):
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
    vectors = np.array(embed([c["text"] for c in chunks]), dtype=np.float32)
    return [{"text": c["text"], "source": name, "page": c["page"], "attached": True, "vector": v}
            for c, v in zip(chunks, vectors)]


# =========================================================================== search
ATTACH_WHOLE_LIMIT = 8     # attachments this small (in chunks) are sent to the LLM in full
ATTACH_TOP = 6             # otherwise: this many best-matching attachment chunks
KB_WITH_ATTACH = 2         # knowledge-base chunks allowed alongside attachments


def search(collection, attachments: list[dict], question: str, k: int = 4) -> list[dict]:
    """Find the passages to answer from.

    No attachments  -> the k best knowledge-base chunks.
    Attachments     -> the user's own files come FIRST, like in ChatGPT:
                       * a small attachment is included in full, so questions
                         like "summarise my PDF" work even though they match no
                         particular passage;
                       * a large one contributes its best-matching chunks;
                       * up to KB_WITH_ATTACH knowledge-base chunks are added
                         only if they match the question better than the
                         attachment does, so the knowledge base can't crowd out
                         the user's file.
    """
    q = np.array(embed([question])[0], dtype=np.float32)

    def kb_hits(n_wanted):
        n = collection.count()
        if not n or n_wanted <= 0:
            return []
        r = collection.query(query_embeddings=[q.tolist()], n_results=min(n_wanted, n))
        return [{"text": t, "source": m["source"], "page": m.get("page") or None, "score": 1 - d, "attached": False,
                 "image": m.get("image")}
                for t, m, d in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])]

    if not attachments:
        return kb_hits(k)

    scores = np.stack([a["vector"] for a in attachments]) @ q           # cosine (vectors are normalised)
    if len(attachments) <= ATTACH_WHOLE_LIMIT:
        chosen = range(len(attachments))                                # whole file(s), in reading order
    else:
        chosen = sorted(np.argsort(-scores)[:ATTACH_TOP])               # best chunks, kept in reading order
    att = [{"text": attachments[i]["text"], "source": attachments[i]["source"], "page": attachments[i]["page"],
            "score": float(scores[i]), "attached": True} for i in chosen]

    best_att = max(h["score"] for h in att)
    extra = [h for h in kb_hits(KB_WITH_ATTACH) if h["score"] > best_att]
    return att + extra