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


def shared_collection():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(DB_DIR))
    return _client.get_or_create_collection(
        "shared_kb", metadata={"hnsw:space": "cosine"}, embedding_function=None)


def list_documents(collection) -> list[dict]:
    docs = {}
    for m in collection.get(include=["metadatas"])["metadatas"]:
        d = docs.setdefault(m["doc_id"], {"doc_id": m["doc_id"], "source": m["source"], "chunks": 0})
        d["chunks"] += 1
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
        content_type = {"pdf": "application/pdf", "md": "text/markdown"}.get(name.rsplit(".", 1)[-1], "text/plain")
        self.bucket.upload(f"{SHARED_PREFIX}/{name}", data, {"content-type": content_type, "upsert": "true"})

    def download(self, name: str) -> bytes:
        return self.bucket.download(f"{SHARED_PREFIX}/{name}")

    def delete(self, name: str) -> None:
        self.bucket.remove([f"{SHARED_PREFIX}/{name}"])


class LocalFiles:
    """Local mode: the shared folder on this computer is the permanent store."""

    def __init__(self):
        SHARED_FOLDER.mkdir(parents=True, exist_ok=True)

    def names(self) -> list[str]:
        return sorted(p.name for p in SHARED_FOLDER.iterdir() if p.suffix.lower() in SUPPORTED)

    def upload(self, name: str, data: bytes) -> None:
        (SHARED_FOLDER / name).write_bytes(data)

    def download(self, name: str) -> bytes:
        return (SHARED_FOLDER / name).read_bytes()

    def delete(self, name: str) -> None:
        (SHARED_FOLDER / name).unlink(missing_ok=True)


# =========================================================================== admin operations
def add_document(files, collection, name: str, data: bytes) -> int:
    SHARED_FOLDER.mkdir(parents=True, exist_ok=True)
    name = safe_name(name)
    path = SHARED_FOLDER / name
    if path.exists() and path.read_bytes() != data:          # new version of an existing file
        collection.delete(where={"doc_id": file_id(path)})
    files.upload(name, data)
    path.write_bytes(data)
    return index_file(path, collection)


def delete_document(files, collection, name: str) -> None:
    files.delete(name)
    path = SHARED_FOLDER / name
    if path.exists():
        collection.delete(where={"doc_id": file_id(path)})
        path.unlink()
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
            present_ids.add(file_id(path))
            if index_file(path, collection):
                report["indexed"] += 1
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
def search(collection, attachments: list[dict], question: str, k: int = 4) -> list[dict]:
    """Search the shared knowledge base AND this chat's attachments, best matches first."""
    q = np.array(embed([question])[0], dtype=np.float32)
    hits = []

    n = collection.count()
    if n:
        r = collection.query(query_embeddings=[q.tolist()], n_results=min(k, n))
        hits += [{"text": t, "source": m["source"], "page": m.get("page") or None, "score": 1 - d, "attached": False}
                 for t, m, d in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])]

    if attachments:
        scores = np.stack([a["vector"] for a in attachments]) @ q       # cosine (vectors are normalised)
        for i in np.argsort(-scores)[:k]:
            a = attachments[i]
            hits.append({"text": a["text"], "source": a["source"], "page": a["page"],
                         "score": float(scores[i]), "attached": True})

    return sorted(hits, key=lambda h: -h["score"])[:k]
