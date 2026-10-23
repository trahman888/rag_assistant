"""
Per-user documents and per-user vector database.

WHY FILES DISAPPEARED BEFORE
    Streamlit Cloud gives the app a temporary disk. When the app restarts or
    goes to sleep, everything written there (uploads/, vector_db/) is wiped.
    So the files must be kept somewhere permanent: Supabase Storage.

HOW IT WORKS NOW
    Permanent:  Supabase Storage bucket "documents"
                    <user_id>/report.pdf
                    <user_id>/notes.md
    Temporary:  user_data/<user_id>/uploads/   local copies, used for reading
                vector_db/  collection "user_<id>"   that user's chunks only

    On login, sync_user() downloads the user's files from Supabase into their
    local folder and indexes anything not yet in their collection. After a
    restart the temporary copies are gone, so the first login re-downloads
    and re-indexes (a few seconds per document); later logins are instant.

    In local mode (no Supabase), user_data/<user_id>/uploads/ IS the
    permanent store.

ISOLATION
    Every user has their own folder, their own storage prefix and their own
    Chroma collection, so search() can only ever return that user's chunks.
    In Supabase, row-level security policies (see SETUP_SUPABASE.md) stop one
    user's token from reading another user's files.

USES FROM vector_store.py
    DB_DIR, embed, file_id, index_file   (keep those names in that file)
"""
import hashlib
import re
from pathlib import Path

import chromadb

from vector_store import DB_DIR, embed, file_id, index_file

BUCKET = "documents"
SUPPORTED = {".pdf", ".md", ".txt"}
DATA_ROOT = Path(__file__).parent / "user_data"


# --------------------------------------------------------------------------- names
def safe_name(name: str) -> str:
    """Storage keys can't contain every character; keep names simple and safe."""
    name = Path(name).name
    stem, suffix = Path(name).stem, Path(name).suffix.lower()
    stem = re.sub(r"[^A-Za-z0-9._() -]+", "_", stem).strip(" .") or "document"
    return f"{stem[:100]}{suffix}"


def _uid_slug(user_id: str) -> str:
    return hashlib.sha1(user_id.encode()).hexdigest()[:16]


def user_folder(user: dict) -> Path:
    folder = DATA_ROOT / _uid_slug(user["id"]) / "uploads"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


# --------------------------------------------------------------------------- vector db
_client = None


def user_collection(user: dict):
    """Each user gets a separate Chroma collection inside the same database."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(DB_DIR))
    return _client.get_or_create_collection(
        f"user_{_uid_slug(user['id'])}", metadata={"hnsw:space": "cosine"}, embedding_function=None)


def list_documents(collection) -> list[dict]:
    docs = {}
    for m in collection.get(include=["metadatas"])["metadatas"]:
        d = docs.setdefault(m["doc_id"], {"doc_id": m["doc_id"], "source": m["source"], "chunks": 0})
        d["chunks"] += 1
    return sorted(docs.values(), key=lambda d: d["source"].lower())


def search(collection, question: str, k: int = 4) -> list[dict]:
    n = collection.count()
    if n == 0:
        return []
    r = collection.query(query_embeddings=embed([question]), n_results=min(k, n))
    return [{"text": t, "source": m["source"], "page": m.get("page") or None, "score": 1 - d}
            for t, m, d in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])]


# --------------------------------------------------------------------------- permanent file storage
class SupabaseFiles:
    """The user's files in Supabase Storage, under the prefix '<user_id>/'."""

    def __init__(self, client, user: dict):
        self.bucket = client.storage.from_(BUCKET)
        self.prefix = user["id"]

    def names(self) -> list[str]:
        items = self.bucket.list(self.prefix, {"limit": 1000})
        return sorted(i["name"] for i in items
                      if i.get("name") and not i["name"].startswith(".") and Path(i["name"]).suffix.lower() in SUPPORTED)

    def upload(self, name: str, data: bytes) -> None:
        content_type = {"pdf": "application/pdf", "md": "text/markdown"}.get(name.rsplit(".", 1)[-1], "text/plain")
        self.bucket.upload(f"{self.prefix}/{name}", data, {"content-type": content_type, "upsert": "true"})

    def download(self, name: str) -> bytes:
        return self.bucket.download(f"{self.prefix}/{name}")

    def delete(self, name: str) -> None:
        self.bucket.remove([f"{self.prefix}/{name}"])


class LocalFiles:
    """Local mode: the user's folder on this computer is the permanent store."""

    def __init__(self, folder: Path):
        self.folder = folder

    def names(self) -> list[str]:
        return sorted(p.name for p in self.folder.iterdir() if p.suffix.lower() in SUPPORTED)

    def upload(self, name: str, data: bytes) -> None:
        (self.folder / name).write_bytes(data)

    def download(self, name: str) -> bytes:
        return (self.folder / name).read_bytes()

    def delete(self, name: str) -> None:
        (self.folder / name).unlink(missing_ok=True)


# --------------------------------------------------------------------------- operations
def add_document(files, collection, folder: Path, name: str, data: bytes) -> int:
    """Save permanently, keep a local copy, index it. Returns chunks added (0 = already there)."""
    name = safe_name(name)
    old = folder / name
    if old.exists() and old.read_bytes() != data:       # same name, new content: replace old version
        _remove_from_index(collection, file_id(old))
    files.upload(name, data)
    old.write_bytes(data)
    return index_file(old, collection)


def delete_document(files, collection, folder: Path, name: str) -> None:
    files.delete(name)
    path = folder / name
    if path.exists():
        _remove_from_index(collection, file_id(path))
        path.unlink()
    collection.delete(where={"source": name})          # in case the local copy was already gone


def _remove_from_index(collection, doc_id: str) -> None:
    collection.delete(where={"doc_id": doc_id})


def sync_user(files, collection, folder: Path) -> dict:
    """Make local copies and the vector index match the permanent store."""
    report = {"downloaded": 0, "indexed": 0, "removed": 0}
    wanted = set(files.names())

    for name in wanted:                                  # 1. bring back missing local copies
        path = folder / name
        if not path.exists():
            path.write_bytes(files.download(name))
            report["downloaded"] += 1

    present_ids = set()
    for name in sorted(wanted):                          # 2. index anything not yet indexed
        path = folder / name
        present_ids.add(file_id(path))
        if index_file(path, collection):
            report["indexed"] += 1

    for d in list_documents(collection):                 # 3. drop chunks of files that are gone
        if d["doc_id"] not in present_ids:
            _remove_from_index(collection, d["doc_id"])
            report["removed"] += 1

    for p in folder.iterdir():                           # 4. drop stale local copies
        if p.is_file() and p.suffix.lower() in SUPPORTED and p.name not in wanted:
            p.unlink()
    return report
