"""
Private document libraries: every user has their own, and nobody else can see or search it.

ONE LIBRARY PER USER (class Library)
    Permanent:  Supabase Storage bucket "documents", folder "<user_id>/"
                    <user_id>/report.pdf
                    <user_id>/_images/<doc_id>/p2_1.png   (figures + index.json)
                    <user_id>/_settings.json              (that user's Top K)
                    <user_id>/_chat.json                  (their current conversation)
                    <user_id>/_last_chat.json             (the one before, for "Continue last conversation")
                    <user_id>/_feedback.json              (their 👍/👎 on answers)
                    <user_id>/_eval.json, _eval_runs.json (their test questions + results)
                Supabase's security rules only let a logged-in user touch the
                folder named after their own id (see SETUP_SUPABASE.md).
                Local mode: user_data/<user>/docs/ on this computer.
    Searchable: that user's OWN Chroma collection "u_<user>_<model>".
                search() only ever queries this one collection, so another
                user's documents can never appear in your answers.

    The server's disk is wiped when Streamlit Cloud restarts, so the first
    time a user opens the app after a restart, Library.sync() downloads their
    files and rebuilds their index (figures come back from their saved
    descriptions, without calling the vision model again).

LIMITS (to keep the database from getting crowded)
    MAX_DOCUMENTS per user, MAX_FILE_MB per file, MAX_FIGURES_PER_PDF.

HOW SEARCH WORKS (Library.search())
    1. Several queries: the question plus the 2–3 search phrases the LLM
       rewrote it into (router.py).
    2. Hybrid retrieval for every query: meaning search (vectors) + keyword
       search (BM25), merged with Reciprocal Rank Fusion.
    3. Figures: "Figure 3" by number is looked up directly; "show me the …
       figure/chart/diagram" makes sure the best-matching figure is included.

CHAT ATTACHMENTS (📎) — this chat only, never saved
    Read, chunked and embedded in memory; they come first in search results.

USES FROM vector_store.py
    DB_DIR, file_id, chunk_pages   (+ extractors.extract_any for Word and PowerPoint)
"""
import hashlib
import json
import math
import re
import tempfile
import threading
from collections import Counter
from pathlib import Path

import chromadb
import numpy as np

from embedder import collection_suffix, embed_passages, embed_query
from extractors import extract_any
from vector_store import DB_DIR, chunk_pages, file_id

BUCKET = "documents"
SUPPORTED = {".pdf", ".md", ".txt", ".docx", ".pptx"}
WITH_FIGURES = {".pdf", ".docx", ".pptx"}     # file types whose pictures are described by the vision model
DATA_ROOT = Path(__file__).parent / "user_data"

# ---- limits per user --------------------------------------------------------
MAX_DOCUMENTS = 20          # documents in one user's library
MAX_FILE_MB = 20            # size of one uploaded file
MAX_FIGURES_PER_PDF = 10    # figures described per PDF (each one uses the vision model's free quota)

DEFAULT_SETTINGS = {"top_k": 4}
SETTINGS_FILE = "_settings.json"
CHAT_FILE = "_chat.json"
LAST_CHAT_FILE = "_last_chat.json"
FEEDBACK_FILE = "_feedback.json"
MAX_FEEDBACK = 500
MAX_SAVED_MESSAGES = 60     # older messages of a very long chat are not kept


class LimitError(ValueError):
    """An upload that would go over the user's limits."""


# =========================================================================== small helpers
def safe_name(name: str) -> str:
    name = Path(name).name
    stem, suffix = Path(name).stem, Path(name).suffix.lower()
    stem = re.sub(r"[^A-Za-z0-9._() -]+", "_", stem).strip(" .") or "document"
    return f"{stem[:100]}{suffix}"


def _slug(user_id: str) -> str:
    return hashlib.sha1(user_id.encode()).hexdigest()[:16]


_client = None
_client_lock = threading.Lock()


def _chroma():
    """One database client per server; the lock stops two visitors opening it at once."""
    global _client
    with _client_lock:
        if _client is None:
            _client = chromadb.PersistentClient(path=str(DB_DIR))
        return _client


# =========================================================================== permanent file storage
class SupabaseFiles:
    """One user's files in Supabase Storage, all under the folder '<user_id>/'."""

    def __init__(self, client, prefix: str):
        self.bucket = client.storage.from_(BUCKET)
        self.prefix = prefix

    def names(self) -> list[str]:
        items = self.bucket.list(self.prefix, {"limit": 1000})
        return sorted(i["name"] for i in items if i.get("name") and not i["name"].startswith(".")
                      and Path(i["name"]).suffix.lower() in SUPPORTED)

    def upload(self, name: str, data: bytes) -> None:
        content_type = {"pdf": "application/pdf", "md": "text/markdown", "png": "image/png",
                        "json": "application/json",
                        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation"}.get(name.rsplit(".", 1)[-1].lower(), "text/plain")
        self.bucket.upload(f"{self.prefix}/{name}", data, {"content-type": content_type, "upsert": "true"})

    def download(self, name: str) -> bytes:
        return self.bucket.download(f"{self.prefix}/{name}")

    def delete(self, name: str) -> None:
        self.bucket.remove([f"{self.prefix}/{name}"])

    def list_folder(self, sub: str) -> list[str]:
        items = self.bucket.list(f"{self.prefix}/{sub}", {"limit": 1000})
        return [i["name"] for i in items if i.get("name") and not i["name"].startswith(".")]


class LocalFiles:
    """Local mode: the user's folder on this computer is the permanent store."""

    def __init__(self, folder: Path):
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)

    def names(self) -> list[str]:
        return sorted(p.name for p in self.folder.iterdir() if p.suffix.lower() in SUPPORTED)

    def upload(self, name: str, data: bytes) -> None:
        target = self.folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    def download(self, name: str) -> bytes:
        return (self.folder / name).read_bytes()

    def delete(self, name: str) -> None:
        (self.folder / name).unlink(missing_ok=True)

    def list_folder(self, sub: str) -> list[str]:
        f = self.folder / sub
        return [p.name for p in f.iterdir() if p.is_file()] if f.is_dir() else []


# =========================================================================== keyword search (BM25)
_STOPWORDS = set("""a an and are as at be been but by can could did do does for from had has have how i if in
into is it its me my of on or our over show tell than that the their them then there these they this to
under up was we were what when where which who why will with would you your about please give find see""".split())


def tokenize(text: str) -> list[str]:
    """Lower-case words and numbers; keeps terms like 'fp4', 'wan2', 'bf16', 'qwen3'."""
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS and len(w) > 1]


class BM25:
    """Keyword ranking: rewards a chunk for containing the query's words, especially
    RARE words (idf), with diminishing returns for repeats (k1) and a length correction (b)."""

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


# Caches shared by everyone on this server, keyed by the user's collection name.
_kw_cache: dict[str, dict] = {}       # keyword index per library
_kw_version: dict[str, int] = {}      # bumped on every add/delete
_synced: set[str] = set()             # libraries already rebuilt since this server started
_settings_cache: dict[str, dict] = {}
_lock = threading.RLock()


# =========================================================================== search constants
RRF_K = 60                 # standard Reciprocal Rank Fusion constant
POOL = 20                  # candidates taken from each ranked list before fusing
ATTACH_WHOLE_LIMIT = 8     # attachments this small (in chunks) are sent to the LLM in full
ATTACH_TOP = 6             # otherwise: this many best-matching attachment chunks
LIB_WITH_ATTACH = 2        # library chunks allowed alongside attachments

_FIGURE_REF = re.compile(r"\b(fig(?:ure)?s?\.?|tables?)\s*((?:\d+(?:\s*(?:,|and|&|or|-)\s*)?)+)", re.IGNORECASE)
_WANTS_PICTURE = re.compile(r"\b(figures?|figs?|charts?|graphs?|plots?|diagrams?|images?|pictures?|photos?|"
                            r"illustrations?|visuali[sz]ations?|screenshots?)\b", re.IGNORECASE)


def _hit(text: str, meta: dict, score: float) -> dict:
    return {"text": text, "source": meta["source"], "page": meta.get("page") or None, "score": score,
            "attached": False, "image": meta.get("image"), "caption": meta.get("caption", "")}


# =========================================================================== the library
class Library:
    """Everything belonging to ONE user: their files, their vector collection,
    their keyword index, their figures and their settings."""

    def __init__(self, user: dict, supabase_client=None):
        slug = _slug(user["id"])
        self.folder = DATA_ROOT / slug / "docs"                       # local copies (cache on Streamlit Cloud)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.files = (SupabaseFiles(supabase_client, user["id"]) if supabase_client is not None
                      else LocalFiles(self.folder))
        self.collection = _chroma().get_or_create_collection(
            f"u_{slug}_{collection_suffix()}"[:63], metadata={"hnsw:space": "cosine"}, embedding_function=None)
        self.key = self.collection.name

    # ------------------------------------------------------------------ contents
    def documents(self) -> list[dict]:
        docs = {}
        for m in self.collection.get(include=["metadatas"])["metadatas"]:
            d = docs.setdefault(m["doc_id"], {"doc_id": m["doc_id"], "source": m["source"], "chunks": 0, "figures": 0})
            d["chunks"] += 1
            if m.get("kind") == "image":
                d["figures"] += 1
        return sorted(docs.values(), key=lambda d: d["source"].lower())

    def _changed(self) -> None:
        with _lock:
            _kw_version[self.key] = _kw_version.get(self.key, 0) + 1

    def _index_text(self, path: Path) -> int:
        doc_id = file_id(path)
        if self.collection.get(where={"doc_id": doc_id}, limit=1)["ids"]:
            return 0
        chunks = chunk_pages(extract_any(path))
        if not chunks:
            return 0
        self.collection.add(
            ids=[f"{doc_id}-{c['chunk_no']}" for c in chunks],
            embeddings=embed_passages([c["text"] for c in chunks]),
            documents=[c["text"] for c in chunks],
            metadatas=[{"doc_id": doc_id, "source": path.name, "page": c["page"] or 0, "path": str(path.resolve())}
                       for c in chunks],
        )
        self._changed()
        return len(chunks)

    # ------------------------------------------------------------------ add / delete
    def check_limits(self, name: str, size_bytes: int) -> None:
        """Raise LimitError if this upload would go over the user's limits."""
        if size_bytes > MAX_FILE_MB * 1024 * 1024:
            raise LimitError(f"{name} is {size_bytes / 1024 / 1024:.1f} MB; the limit is {MAX_FILE_MB} MB per file.")
        current = {d["source"] for d in self.documents()}
        if safe_name(name) not in current and len(current) >= MAX_DOCUMENTS:
            raise LimitError(f"Your library is full ({MAX_DOCUMENTS} documents). Delete one to add {name}.")

    def add(self, name: str, data: bytes, describe_fn=None, progress=None, on_error=None) -> tuple[int, int]:
        """Save + index one document. For PDFs, describe_fn (a vision model) also turns
        up to MAX_FIGURES_PER_PDF figures into searchable chunks. Returns (text_chunks, figures)."""
        from images import add_pdf_images, delete_pdf_images

        self.check_limits(name, len(data))
        name = safe_name(name)
        path = self.folder / name
        if path.exists() and path.read_bytes() != data:              # new version of an existing file
            old_id = file_id(path)
            self.collection.delete(where={"doc_id": old_id})
            delete_pdf_images(self.files, old_id)
            self._changed()
        self.files.upload(name, data)
        path.write_bytes(data)
        chunks = self._index_text(path)
        figures = 0
        if chunks and describe_fn and path.suffix.lower() in WITH_FIGURES:
            figures = add_pdf_images(self.files, self.collection, path, file_id(path), name, describe_fn,
                                     progress, on_error, max_images=MAX_FIGURES_PER_PDF)
            self._changed()
        return chunks, figures

    def delete(self, name: str) -> None:
        """Remove a document everywhere: storage, its figures, and all its chunks."""
        from images import delete_pdf_images

        metas = self.collection.get(where={"source": name}, include=["metadatas"])["metadatas"]
        doc_ids = {m["doc_id"] for m in metas}
        self.files.delete(name)
        path = self.folder / name
        if path.exists():
            doc_ids.add(file_id(path))
            path.unlink()
        for doc_id in doc_ids:
            delete_pdf_images(self.files, doc_id)
            self.collection.delete(where={"doc_id": doc_id})
        self.collection.delete(where={"source": name})
        self._changed()

    # ------------------------------------------------------------------ restore after restart
    def sync(self, force: bool = False) -> dict | None:
        """Make this user's index match their stored files — once per server start
        (or when they press Reload). Figures come back without new vision calls."""
        from images import restore_image_chunks

        with _lock:
            if self.key in _synced and not force:
                return None
            report = {"downloaded": 0, "indexed": 0, "removed": 0}
            wanted = set(self.files.names())
            present = set()
            for name in sorted(wanted):
                path = self.folder / name
                if not path.exists():
                    path.write_bytes(self.files.download(name))
                    report["downloaded"] += 1
                doc_id = file_id(path)
                present.add(doc_id)
                if self._index_text(path):
                    report["indexed"] += 1
                    if path.suffix.lower() in WITH_FIGURES:
                        restore_image_chunks(self.files, self.collection, doc_id, name)
            for d in self.documents():
                if d["doc_id"] not in present:
                    self.collection.delete(where={"doc_id": d["doc_id"]})
                    report["removed"] += 1
            for p in self.folder.iterdir():
                if p.is_file() and p.suffix.lower() in SUPPORTED and p.name not in wanted:
                    p.unlink()
            self._changed()
            _synced.add(self.key)
            return report

    # ------------------------------------------------------------------ settings (per user)
    def settings(self) -> dict:
        with _lock:
            if self.key not in _settings_cache:
                try:
                    saved = json.loads(self.files.download(SETTINGS_FILE))
                except Exception:
                    saved = {}
                _settings_cache[self.key] = {**DEFAULT_SETTINGS,
                                             **{k: v for k, v in saved.items() if k in DEFAULT_SETTINGS}}
            return dict(_settings_cache[self.key])

    def save_setting(self, key: str, value) -> None:
        current = self.settings()
        current[key] = value
        self.files.upload(SETTINGS_FILE, json.dumps(current, indent=2).encode())
        with _lock:
            _settings_cache[self.key] = current

    # ------------------------------------------------------------------ conversation (survives a page refresh)
    def load_chat(self) -> list[dict]:
        try:
            messages = json.loads(self.files.download(CHAT_FILE))
            return messages if isinstance(messages, list) else []
        except Exception:
            return []                                   # no saved chat yet

    def save_chat(self, messages: list[dict]) -> None:
        """Save the conversation in the user's private folder. Passages that came from
        📎 attachments are saved WITHOUT their text, because attachments are never stored."""
        keep = []
        for m in messages[-MAX_SAVED_MESSAGES:]:
            m = dict(m)
            if m.get("sources"):
                m["sources"] = [{k: v for k, v in c.items() if k != "vector"} | ({"text": "(attached file, not saved)"}
                                if c.get("attached") else {}) for c in m["sources"]]
            keep.append(m)
        self.files.upload(CHAT_FILE, json.dumps(keep, default=float).encode())

    def clear_chat(self) -> None:
        try:
            self.files.delete(CHAT_FILE)
        except Exception:
            pass

    def archive_chat(self) -> None:
        """Start fresh but keep the current conversation as the "last" one,
        so the user can choose to continue it. An empty chat doesn't replace the last one."""
        current = self.load_chat()
        if current:
            self.files.upload(LAST_CHAT_FILE, json.dumps(current, default=float).encode())
            self.clear_chat()

    def load_last_chat(self) -> list[dict]:
        try:
            messages = json.loads(self.files.download(LAST_CHAT_FILE))
            return messages if isinstance(messages, list) else []
        except Exception:
            return []

    def resume_last_chat(self) -> list[dict]:
        """Make the last conversation the current one again."""
        messages = self.load_last_chat()
        if messages:
            self.files.upload(CHAT_FILE, json.dumps(messages, default=float).encode())
            try:
                self.files.delete(LAST_CHAT_FILE)
            except Exception:
                pass
        return messages

    # ------------------------------------------------------------------ small JSON files (feedback, evaluation)
    def load_json(self, name: str, default):
        try:
            return json.loads(self.files.download(name))
        except Exception:
            return default

    def save_json(self, name: str, data) -> None:
        self.files.upload(name, json.dumps(data, indent=1, default=float).encode())

    def add_feedback(self, record: dict) -> None:
        """Append one 👍/👎 (with the question, answer and passages) to _feedback.json."""
        items = self.load_json(FEEDBACK_FILE, [])
        items.append(record)
        self.save_json(FEEDBACK_FILE, items[-MAX_FEEDBACK:])

    def chunk_samples(self, n: int, seed: int | None = None) -> list[dict]:
        """n random text chunks (not figures) with enough text to ask a question about."""
        import random
        kw = self._keyword_index()
        def looks_like_references(text):          # bibliography chunks make useless test questions
            return len(re.findall(r"\b(?:19|20)\d{2}\b", text)) >= 4 or text.lower().count("et al") >= 2 \
                or len(re.findall(r"\b[A-Z]\.,", text)) >= 4

        pool = [i for i, (d, m) in enumerate(zip(kw["docs"], kw["metas"]))
                if m.get("kind") != "image" and len(d) > 300 and not looks_like_references(d)]
        random.Random(seed).shuffle(pool)
        return [{"text": kw["docs"][i], "source": kw["metas"][i]["source"], "page": kw["metas"][i].get("page") or None}
                for i in pool[:n]]

    # ------------------------------------------------------------------ search
    def _keyword_index(self) -> dict:
        with _lock:
            version = (_kw_version.get(self.key, 0), self.collection.count())
            cached = _kw_cache.get(self.key)
            if not cached or cached["version"] != version:
                got = self.collection.get(include=["documents", "metadatas"])
                cached = {"version": version, "ids": got["ids"], "docs": got["documents"], "metas": got["metadatas"],
                          "bm25": BM25([tokenize(d) for d in got["documents"]]),
                          "pos": {cid: i for i, cid in enumerate(got["ids"])}}
                _kw_cache[self.key] = cached
            return cached

    def _hybrid_rank(self, queries: list[str], mode: str = "hybrid") -> list[tuple[str, float]]:
        """mode: "hybrid" (vectors + keywords), "vector" or "keyword" (used by the evaluation page)."""
        n = self.collection.count()
        if not n:
            return []
        kw = self._keyword_index()
        fused: dict[str, float] = {}

        def add(ranked_ids):
            for rank, cid in enumerate(ranked_ids, start=1):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)

        for q in queries:
            if mode in ("hybrid", "vector"):
                add(self.collection.query(query_embeddings=[embed_query(q)], n_results=min(POOL, n))["ids"][0])
            if mode == "vector":
                continue
            s = kw["bm25"].scores(tokenize(q))
            add([kw["ids"][i] for i in sorted((i for i in range(len(s)) if s[i] > 0), key=lambda i: -s[i])[:POOL]])
        return sorted(fused.items(), key=lambda kv: -kv[1])

    def _named_figures(self, texts: list[str]) -> list[dict]:
        refs = {(("table" if kind.lower().startswith("tab") else "figure"), num)
                for t in texts for kind, nums in _FIGURE_REF.findall(t or "") for num in re.findall(r"\d+", nums)}
        if not refs:
            return []
        got = self.collection.get(where={"kind": "image"}, include=["documents", "metadatas"])
        out = []
        for text, m in zip(got["documents"], got["metadatas"]):
            cap = (m.get("caption") or "").strip()
            hit = re.match(r"(fig(?:ure)?\.?|table)\s*(\d+)", cap, re.IGNORECASE)
            if hit and (("table" if hit.group(1).lower().startswith("tab") else "figure"), hit.group(2)) in refs:
                out.append(_hit(text, m, 1.0))
        return out

    def _best_figure(self, queries: list[str], ranked) -> list[dict]:
        kw = self._keyword_index()
        for cid, score in ranked:
            i = kw["pos"].get(cid)
            if i is not None and kw["metas"][i].get("kind") == "image":
                return [_hit(kw["docs"][i], kw["metas"][i], score)]
        try:
            r = self.collection.query(query_embeddings=[embed_query(queries[0])], n_results=1, where={"kind": "image"})
            return [_hit(r["documents"][0][0], r["metadatas"][0][0], 0.0)] if r["ids"][0] else []
        except Exception:
            return []

    def search(self, question: str, attachments: list[dict] | None = None, k: int = 4,
               queries: list[str] | None = None, mode: str = "hybrid") -> list[dict]:
        """The passages to answer from — ONLY from this user's library (+ their chat attachments)."""
        attachments = attachments or []
        all_queries = list(dict.fromkeys([question] + [q for q in (queries or []) if q and q.strip()]))[:4]

        figures = self._named_figures(all_queries)
        ranked = self._hybrid_rank(all_queries, mode)
        if not figures and _WANTS_PICTURE.search(question or "") and ranked:
            figures = self._best_figure(all_queries, ranked)
        taken = {f["image"] for f in figures}

        lib = []
        if ranked:
            kw = self._keyword_index()
            for cid, score in ranked:
                i = kw["pos"].get(cid)
                if i is not None and kw["metas"][i].get("image") not in taken:
                    lib.append(_hit(kw["docs"][i], kw["metas"][i], score))

        if not attachments:
            return (figures + lib)[:max(k, len(figures))]

        qvecs = np.array([embed_query(q) for q in all_queries], dtype=np.float32)
        scores = (np.stack([a["vector"] for a in attachments]) @ qvecs.T).max(axis=1)
        chosen = range(len(attachments)) if len(attachments) <= ATTACH_WHOLE_LIMIT \
            else sorted(np.argsort(-scores)[:ATTACH_TOP])
        att = [{"text": attachments[i]["text"], "source": attachments[i]["source"], "page": attachments[i]["page"],
                "score": float(scores[i]), "attached": True} for i in chosen]
        best_att = float(scores.max())
        extra = []
        if self.collection.count():
            r = self.collection.query(query_embeddings=[qvecs[0].tolist()],
                                      n_results=min(LIB_WITH_ATTACH, self.collection.count()))
            extra = [_hit(t, m, 1 - d) for t, m, d in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])
                     if 1 - d > best_att and m.get("image") not in taken]
        return att + figures + extra


# =========================================================================== temporary attachments
def read_attachment(name: str, data: bytes) -> list[dict]:
    """Parse + chunk + embed one attached file entirely in memory/temp. Not saved anywhere."""
    name = safe_name(name)
    if Path(name).suffix.lower() not in SUPPORTED:
        raise ValueError(f"{name}: only PDF, Word, PowerPoint, Markdown and text files are supported")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / name
        path.write_bytes(data)
        chunks = chunk_pages(extract_any(path))
    if not chunks:
        raise ValueError(f"{name}: no readable text (a scanned PDF, or slides with only pictures?)")
    vectors = np.array(embed_passages([c["text"] for c in chunks]), dtype=np.float32)
    return [{"text": c["text"], "source": name, "page": c["page"], "attached": True, "vector": v}
            for c, v in zip(chunks, vectors)]
