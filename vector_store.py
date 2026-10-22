"""
STEP 3 — Embed the chunks and save them in a vector database
=============================================================

GOAL
    Turn every chunk into a vector, store it in a vector database on disk, and
    check that a question brings back the right chunks — BEFORE any LLM.

RUN
    python step3_vector_store.py                       # index uploads/ + demo search
    python step3_vector_store.py "your question here"  # search only

TOOLS (all open-source, all run on your machine, no API keys)
    * Embedding model: sentence-transformers/all-MiniLM-L6-v2
        - turns text into a list of 384 numbers (a vector)
        - texts with similar MEANING get vectors pointing in similar
          directions, so "cheaper inference" finds "reduce serving cost"
        - ~90 MB, downloaded automatically the first time
    * Vector database: ChromaDB
        - runs inside Python, saves to the folder vector_db/
        - stores, for every chunk: an id, the vector, the text, and metadata

WHAT GETS STORED PER CHUNK
    id        "3f2a9c...-7"        (document id + chunk number)
    vector    [0.012, -0.044, ...] (from the embedding model)
    text      "AWQ protects salient channels..."
    metadata  {"doc_id", "source": "paper.pdf", "page": 2}

    The metadata is what makes the rest possible:
      - citations:  source + page
      - deletion:   collection.delete(where={"doc_id": ...}) removes a whole file

HOW SEARCH WORKS
    1. Embed the question with the SAME model.
    2. Chroma returns the k stored vectors closest to it (cosine distance).
    3. similarity = 1 - distance  (1.0 = identical meaning)

SYNC, NOT JUST ADD
    index_folder() makes the database match the uploads/ folder:
      new file      -> indexed
      same file     -> skipped (recognised by a hash of its content)
      deleted file  -> its chunks are removed
      edited file   -> new content = new hash, so old version removed, new added
"""
import hashlib
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from extract import DOCUMENTS_DIR, extract_file, list_files
from chunk import chunk_pages

from pathlib import Path
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"           # name on Hugging Face
MODEL_DIR = Path(__file__).parent / "models" / "all-MiniLM-L6-v2" # local copy
DB_DIR = "vector_db"
COLLECTION = "documents"

_model = None


def load_model() -> SentenceTransformer:
    """First run: download from Hugging Face and save to models/.
    Every run after that: load from models/ (no internet needed)."""
    if MODEL_DIR.exists():
        return SentenceTransformer(str(MODEL_DIR))
    print(f"Downloading {EMBED_MODEL} (one time only)...")
    model = SentenceTransformer(EMBED_MODEL)
    model.save(str(MODEL_DIR))
    print(f"Saved to {MODEL_DIR}")
    return model


def embed(texts: list[str]) -> list[list[float]]:
    """Text -> vectors. normalize=True makes cosine similarity a simple dot product."""
    global _model
    if _model is None:                      # load once, reuse for every call
        _model = load_model()
    return _model.encode(texts, normalize_embeddings=True).tolist()


def get_collection():
    client = chromadb.PersistentClient(path=DB_DIR)
    # embedding_function=None: we compute vectors ourselves, so you can see it happen.
    return client.get_or_create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine"}, embedding_function=None)


def file_id(path: Path) -> str:
    """Identify a file by its content, so renaming doesn't matter and edits are noticed."""
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:12]


def index_file(path: Path, collection=None) -> int:
    """Extract -> chunk -> embed -> store one file. Returns number of chunks added."""
    collection = collection or get_collection()
    doc_id = file_id(path)
    if collection.get(where={"doc_id": doc_id}, limit=1)["ids"]:
        return 0                                         # already indexed
    chunks = chunk_pages(extract_file(path))
    if not chunks:
        print(f"  ! no text in {Path(path).name} (scanned PDF? needs OCR)")
        return 0
    collection.add(
        ids=[f"{doc_id}-{c['chunk_no']}" for c in chunks],
        embeddings=embed([c["text"] for c in chunks]),
        documents=[c["text"] for c in chunks],
        # Chroma metadata can't hold None, so whole-file documents get page 0
        metadatas=[{"doc_id": doc_id, "source": c["source"], "page": c["page"] or 0,
                    "path": str(Path(path).resolve())} for c in chunks],
    )
    return len(chunks)


def list_documents(collection=None) -> list[dict]:
    collection = collection or get_collection()
    docs = {}
    for meta in collection.get(include=["metadatas"])["metadatas"]:
        d = docs.setdefault(meta["doc_id"], {"doc_id": meta["doc_id"], "source": meta["source"],
                                             "path": meta["path"], "chunks": 0})
        d["chunks"] += 1
    return list(docs.values())


def delete_document(doc_id: str, delete_file: bool = True, collection=None) -> None:
    """Remove all chunks of one document (and its file in uploads/)."""
    collection = collection or get_collection()
    docs = {d["doc_id"]: d for d in list_documents(collection)}
    collection.delete(where={"doc_id": doc_id})
    path = Path(docs[doc_id]["path"]) if doc_id in docs else None
    if delete_file and path and path.exists() and path.parent == DOCUMENTS_DIR.resolve():
        path.unlink()


def index_folder(folder: Path = DOCUMENTS_DIR) -> None:
    """Make the vector database match the folder."""
    collection = get_collection()
    present = set()
    for path in list_files(folder):
        present.add(file_id(path))
        added = index_file(path, collection)
        print(f"  {'added  ' if added else 'skipped'} {path.name}" + (f" ({added} chunks)" if added else ""))
    for doc in list_documents(collection):
        if doc["doc_id"] not in present:
            delete_document(doc["doc_id"], delete_file=False, collection=collection)
            print(f"  removed {doc['source']} (file no longer in folder)")


def search(question: str, k: int = 5) -> list[dict]:
    """Return the k most similar chunks: [{"text", "source", "page", "score"}]."""
    collection = get_collection()
    if collection.count() == 0:
        return []
    result = collection.query(query_embeddings=embed([question]), n_results=min(k, collection.count()))
    return [
        {"text": text, "source": meta["source"], "page": meta["page"] or None, "score": 1 - dist}
        for text, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0])
    ]


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Indexing uploads/ ...")
        index_folder()
        print(f"\nVector database now holds {get_collection().count()} chunks:")
        for d in list_documents():
            print(f"  {d['source']}: {d['chunks']} chunks")
        question = "What are the main ways to reduce LLM inference cost?"
    else:
        question = " ".join(sys.argv[1:])

    print(f"\nQUESTION: {question}\n")
    for rank, hit in enumerate(search(question), start=1):
        page = f", page {hit['page']}" if hit["page"] else ""
        print(f"#{rank}  score {hit['score']:.3f}  {hit['source']}{page}")
        print(f"    {hit['text'][:200]}...\n")
    print("Check: would YOU answer the question from these chunks? If not, no LLM can.")