"""
The embedding model used for every user's document library: BAAI/bge-small-en-v1.5.

WHY THIS MODEL
    all-MiniLM-L6-v2 (2021) is tiny and fast but weak at paraphrases and
    technical vocabulary, so questions had to be worded almost like the text.
    bge-small-en-v1.5 is the same size class (384 numbers per vector, ~130 MB),
    runs on CPU, fits Streamlit Cloud's free memory, and is clearly better at
    matching a question to a passage that says the same thing differently.

QUERIES vs PASSAGES
    BGE models were trained with a short instruction in front of search
    queries (not in front of documents). embed_query() adds it.

LOCAL COPY
    The model downloads once from Hugging Face and is saved in models/, then
    loaded from there (no internet needed after the first run).

CHANGING THE MODEL LATER
    Vectors from different models can't be mixed, so knowledge_base.py keeps
    one database collection per model (collection_suffix()). Changing MODEL
    starts a fresh collection that is rebuilt automatically from storage.
"""
import re
from pathlib import Path

MODEL = "BAAI/bge-small-en-v1.5"
MODEL_DIR = Path(__file__).parent / "models" / MODEL.split("/")[-1]
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

_model = None


def _load():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        if MODEL_DIR.exists():
            _model = SentenceTransformer(str(MODEL_DIR))
        else:
            print(f"[embedder] downloading {MODEL} (one time only)…")
            _model = SentenceTransformer(MODEL)
            MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
            _model.save(str(MODEL_DIR))
    return _model


def embed_passages(texts: list[str]) -> list[list[float]]:
    """Vectors for document chunks and figure descriptions."""
    if not texts:
        return []
    return _load().encode(texts, normalize_embeddings=True, batch_size=32).tolist()


def embed_query(text: str) -> list[float]:
    """Vector for a search query (with the instruction BGE was trained with)."""
    return _load().encode([QUERY_INSTRUCTION + text], normalize_embeddings=True).tolist()[0]


def collection_suffix() -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", MODEL.split("/")[-1]).strip("-").lower()
