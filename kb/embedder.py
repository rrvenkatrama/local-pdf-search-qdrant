"""Local embedding models — one dense (semantic), one sparse (keyword).

Dense — BAAI/bge-m3 via sentence-transformers. 1024-dimensional
multilingual vectors with retrieval quality comparable to OpenAI's
text-embedding-3-small, running entirely on this machine (Apple Silicon
GPU via MPS when available). ~2.3 GB one-time download, then offline.

Sparse — "Qdrant/bm25" via fastembed. Produces a term→weight mapping per
text (tokenized, stemmed, term-frequency-saturated). Crucially, the IDF
half of BM25 is applied by the QDRANT SERVER at query time (the sparse
index is created with Modifier.IDF in kb/store.py), so keyword rarity is
always measured against the current corpus.

Both the indexer (embedding chunks) and the search service (embedding
queries) use this module — the same models must produce both sides.
"""

from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer

_dense: SentenceTransformer | None = None
_sparse: SparseTextEmbedding | None = None


def _dense_model(name: str) -> SentenceTransformer:
    """Load the dense model once (takes several seconds; ~2 GB of weights)."""
    global _dense
    if _dense is None:
        # device=None → sentence-transformers auto-picks MPS (Apple GPU) or CPU.
        _dense = SentenceTransformer(name, device=None)
    return _dense


def _sparse_model(name: str) -> SparseTextEmbedding:
    """Load the BM25 sparse model once (small, CPU-only)."""
    global _sparse
    if _sparse is None:
        _sparse = SparseTextEmbedding(model_name=name)
    return _sparse


def embed_dense(model_name: str, texts: list[str]) -> list[list[float]]:
    """Dense-embed a batch of texts. Normalized for cosine similarity."""
    vectors = _dense_model(model_name).encode(
        texts,
        batch_size=32,   # bge-m3 is a large model; keep batches moderate
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.tolist()


def embed_sparse(model_name: str, texts: list[str]) -> list[dict]:
    """Sparse-embed a batch of texts → [{"indices": [...], "values": [...]}]."""
    return [
        {"indices": emb.indices.tolist(), "values": emb.values.tolist()}
        for emb in _sparse_model(model_name).embed(texts)
    ]


def embed_query(dense_name: str, sparse_name: str, query: str) -> tuple:
    """Embed one search query with both models → (dense_vector, sparse_dict)."""
    dense = embed_dense(dense_name, [query])[0]
    sparse_emb = next(_sparse_model(sparse_name).query_embed(query))
    sparse = {"indices": sparse_emb.indices.tolist(),
              "values": sparse_emb.values.tolist()}
    return dense, sparse
