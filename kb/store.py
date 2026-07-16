"""Qdrant collection — BOTH search indexes live here, and fusion too.

One collection ("pdf_chunks") holds, per chunk (point):
  - named dense vector  "dense"  (1024-dim bge-m3, cosine) → semantic search
  - named sparse vector "bm25"   (term weights, IDF applied server-side)
                                                           → keyword search
  - payload: {file_path, page, text, doc_type}

Hybrid search is ONE server-side query: two prefetch branches (dense +
sparse) fused with Reciprocal Rank Fusion inside Qdrant — no client-side
merging code, unlike the v1 Chroma+FTS5 edition.

Point IDs are deterministic UUIDs derived from (file_path, sequence), so a
file's chunks can always be re-derived, and deletion uses a payload filter
on file_path (backed by a keyword payload index).
"""

import uuid

from qdrant_client import QdrantClient, models

from kb.config import Config

_client: QdrantClient | None = None


def client(cfg: Config) -> QdrantClient:
    """Connect to the local Qdrant server (launchd agent, native binary)."""
    global _client
    if _client is None:
        _client = QdrantClient(url=cfg.qdrant_url, timeout=60)
    return _client


def ensure_collection(cfg: Config) -> None:
    """Create the collection on first run; (re)declare payload indexes.

    Index declarations run every time (they are idempotent) so fields
    added after the collection was first created — doc_type arrived after
    the initial full index — get indexed without a rebuild.
    """
    c = client(cfg)
    if not c.collection_exists(cfg.collection):
        c.create_collection(
            collection_name=cfg.collection,
            vectors_config={
                "dense": models.VectorParams(
                    size=cfg.dense_dim, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                # Modifier.IDF = Qdrant weighs each term by corpus-wide rarity
                # at query time — the "keyword-ness" half of BM25. The term-
                # frequency half comes from the fastembed BM25 model weights.
                "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )
    # file_path: fast delete-by-file. doc_type: the UI's facet filter.
    for field in ("file_path", "doc_type"):
        c.create_payload_index(
            collection_name=cfg.collection,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )


def point_id(path: str, seq: int) -> str:
    """Deterministic UUID for chunk #seq of a file."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"pdfkb://{path}#{seq}"))


def upsert_chunks(cfg: Config, points: list[models.PointStruct]) -> None:
    """Insert/overwrite chunk points, batched to keep request sizes sane."""
    c = client(cfg)
    batch = 256
    for i in range(0, len(points), batch):
        c.upsert(collection_name=cfg.collection, points=points[i:i + batch],
                 wait=True)


def make_point(cfg: Config, path: str, seq: int, page: int, text: str,
               dense: list[float], sparse: dict) -> models.PointStruct:
    """Build one Qdrant point for a chunk."""
    return models.PointStruct(
        id=point_id(path, seq),
        vector={
            "dense": dense,
            "bm25": models.SparseVector(**sparse),
        },
        payload={"file_path": path, "page": page, "text": text,
                 "doc_type": cfg.doc_type(path)},
    )


def set_doc_type(cfg: Config, paths: list[str], doc_type: str) -> None:
    """Stamp doc_type onto every chunk of the given files (payload-only,
    no re-embedding). Used by `indexer.py --retag` to reclassify the
    existing index after doc_type_globs changes."""
    c = client(cfg)
    batch = 200
    for i in range(0, len(paths), batch):
        c.set_payload(
            collection_name=cfg.collection,
            payload={"doc_type": doc_type},
            points=models.Filter(must=[
                models.FieldCondition(
                    key="file_path",
                    match=models.MatchAny(any=paths[i:i + batch]),
                )
            ]),
            wait=True,
        )


def _doc_type_filter(doc_type: str | None) -> models.Filter | None:
    if not doc_type:
        return None
    return models.Filter(must=[
        models.FieldCondition(key="doc_type",
                              match=models.MatchValue(value=doc_type))
    ])


def delete_file_chunks(cfg: Config, path: str) -> None:
    """Remove every chunk of a file (file modified or deleted)."""
    client(cfg).delete(
        collection_name=cfg.collection,
        points_selector=models.FilterSelector(
            filter=models.Filter(must=[
                models.FieldCondition(
                    key="file_path", match=models.MatchValue(value=path)
                )
            ])
        ),
        wait=True,
    )


def hybrid_query(cfg: Config, dense: list[float], sparse: dict,
                 top_k: int,
                 doc_type: str | None = None) -> list[models.ScoredPoint]:
    """The heart of v2: one server-side hybrid query.

    Qdrant runs both branches and fuses them with RRF internally:
        dense  branch → top dense_candidates by cosine similarity
        sparse branch → top sparse_candidates by BM25
        → RRF → top_k
    The dense/sparse candidate counts (config.yaml) act as a bias knob:
    more dense candidates → more semantic results survive fusion.

    doc_type filters BOTH branches (each Prefetch carries the filter),
    so the facet doesn't just mask results — filtered-out chunks free
    their candidate slots for matching ones.
    """
    flt = _doc_type_filter(doc_type)
    return client(cfg).query_points(
        collection_name=cfg.collection,
        prefetch=[
            models.Prefetch(query=dense, using="dense",
                            limit=cfg.dense_candidates, filter=flt),
            models.Prefetch(query=models.SparseVector(**sparse), using="bm25",
                            limit=cfg.sparse_candidates, filter=flt),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
        with_payload=True,
    ).points


def single_engine_ranks(cfg: Config, dense: list[float], sparse: dict,
                        limit: int,
                        doc_type: str | None = None) -> tuple[dict, dict]:
    """Per-engine rank of each chunk id — debug info shown by the API.

    Two extra lightweight queries (ids only, no payload); a few ms each.
    """
    c = client(cfg)
    flt = _doc_type_filter(doc_type)
    dense_hits = c.query_points(cfg.collection, query=dense, using="dense",
                                limit=limit, with_payload=False,
                                query_filter=flt).points
    sparse_hits = c.query_points(cfg.collection,
                                 query=models.SparseVector(**sparse),
                                 using="bm25", limit=limit,
                                 with_payload=False, query_filter=flt).points
    dense_rank = {str(p.id): i + 1 for i, p in enumerate(dense_hits)}
    sparse_rank = {str(p.id): i + 1 for i, p in enumerate(sparse_hits)}
    return dense_rank, sparse_rank


def count(cfg: Config) -> int:
    """Total chunks in the collection."""
    return client(cfg).count(cfg.collection, exact=True).count
