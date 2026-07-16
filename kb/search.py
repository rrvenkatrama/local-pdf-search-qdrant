"""Query orchestration and result formatting.

Flow: embed the query with both models → one hybrid (RRF-fused) Qdrant
query → build highlighted snippets from the chunk text in each payload.

Snippets are built client-side (Qdrant has no equivalent of FTS5's
snippet() function): we window the chunk text around the densest cluster
of query-word hits and <mark>-highlight the words. Snippet text is
HTML-escaped first, so raw PDF text can't inject markup into the UI.
"""

import html
import re

from kb import embedder, store
from kb.config import Config

_SNIPPET_WORDS = 50  # target snippet length in words


def _query_words(query: str) -> list[str]:
    """Meaningful words to highlight (skip 1–2 letter noise)."""
    return [w for w in re.findall(r"\w+", query) if len(w) >= 3]


def make_snippet(text: str, query: str) -> str:
    """Window the chunk text around the most query-relevant region.

    1. Find every position where a query word occurs.
    2. Center a ~50-word window on the region with the most hits
       (fall back to the start of the chunk when nothing matches —
       common for purely semantic hits).
    3. HTML-escape, then wrap query words in <mark>.
    """
    words = text.split()
    q = [w.lower() for w in _query_words(query)]
    hit_positions = [i for i, w in enumerate(words)
                     if any(qw in w.lower() for qw in q)]

    if hit_positions:
        # Densest region ≈ the window containing the most hit positions.
        best_start, best_hits = 0, 0
        for pos in hit_positions:
            start = max(0, pos - _SNIPPET_WORDS // 2)
            hits = sum(1 for p in hit_positions if start <= p < start + _SNIPPET_WORDS)
            if hits > best_hits:
                best_start, best_hits = start, hits
        start = best_start
    else:
        start = 0

    window = words[start:start + _SNIPPET_WORDS]
    snippet = html.escape(" ".join(window))
    if start > 0:
        snippet = "… " + snippet
    if start + _SNIPPET_WORDS < len(words):
        snippet += " …"

    for word in set(_query_words(query)):
        snippet = re.sub(rf"\b({re.escape(word)})\b", r"<mark>\1</mark>",
                         snippet, flags=re.IGNORECASE)
    return snippet


def hybrid_search(cfg: Config, query: str, top_k: int | None = None) -> list[dict]:
    """Run the hybrid query and return display-ready results.

    Each result dict: file_path, filename, page, snippet, score (RRF),
    dense_rank, sparse_rank (None when the chunk wasn't in that engine's
    candidate list — useful for judging semantic vs keyword balance).
    """
    top_k = top_k or cfg.top_k
    dense, sparse = embedder.embed_query(cfg.dense_model, cfg.sparse_model, query)

    hits = store.hybrid_query(cfg, dense, sparse, top_k)
    dense_rank, sparse_rank = store.single_engine_ranks(
        cfg, dense, sparse, max(cfg.dense_candidates, cfg.sparse_candidates)
    )

    results = []
    for point in hits:
        payload = point.payload or {}
        path = payload.get("file_path", "")
        results.append({
            "chunk_id": str(point.id),
            "file_path": path,
            "filename": path.rsplit("/", 1)[-1],
            "page": payload.get("page"),
            "snippet": make_snippet(payload.get("text", ""), query),
            "score": round(point.score, 5),
            "dense_rank": dense_rank.get(str(point.id)),
            "sparse_rank": sparse_rank.get(str(point.id)),
        })
    return results
