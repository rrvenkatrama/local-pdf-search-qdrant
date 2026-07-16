# Project Context — v2 (Qdrant + bge-m3)

_Last updated: 2026-07-15_

## What this is
v2 of the fully-local hybrid PDF search engine over `~/Documents`.
Repo: https://github.com/rrvenkatrama/local-pdf-search-qdrant (private).
Predecessor: https://github.com/rrvenkatrama/local-pdf-search-chromadb-sqlite-fts5
(folder `/Users/rajeshramani/ai/Local PDF Search RAG KB`, UI :8130 — still live).
Built because v1's MiniLM (384-dim) skewed keyword-heavy; bge-m3 (1024-dim)
is the quality lever, targeted at text-embedding-3-small parity.

## Current deployed state (this Mac)
- Qdrant 1.18.2 native Apple Silicon binary (`bin/qdrant`, NO Docker),
  port 6333 — launchd agent `com.rajesh.pdfqdrant.qdrant` (always on)
- UI: http://localhost:8131/ — `com.rajesh.pdfqdrant.server` (always on)
- Daily indexer 08:15 — `com.rajesh.pdfqdrant.indexer`
- As of writing: first FULL index still running (bge-m3 is ~20× MiniLM
  compute; check `curl -s localhost:8131/status`)
- Generated data in `data/` (gitignored): `qdrant-storage/`, `manifest.db`, logs

## Architecture in one line
indexer.py (crawl → sentence-chunk → bge-m3 dense + fastembed BM25 sparse →
one Qdrant point each) ⇢ server.py (ONE hybrid query: dense+sparse prefetch,
RRF fused INSIDE Qdrant) ⇢ static/index.html.

## Tuning & debugging
- Semantic/keyword balance: `dense_candidates` (30) vs `sparse_candidates`
  (20) in config.yaml — raise dense to lean more semantic.
- Qdrant dashboard: http://localhost:6333/dashboard
- Sparse BM25: fastembed computes term weights; IDF is applied SERVER-side
  (`Modifier.IDF` on the sparse index) so rarity tracks the live corpus.

## Open items
- v1-vs-v2 quality comparison on "What projects do I have where I managed
  change" (the query that motivated v2) once the full index completes.
