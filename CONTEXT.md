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
- Fully indexed 2026-07-15: 1,347 PDFs / 277,958 chunks (28 min for the
  full bge-m3 run on Apple Silicon; exact chunk parity with v1)
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

## Quality comparison outcome (2026-07-15)
v2 is decisively more semantic (dense ranks dominate its fusion; keyword
queries still nail exact docs like the Oracle offer letter). BUT: on
"What projects do I have where I managed change", v2 returns PM TEXTBOOKS
(PMBOK, practice guides) — semantically closest to the words, not the
user's intent (his own project docs). Root cause: the corpus mixes
reference books with personal documents. Candidate remedies, not yet
applied: exclude_globs for textbook folders (e.g. ~/Documents/PMI),
raising sparse_candidates for own-doc queries, or a doc-type facet.
