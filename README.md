# Local PDF Search — Qdrant Edition

Fully local hybrid search over your PDF documents. v2 of
[local-pdf-search-chromadb-sqlite-fts5](https://github.com/rrvenkatrama/local-pdf-search-chromadb-sqlite-fts5):
a much stronger embedding model and Qdrant replacing both Chroma and
SQLite FTS5. No cloud, no API keys, no Docker.

## What changed vs v1

| | v1 (`local-pdf-search-chromadb-sqlite-fts5`) | v2 (this repo) |
|---|---|---|
| Dense model | multilingual MiniLM (384-dim) | **BAAI/bge-m3 (1024-dim)** — quality on par with OpenAI text-embedding-3-small |
| Semantic index | Chroma (embedded) | **Qdrant server** (native Apple Silicon binary, launchd — no Docker) |
| Keyword index | SQLite FTS5 (BM25) | **Qdrant sparse vectors** (fastembed BM25 term weights + server-side IDF) |
| Fusion | RRF in Python | **RRF inside Qdrant** — one hybrid query |
| Staleness | server had to refresh after re-index | none — Qdrant is a real server |
| UI port | 8130 | 8131 (both editions can run side by side) |

## How it works

```
              ┌──────────────────────────────────────────────┐
 launchd 8:15 │  indexer.py                                   │
 UI button ──▶│  crawl ~/Documents → extract → sentence-      │
              │  chunk → embed dense (bge-m3) + sparse (BM25) │
              └───────────────┬──────────────────────────────┘
                              ▼  upsert points
                    Qdrant server :6333  (launchd, bin/qdrant)
                    one collection, per chunk:
                      dense[1024] + sparse{term→weight} + payload{path,page,text,doc_type}
                              ▲  one hybrid query (RRF fusion server-side)
              ┌───────────────┴──────────────────────────────┐
 browser ────▶│  server.py (FastAPI :8131) + static UI        │
              └──────────────────────────────────────────────┘
```

- **Chunking:** complete sentences only (pysbd), 8-sentence windows with
  4-sentence overlap, never crossing page boundaries.
- **Change detection:** mtime+size fast path, sha1 to confirm; modified
  files have all old points deleted before re-inserting; deleted files
  are purged. Scanned (no text layer) and password-protected PDFs are
  skipped and logged.
- **Semantic/keyword balance:** `dense_candidates` vs `sparse_candidates`
  in `config.yaml` biases fusion (30/20 default leans semantic).
- **Doc-type facet:** the corpus mixes personal documents with reference
  material (textbooks, courseware), and semantic search happily ranks a
  PMBOK chapter above your own project docs. `doc_type_globs` in
  `config.yaml` tags every file `personal` or `reference` at index time;
  the UI's All / My docs / Reference chips filter both prefetch branches
  server-side. After editing the globs, re-stamp the existing index with
  `./venv/bin/python indexer.py --retag` (payload-only, seconds).

## Setup

```bash
./install.sh                    # venv + Qdrant binary + 3 launchd agents
./venv/bin/python indexer.py    # first full index (bge-m3 is a big model —
                                #  expect ~1 hour for ~1,600 PDFs on M-series)
# → http://localhost:8131/
```

## Files

| Path | What it is |
|---|---|
| `config.yaml` | folders, models, chunking, fusion balance, ports |
| `indexer.py` | the crawler (daily via launchd + on demand) |
| `server.py` | FastAPI search service + static UI |
| `kb/` | shared library (see `kb/__init__.py` for a module map) |
| `static/index.html` | the single-file web UI |
| `launchd/`, `install.sh` | Qdrant server, search service, daily indexer |
| `bin/qdrant` | native Qdrant binary (downloaded by install.sh, gitignored) |
| `data/` | generated: Qdrant storage, manifest DB, logs (gitignored) |

## Handy debugging

```bash
curl -s localhost:6333/collections/pdf_chunks | jq   # collection stats
sqlite3 data/manifest.db "SELECT status, COUNT(*) FROM manifest GROUP BY status"
tail -f data/index.log
# Qdrant web dashboard:
open http://localhost:6333/dashboard
```
