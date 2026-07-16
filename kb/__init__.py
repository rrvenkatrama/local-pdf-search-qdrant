# kb — the shared library used by both indexer.py and server.py.
#
# Modules:
#   config.py    — loads config.yaml, defines all project paths
#   manifest.py  — SQLite bookkeeping: which files are indexed, run history
#   chunker.py   — sentence-aligned chunking with overlap
#   embedder.py  — dense (bge-m3) + sparse (BM25) local embedding models
#   store.py     — Qdrant collection: both indexes + server-side hybrid fusion
#   search.py    — query orchestration + snippet building
#
# v2 architecture note: unlike the Chroma+FTS5 edition, BOTH search indexes
# (semantic dense vectors and keyword BM25 sparse vectors) live in one
# Qdrant collection, and hybrid fusion (RRF) happens inside the Qdrant
# server in a single query.
