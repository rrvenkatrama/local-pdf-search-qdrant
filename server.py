#!/usr/bin/env python3
"""FastAPI search service (Qdrant edition).

Serves the HTML UI and four JSON endpoints:

  POST /search   {query, top_k?}  → hybrid search results (see kb/search.py)
  POST /open     {path}           → open the PDF in Preview/Acrobat (macOS `open`)
  POST /reindex                   → launch indexer.py in the background
  GET  /status                    → last run summary + totals + indexing flag

Start with:  ./venv/bin/python server.py     (UI at http://localhost:8131/)

Because Qdrant is a real server (unlike embedded Chroma in v1), there is
no stale-snapshot problem: every search sees whatever the indexer has
upserted, immediately. The dense embedding model loads once at startup.
"""

import subprocess
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from kb import embedder, manifest, search, store
from kb.config import LOCK_PATH, PROJECT_ROOT, load_config

cfg = load_config()
app = FastAPI(title="Local PDF Search (Qdrant)")


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str
    top_k: int | None = None


class OpenRequest(BaseModel):
    path: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/search")
def search_endpoint(req: SearchRequest) -> dict:
    """Hybrid (dense + BM25, RRF-fused inside Qdrant) search."""
    query = req.query.strip()
    if not query:
        return {"query": query, "results": []}
    return {"query": query, "results": search.hybrid_search(cfg, query, req.top_k)}


@app.post("/open")
def open_endpoint(req: OpenRequest) -> dict:
    """Open a result PDF in the default macOS viewer (Preview/Acrobat).

    Safety: only paths that live under a configured root AND are present in
    the manifest can be opened — the endpoint can't be used to open
    arbitrary files.
    """
    path = Path(req.path).resolve()
    if not any(path.is_relative_to(root) for root in cfg.roots):
        raise HTTPException(403, "Path is outside the configured roots")
    conn = manifest.connect()
    try:
        known = manifest.get(conn, str(path))
    finally:
        conn.close()
    if known is None or not path.exists():
        raise HTTPException(404, "File not found in the index")
    subprocess.run(["open", str(path)], check=False)
    return {"opened": str(path)}


@app.post("/reindex")
def reindex_endpoint() -> dict:
    """Start the indexer in the background (same script launchd runs daily)."""
    if LOCK_PATH.exists():
        raise HTTPException(409, "An index run is already in progress")
    subprocess.Popen(
        [sys.executable, str(PROJECT_ROOT / "indexer.py")],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,   # indexer logs to data/index.log itself
        stderr=subprocess.DEVNULL,
        start_new_session=True,      # keeps running if the server restarts
    )
    return {"started": True}


@app.get("/status")
def status_endpoint() -> dict:
    """Everything the UI footer shows: totals, last run, whether indexing now."""
    conn = manifest.connect()
    try:
        return {
            "indexing": LOCK_PATH.exists(),
            "totals": manifest.totals(conn),
            "qdrant_points": store.count(cfg),
            "last_run": manifest.last_run(conn),
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Static UI (registered last so API routes take precedence)
# ---------------------------------------------------------------------------

@app.get("/")
def index_page() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "static" / "index.html")


app.mount("/static", StaticFiles(directory=PROJECT_ROOT / "static"))


if __name__ == "__main__":
    # Warm both embedding models before accepting requests.
    print(f"Loading embedding models ({cfg.dense_model} + {cfg.sparse_model}) …")
    embedder.embed_query(cfg.dense_model, cfg.sparse_model, "warmup")
    store.ensure_collection(cfg)
    print(f"Ready — UI at http://localhost:{cfg.port}/")
    uvicorn.run(app, host="127.0.0.1", port=cfg.port, log_level="warning")
