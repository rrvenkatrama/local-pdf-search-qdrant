#!/usr/bin/env python3
"""PDF crawler / indexer (Qdrant edition).

Scans the configured folders for PDFs and keeps the Qdrant collection
in sync with what's on disk:

  new file        → extract, chunk, embed (dense + sparse), upsert points
  modified file   → DELETE all its old points, re-insert
  unchanged file  → skipped (fast mtime+size check, hash to confirm)
  deleted file    → all its points purged
  no text layer   → skipped and logged (scanned PDFs; OCR is out of scope)
  encrypted       → skipped and logged
  unreadable      → logged as failed, indexing continues

Runs to completion and exits — launched daily by launchd and on demand by
the UI's "Re-index now" button. A lock file prevents overlapping runs.

Usage:
    python indexer.py             # index everything configured
    python indexer.py --limit 20  # stop after 20 changed files (testing)
    python indexer.py --retag     # re-stamp doc_type on the existing index
                                  # (after editing doc_type_globs; payload-only,
                                  # no re-embedding, takes seconds)
"""

import argparse
import fnmatch
import hashlib
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import fitz  # pymupdf

from kb import chunker, embedder, manifest, store
from kb.config import LOCK_PATH, LOG_PATH, load_config

log = logging.getLogger("indexer")


# ---------------------------------------------------------------------------
# Lock file — prevents the daily launchd run and the UI button colliding
# ---------------------------------------------------------------------------

def acquire_lock() -> bool:
    """Create the lock file with our pid. Returns False if another indexer
    is genuinely running; silently removes the lock if its owner is dead
    (e.g. machine rebooted mid-index)."""
    if LOCK_PATH.exists():
        try:
            other_pid = int(LOCK_PATH.read_text().strip())
            os.kill(other_pid, 0)  # signal 0 = "does this pid exist?"
            return False           # yes → a real indexer is running
        except (ValueError, ProcessLookupError, PermissionError):
            log.warning("Removing stale lock file (owner no longer running)")
            LOCK_PATH.unlink(missing_ok=True)
    LOCK_PATH.write_text(str(os.getpid()))
    return True


def release_lock() -> None:
    LOCK_PATH.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Filesystem scan + change detection
# ---------------------------------------------------------------------------

def find_pdfs(roots: list[Path], exclude_globs: list[str]) -> list[Path]:
    """All *.pdf files under the roots (recursive), minus excluded patterns."""
    pdfs: list[Path] = []
    for root in roots:
        if not root.exists():
            log.warning("Root does not exist, skipping: %s", root)
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                if not name.lower().endswith(".pdf"):
                    continue
                path = Path(dirpath) / name
                if any(fnmatch.fnmatch(str(path), g) for g in exclude_globs):
                    continue
                pdfs.append(path)
    return pdfs


def file_hash(path: Path) -> str:
    """sha1 of the file contents (streamed, so large files are fine)."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):  # 1 MB at a time
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Indexing one file
# ---------------------------------------------------------------------------

def extract_pages(path: Path) -> list[str] | None:
    """Text of each page. Returns None if the PDF is password-protected."""
    with fitz.open(path) as doc:
        if doc.needs_pass:
            return None
        return [page.get_text("text") for page in doc]


def index_file(conn, cfg, path: Path, stat: os.stat_result,
               content_hash: str) -> tuple[str, int]:
    """(Re)index one PDF. Returns (status, chunks_added).

    Old points are deleted first, so a modified file never leaves stale
    chunks behind.
    """
    path_str = str(path)

    pages = extract_pages(path)
    if pages is None:
        log.info("SKIP (encrypted): %s", path)
        manifest.upsert(conn, path_str, mtime=stat.st_mtime,
                        size=stat.st_size, content_hash=content_hash,
                        status="skipped_encrypted")
        return "skipped", 0

    # Chunk every page; each chunk remembers its 1-based page number.
    chunks: list[tuple[int, int, str]] = []  # (seq, page, text)
    seq = 0
    for page_number, page_text in enumerate(pages, start=1):
        for chunk_text in chunker.chunk_page(page_text, cfg.chunk_sentences,
                                             cfg.chunk_overlap_sentences):
            chunks.append((seq, page_number, chunk_text))
            seq += 1

    # Purge old points BEFORE inserting new ones (file may have shrunk).
    store.delete_file_chunks(cfg, path_str)

    if not chunks:
        # No usable text on any page → almost certainly a scanned PDF.
        log.info("SKIP (no text layer — scanned?): %s", path)
        manifest.upsert(conn, path_str, mtime=stat.st_mtime,
                        size=stat.st_size, content_hash=content_hash,
                        status="skipped_no_text", page_count=len(pages))
        return "skipped", 0

    # Embed with both models and build one point per chunk.
    texts = [text for _seq, _page, text in chunks]
    dense_vectors = embedder.embed_dense(cfg.dense_model, texts)
    sparse_vectors = embedder.embed_sparse(cfg.sparse_model, texts)
    points = [
        store.make_point(cfg, path_str, s, page, text, dense, sparse)
        for (s, page, text), dense, sparse
        in zip(chunks, dense_vectors, sparse_vectors)
    ]
    store.upsert_chunks(cfg, points)

    manifest.upsert(conn, path_str, mtime=stat.st_mtime, size=stat.st_size,
                    content_hash=content_hash, status="indexed",
                    page_count=len(pages), chunk_count=len(chunks))
    log.info("INDEXED (%d pages, %d chunks): %s", len(pages), len(chunks), path)
    return "indexed", len(chunks)


# ---------------------------------------------------------------------------
# Re-tag doc_type (facet) without re-indexing
# ---------------------------------------------------------------------------

def retag_doc_types() -> None:
    """Reclassify every indexed file per config doc_type_globs and stamp
    the doc_type payload onto its chunks. Pure payload update — vectors
    are untouched, so this is cheap enough to run after every glob edit."""
    cfg = load_config()
    store.ensure_collection(cfg)  # (re)declares the doc_type payload index
    conn = manifest.connect()
    by_type: dict[str, list[str]] = {}
    for path in manifest.all_paths(conn):
        by_type.setdefault(cfg.doc_type(path), []).append(path)
    conn.close()
    for dtype, paths in sorted(by_type.items()):
        store.set_doc_type(cfg, paths, dtype)
        log.info("Tagged %d files as doc_type=%s", len(paths), dtype)


# ---------------------------------------------------------------------------
# Main crawl
# ---------------------------------------------------------------------------

def run(limit: int | None = None) -> None:
    cfg = load_config()
    store.ensure_collection(cfg)
    conn = manifest.connect()
    started = datetime.now().isoformat(timespec="seconds")
    t0 = time.time()
    counts = {"files_scanned": 0, "files_indexed": 0, "files_skipped": 0,
              "files_failed": 0, "files_deleted": 0, "chunks_added": 0}

    pdfs = find_pdfs(cfg.roots, cfg.exclude_globs)
    counts["files_scanned"] = len(pdfs)
    log.info("Scan found %d PDFs under %s",
             len(pdfs), ", ".join(str(r) for r in cfg.roots))

    changed_processed = 0
    for path in pdfs:
        path_str = str(path)
        try:
            stat = path.stat()
            row = manifest.get(conn, path_str)

            # Fast path: same mtime and size as last time → unchanged.
            if row and row["mtime"] == stat.st_mtime and row["size"] == stat.st_size:
                continue

            # mtime/size differ → hash to confirm the CONTENT changed
            # (a plain `touch` shouldn't trigger a full re-embed).
            content_hash = file_hash(path)
            if row and row["content_hash"] == content_hash:
                manifest.upsert(
                    conn, path_str, mtime=stat.st_mtime, size=stat.st_size,
                    content_hash=content_hash, status=row["status"],
                    page_count=row["page_count"] or 0,
                    chunk_count=row["chunk_count"] or 0)
                conn.commit()
                continue

            status, chunks = index_file(conn, cfg, path, stat, content_hash)
            counts["files_indexed" if status == "indexed" else "files_skipped"] += 1
            counts["chunks_added"] += chunks
            conn.commit()

            changed_processed += 1
            if limit and changed_processed >= limit:
                log.info("--limit %d reached, stopping early", limit)
                break

        except Exception as exc:  # one bad PDF must not kill the whole run
            log.error("FAILED: %s (%s)", path, exc)
            counts["files_failed"] += 1
            try:
                manifest.upsert(conn, path_str, mtime=0, size=0,
                                content_hash="", status="failed",
                                error=str(exc)[:500])
                conn.commit()
            except Exception:
                pass  # even the bookkeeping failed; carry on regardless

    # Purge index entries for files that no longer exist on disk.
    # (Skipped when --limit is set: a partial scan must not delete anything.)
    if not limit:
        on_disk = {str(p) for p in pdfs}
        for known_path in manifest.all_paths(conn):
            if known_path not in on_disk:
                log.info("PURGE (file deleted): %s", known_path)
                store.delete_file_chunks(cfg, known_path)
                manifest.delete(conn, known_path)
                counts["files_deleted"] += 1
        conn.commit()

    summary = {"started": started,
               "finished": datetime.now().isoformat(timespec="seconds"),
               **counts}
    manifest.record_run(conn, summary)
    conn.commit()
    conn.close()
    log.info("Done in %.1fs — %s", time.time() - t0,
             ", ".join(f"{k}={v}" for k, v in counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after N changed files (for testing)")
    parser.add_argument("--retag", action="store_true",
                        help="only re-stamp doc_type payloads (facet), no indexing")
    args = parser.parse_args()

    # Log to both the console and data/index.log.
    LOG_PATH.parent.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(LOG_PATH, encoding="utf-8")],
    )
    # Keep third-party HTTP/download chatter out of the index log.
    for noisy in ("httpx", "urllib3", "sentence_transformers", "fastembed"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if not acquire_lock():
        log.error("Another indexer is already running (lock: %s)", LOCK_PATH)
        sys.exit(1)
    try:
        if args.retag:
            retag_doc_types()
        else:
            run(limit=args.limit)
    finally:
        release_lock()


if __name__ == "__main__":
    main()
