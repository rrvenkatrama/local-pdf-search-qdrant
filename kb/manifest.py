"""SQLite bookkeeping: file manifest + indexer run history.

In this edition all SEARCH data (chunk text, dense vectors, sparse BM25
vectors) lives in Qdrant. SQLite only tracks crawler state:

  manifest — one row per PDF seen: mtime/size/hash for change detection,
             status, chunk counts, errors.
  runs     — one row per indexer run, so the UI can show "last indexed at
             X, N files, M chunks".
"""

import sqlite3
from datetime import datetime

from kb.config import DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS manifest (
    path         TEXT PRIMARY KEY,  -- absolute file path
    mtime        REAL,              -- file modified time at last index
    size         INTEGER,           -- file size in bytes at last index
    content_hash TEXT,              -- sha1 of file contents at last index
    status       TEXT,              -- indexed | skipped_no_text |
                                    -- skipped_encrypted | failed
    page_count   INTEGER,
    chunk_count  INTEGER,
    last_indexed TEXT,              -- ISO timestamp
    error        TEXT               -- exception message when status=failed
);

CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    started         TEXT,
    finished        TEXT,
    files_scanned   INTEGER,
    files_indexed   INTEGER,
    files_skipped   INTEGER,   -- no text layer / encrypted
    files_failed    INTEGER,
    files_deleted   INTEGER,   -- purged because the PDF disappeared
    chunks_added    INTEGER
);
"""


def connect() -> sqlite3.Connection:
    """Open (and if needed create) the database. Safe to call from anywhere."""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def get(conn: sqlite3.Connection, path: str) -> sqlite3.Row | None:
    """Manifest row for one file, or None if never seen."""
    return conn.execute("SELECT * FROM manifest WHERE path = ?", (path,)).fetchone()


def upsert(conn: sqlite3.Connection, path: str, *, mtime: float, size: int,
           content_hash: str, status: str, page_count: int = 0,
           chunk_count: int = 0, error: str = "") -> None:
    """Insert or update one file's bookkeeping row."""
    conn.execute(
        """INSERT INTO manifest
               (path, mtime, size, content_hash, status, page_count,
                chunk_count, last_indexed, error)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(path) DO UPDATE SET
               mtime=excluded.mtime, size=excluded.size,
               content_hash=excluded.content_hash, status=excluded.status,
               page_count=excluded.page_count, chunk_count=excluded.chunk_count,
               last_indexed=excluded.last_indexed, error=excluded.error""",
        (path, mtime, size, content_hash, status, page_count, chunk_count,
         datetime.now().isoformat(timespec="seconds"), error),
    )


def delete(conn: sqlite3.Connection, path: str) -> None:
    """Forget a file (its PDF was deleted from disk)."""
    conn.execute("DELETE FROM manifest WHERE path = ?", (path,))


def all_paths(conn: sqlite3.Connection) -> list[str]:
    """Every path the crawler has ever recorded (for deletion detection)."""
    return [r["path"] for r in conn.execute("SELECT path FROM manifest")]


def record_run(conn: sqlite3.Connection, summary: dict) -> None:
    """Append one row to the run history."""
    conn.execute(
        """INSERT INTO runs (started, finished, files_scanned, files_indexed,
                             files_skipped, files_failed, files_deleted,
                             chunks_added)
           VALUES (:started, :finished, :files_scanned, :files_indexed,
                   :files_skipped, :files_failed, :files_deleted,
                   :chunks_added)""",
        summary,
    )


def last_run(conn: sqlite3.Connection) -> dict | None:
    """Most recent run summary (shown in the UI footer)."""
    row = conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT 1").fetchone()
    return dict(row) if row else None


def totals(conn: sqlite3.Connection) -> dict:
    """Indexed file / chunk counts (shown in the UI footer)."""
    row = conn.execute(
        """SELECT COUNT(*), COALESCE(SUM(chunk_count), 0)
           FROM manifest WHERE status = 'indexed'"""
    ).fetchone()
    return {"files_indexed": row[0], "total_chunks": row[1]}
