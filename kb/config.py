"""Configuration loading and project paths.

Everything configurable lives in config.yaml at the project root.
This module loads it once and exposes it as a simple Config object,
plus the standard locations for generated data (data/ folder).
"""

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Project root = the folder containing config.yaml (parent of this kb/ package).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# All generated artifacts live under data/ (gitignored).
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "manifest.db"    # SQLite: file manifest + run history
LOG_PATH = DATA_DIR / "index.log"     # indexer log
LOCK_PATH = DATA_DIR / "indexer.lock" # prevents two indexers running at once


@dataclass
class Config:
    """Typed view of config.yaml. See that file for per-field documentation."""

    roots: list[Path] = field(default_factory=list)
    exclude_globs: list[str] = field(default_factory=list)
    doc_type_globs: dict[str, list[str]] = field(default_factory=dict)
    dense_model: str = "BAAI/bge-m3"
    dense_dim: int = 1024
    sparse_model: str = "Qdrant/bm25"
    chunk_sentences: int = 8
    chunk_overlap_sentences: int = 4
    top_k: int = 10
    dense_candidates: int = 30
    sparse_candidates: int = 20
    qdrant_url: str = "http://127.0.0.1:6333"
    collection: str = "pdf_chunks"
    port: int = 8131

    def doc_type(self, path: str) -> str:
        """Classify a file path via doc_type_globs; unmatched → "personal"."""
        for dtype, globs in self.doc_type_globs.items():
            if any(fnmatch.fnmatch(path, g) for g in globs):
                return dtype
        return "personal"


def load_config() -> Config:
    """Read config.yaml and return a Config with expanded, absolute root paths."""
    raw = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text())
    cfg = Config(
        roots=[Path(r).expanduser().resolve() for r in raw.get("roots", [])],
        exclude_globs=raw.get("exclude_globs") or [],
        doc_type_globs={
            dtype: [str(Path(g).expanduser()) for g in globs or []]
            for dtype, globs in (raw.get("doc_type_globs") or {}).items()
        },
        dense_model=raw.get("dense_model", Config.dense_model),
        dense_dim=int(raw.get("dense_dim", Config.dense_dim)),
        sparse_model=raw.get("sparse_model", Config.sparse_model),
        chunk_sentences=int(raw.get("chunk_sentences", Config.chunk_sentences)),
        chunk_overlap_sentences=int(
            raw.get("chunk_overlap_sentences", Config.chunk_overlap_sentences)
        ),
        top_k=int(raw.get("top_k", Config.top_k)),
        dense_candidates=int(raw.get("dense_candidates", Config.dense_candidates)),
        sparse_candidates=int(raw.get("sparse_candidates", Config.sparse_candidates)),
        qdrant_url=raw.get("qdrant_url", Config.qdrant_url),
        collection=raw.get("collection", Config.collection),
        port=int(raw.get("port", Config.port)),
    )
    DATA_DIR.mkdir(exist_ok=True)
    return cfg
