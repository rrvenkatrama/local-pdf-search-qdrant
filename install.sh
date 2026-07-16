#!/bin/zsh
# ============================================================================
# Local PDF Search (Qdrant edition) — one-shot setup. Idempotent.
#
#   1. Creates the Python venv and installs dependencies (if missing)
#   2. Downloads the native Qdrant binary (if missing)
#   3. Installs three launchd agents:
#        - Qdrant server        (always on, port 6333)
#        - search service       (always on, port 8131)
#        - daily indexer        (08:15 every day)
# ============================================================================
set -e
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
QDRANT_VERSION="v1.18.2"

echo "Project root: $PROJECT_ROOT"

# --- 1. Python environment --------------------------------------------------
if [ ! -x "$PROJECT_ROOT/venv/bin/python" ]; then
    echo "Creating venv and installing dependencies…"
    python3 -m venv "$PROJECT_ROOT/venv"
    "$PROJECT_ROOT/venv/bin/pip" install --quiet --upgrade pip
    "$PROJECT_ROOT/venv/bin/pip" install --quiet -r "$PROJECT_ROOT/requirements.txt"
else
    echo "venv already present — skipping dependency install."
fi

# --- 2. Qdrant native binary -------------------------------------------------
if [ ! -x "$PROJECT_ROOT/bin/qdrant" ]; then
    echo "Downloading Qdrant $QDRANT_VERSION (native Apple Silicon binary)…"
    mkdir -p "$PROJECT_ROOT/bin"
    curl -sL "https://github.com/qdrant/qdrant/releases/download/$QDRANT_VERSION/qdrant-aarch64-apple-darwin.tar.gz" \
        | tar xz -C "$PROJECT_ROOT/bin"
else
    echo "Qdrant binary already present."
fi

# --- 3. launchd agents --------------------------------------------------------
mkdir -p "$HOME/Library/LaunchAgents" "$PROJECT_ROOT/data"
for PLIST_NAME in com.rajesh.pdfqdrant.qdrant \
                  com.rajesh.pdfqdrant.server \
                  com.rajesh.pdfqdrant.indexer; do
    PLIST_DEST="$HOME/Library/LaunchAgents/$PLIST_NAME.plist"
    echo "Installing launchd agent → $PLIST_DEST"
    # Fill in the real project path (plist templates use __PROJECT_ROOT__).
    sed "s|__PROJECT_ROOT__|$PROJECT_ROOT|g" \
        "$PROJECT_ROOT/launchd/$PLIST_NAME.plist" > "$PLIST_DEST"
    # Reload cleanly: unload any previous version first (ignore if not loaded).
    launchctl bootout "gui/$(id -u)/$PLIST_NAME" 2>/dev/null || true
    launchctl bootstrap "gui/$(id -u)" "$PLIST_DEST"
done

echo
echo "Done. Qdrant (6333) and the search service (8131) are starting now;"
echo "daily index runs at 08:15. Verify with:"
echo "  launchctl list | grep pdfqdrant"
echo "  → http://localhost:8131/"
