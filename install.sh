#!/usr/bin/env bash
# install.sh — set up Real Estate Agent v2 on macOS.
# Mirrors the installation steps documented in writeup.md §7.

set -euo pipefail

cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# 1. Prerequisites
# ---------------------------------------------------------------------------
command -v uv >/dev/null 2>&1 || {
  echo "error: uv is not installed."
  echo "       install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
}

command -v brew >/dev/null 2>&1 || {
  echo "error: Homebrew is not installed."
  echo "       install from: https://brew.sh"
  exit 1
}

# ---------------------------------------------------------------------------
# 2. Python dependencies
# ---------------------------------------------------------------------------
echo "==> Syncing Python dependencies with uv"
uv sync

# ---------------------------------------------------------------------------
# 3. Configure
# ---------------------------------------------------------------------------
if [ ! -f .env ]; then
  echo "==> Creating .env from .env.example"
  cp .env.example .env
  echo "    edit .env and set GOOGLE_MAPS_API_KEY before running"
else
  echo "==> .env already exists, leaving it alone"
fi

# ---------------------------------------------------------------------------
# 4. Local vision model (Ollama + gemma4:31b)
# ---------------------------------------------------------------------------
if ! command -v ollama >/dev/null 2>&1; then
  echo "==> Installing ollama via Homebrew"
  brew install ollama
else
  echo "==> ollama already installed"
fi

if ! pgrep -x ollama >/dev/null 2>&1; then
  echo "==> Starting ollama serve in the background"
  ollama serve >/dev/null 2>&1 &
  sleep 2
else
  echo "==> ollama serve already running"
fi

echo "==> Pulling gemma4:31b (large download, ~19 GB — first time only)"
ollama pull gemma4:31b

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
cat <<'EOF'

Setup complete.

Next steps:
  1. Edit .env and add your GOOGLE_MAPS_API_KEY
  2. Edit config/preferences.json (locations, budget, beds, schools)
  3. Run the agent:
       uv run main.py            # full run
       uv run main.py --test     # stop after the first matched listing
EOF
