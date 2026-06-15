#!/usr/bin/env bash
set -euo pipefail

# ── Check for .env ─────────────────────────────────────────────────────────────
if [[ ! -f ".env" ]]; then
    echo "ERROR: .env file not found."
    echo "  cp .env.example .env  # then edit with your API keys"
    echo ""
    echo "TIP: Store secrets securely in the OS keychain instead of .env:"
    echo "  uv run python setup_secrets.py"
    exit 1
fi

# ── Install uv if missing ──────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "uv not found. Installing uv (one-time setup)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# ── Run spotAIfy ───────────────────────────────────────────────────────────────
# uv automatically creates a venv and installs dependencies on first run.
exec uv run python -m spotaify.agent "$@"
