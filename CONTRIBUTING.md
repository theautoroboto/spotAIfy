# Contributing to spotAIfy

Thanks for your interest in improving spotAIfy! This guide covers how the
project is laid out, how to get a dev environment running, and the conventions
that keep changes easy to review.

For what the app *does* and first-time credential setup (Spotify, Anthropic,
etc.), start with the [README](README.md).

---

## Project layout

| Path | What lives there |
|------|------------------|
| `cmd/web/` | Go web server — auth, sessions, SSE streaming, profile page handlers |
| `internal/auth/` | Login, signed session cookies, rate limiting |
| `internal/spotify/` | OAuth token exchange/refresh, Spotify Web API calls (Go side) |
| `internal/history/` | Parses Spotify Extended Streaming History exports into profile stats |
| `internal/ai/` | Persona text generation and AI image prompts (banner + avatar) |
| `internal/db/` | SQLite storage: tokens, personas, run history |
| `spotaify/` | Python agent — the Claude tool-use loop and all playlist modes |
| `spotaify/clients/` | Spotify, Genius, MusicBrainz, WhoSampled API clients |
| `spotaify/core/` | History profile, artist graph, track DNA, ranking logic |
| `templates/` | Go HTML templates (`base.html` is the shell) |
| `static/` | CSS and the site icon |
| `tests/` | Python test suite (pytest) |
| `data/history/<user>/` | Per-user Spotify history exports (gitignored, mounted read-only) |

Two runtimes cooperate: the **Go server** owns HTTP, auth, and persistence;
the **Python agent** runs as a subprocess per playlist run and streams progress
over stdout. They share no memory — stdout lines and environment variables are
the entire contract, so changes to that interface need care on both sides.

## Dev environment

Prerequisites: Go 1.22+, Python 3.11+ with [uv](https://docs.astral.sh/uv/),
Docker (for the containerized deployment), and a filled-in `.env`
(copy `.env.example` if present, or see the README's Setup section).

```bash
# Python deps (incl. dev extras: pytest, pytest-mock)
uv sync --extra dev

# Run the full stack the same way production runs it
docker compose build web && docker compose up -d

# Or run the Go server directly for faster iteration
go run ./cmd/web

# Run the agent CLI without the web UI
uv run python -m spotaify.agent "dark ambient for late-night coding" --count 10
```

Note: the Docker image bakes in `spotaify/`, `templates/`, and `static/` —
after changing any of those you must `docker compose build web && docker
compose up -d --force-recreate web` to see it in the container.

## Tests

```bash
# Python
uv run --extra dev pytest tests/

# Go
go test ./...
go vet ./...
```

Please add tests alongside behavior changes:

- Pure logic (ranking, merging, parsing) belongs in fast, dependency-free
  tests — see `tests/test_everyone.py` and `internal/ai/background_test.go`
  for the style.
- Mock external services (Spotify, Anthropic); tests must pass offline.
  `tests/test_agent_tools.py` shows the mocking conventions.

## Conventions

- **Go**: standard `gofmt` formatting; errors are logged with enough context
  to find the user/run they belong to. Handlers stay thin — push logic into
  `internal/` packages.
- **Python**: keep the agent's stdout protocol intact — `-> ` for tool calls,
  `  <- ` for results, `[error] ...` for failures. The web UI and run-history
  persistence parse these prefixes.
- **Templates**: the CSP forbids inline scripts and event handlers. Put JS in
  the nonce'd `<script nonce="{{.Nonce}}">` blocks and bind events with
  `addEventListener` — an `onclick=` attribute will be silently blocked by
  the browser.
- **Secrets** stay in `.env` (or the OS keyring for the CLI) — never in code,
  templates, or test fixtures. `USERS` is mandatory (the server exits on startup
  without it); `SESSION_SECRET` defaults to an insecure placeholder with a
  warning — always set it in production.
- **User data**: anything under `data/` is real listening history and stays
  out of git. Don't commit fixtures derived from it — synthesize test data
  instead.

## Pull requests

1. Branch from `main` (`feature/<short-name>`).
2. Keep PRs focused — one feature or fix per PR.
3. Make sure `go build ./...`, `go vet ./...`, and both test suites pass.
4. If you changed UI, include a screenshot; if you changed the agent's
   behavior, paste a sample run transcript.
5. Describe *why* in the PR body, not just what — especially for prompt
   changes (image themes, persona wording), where intent is everything.

## Reporting issues

Include the playlist mode, the run's output from the History panel (it
persists the full transcript server-side), and relevant `docker logs
spotaify-web-1` lines. For Spotify 403s, note which user hit it — development
mode apps only allow allowlisted accounts.
