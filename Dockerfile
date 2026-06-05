# ── Stage 1: build the Go web server ─────────────────────────────────────────
FROM golang:1.22-alpine AS go-builder
WORKDIR /build
COPY go.mod go.sum ./
RUN go mod download
COPY cmd/ ./cmd/
COPY internal/ ./internal/
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o spotaify-web ./cmd/web

# ── Stage 2: Python + uv environment ─────────────────────────────────────────
FROM python:3.12-slim AS py-builder
WORKDIR /app
RUN pip install uv --no-cache-dir
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-cache --no-install-project

# ── Stage 3: final image ──────────────────────────────────────────────────────
FROM python:3.12-slim
WORKDIR /app

# Go binary
COPY --from=go-builder /build/spotaify-web /usr/local/bin/spotaify-web

# Python venv
COPY --from=py-builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Application source
COPY spotaify/ ./spotaify/
COPY templates/ ./templates/
COPY static/ ./static/

# Data directory for SQLite DB and MusicBrainz graph cache
RUN mkdir -p /data/graph
VOLUME ["/data"]

# Non-root user
RUN useradd -r -u 1001 -g root spotaify && chown -R spotaify /app /data
USER spotaify

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python3 -c "import urllib.request, os; urllib.request.urlopen('http://localhost:' + os.environ.get('PORT', '8000') + '/login')" || exit 1

ENV PORT=8000 \
    DB_PATH=/data/spotaify.db \
    GRAPH_DIR=/data/graph \
    HISTORY_DIR=/data/history \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["spotaify-web"]
