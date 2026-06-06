# spotAIfy

A local AI agent that generates Spotify playlists using Claude Sonnet. Runs as a web app (Go + Python) or as a CLI tool.

Six playlist modes:

| Mode | How it works |
|------|-------------|
| **Sonic** | Describe a mood or vibe — Claude searches the catalog and ranks by audio similarity |
| **Connection** | Explore an artist's network (band members, side projects, collabs) via MusicBrainz |
| **DNA** | Map a track's creative lineage — what it samples, what samples it, any melody interpolations |
| **Rediscovery** | Build a playlist from tracks you used to love but haven't played in a while |
| **Expand** | Discover new music by traversing your top artists' connection graphs and Spotify's recommendation engine |
| **Setlist** | Build a playlist from an artist's actual live repertoire, ranked by how often they play each song |

---

## How Agent Scoring Works

The `rank_and_select` tool scores each candidate track using a combination of sonic and personal signals:

| Signal | Weight | Source |
|--------|--------|--------|
| `audio_sim` | 1.0× base | Set by the agent based on sonic match to the prompt |
| `completion_rate` | +0.3× | Fraction of your plays where the track ran to the end |
| `skip_rate` | −0.2× | Fraction of plays where you skipped it |
| `is_new_discovery` | +0.15 flat | True when the artist never appears in your listening history |

`completion_rate`, `skip_rate`, and `is_new_discovery` are populated automatically from your Spotify Extended Streaming History export. Without the export, they default to neutral (0.0 / False) and the agent scores purely on audio similarity.

---

## Taste Profile

The `get_taste_profile` agent tool (also invoked automatically by Rediscovery mode) computes two signals from your listening history:

- **Audio fingerprint** — mean energy, valence, tempo, danceability, etc. across your top 100 played tracks
- **Era distribution** — fraction of your top tracks by release decade, e.g. `{"1980s": 0.12, "1990s": 0.31, "2000s": 0.28, ...}`

Results are cached to `data/cache/taste_profile.json`. Force a rebuild by calling the tool with `rebuild: true`.

---

## Architecture

```
Browser  ──POST /run──►  Go web server (cmd/web/main.go)
                               │
                               │  spawns subprocess
                               ▼
                         Python agent (spotaify/agent.py)
                               │
              ┌────────────────┼───────────────────┐
              ▼                ▼                   ▼
       Anthropic API     Spotify API         External data
       (Claude Sonnet)   search, features,   MusicBrainz · WhoSampled
                         playlists           Genius · history export
```

The Go server handles auth, session management, and SSE streaming of agent output back to the browser in real time. The Python agent runs as a subprocess and communicates via stdout.

---

## Setup

### 1. Credentials

- **Spotify**: Create an app at the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard). Note your Client ID and Client Secret. Add these redirect URIs:
  - Web app: `http://localhost:8000/spotify/callback`
  - CLI: `http://127.0.0.1:9090`

- **Anthropic**: Create an API key at the [Anthropic Console](https://console.anthropic.com).

- **Genius** (optional, improves DNA mode): Create an app at [genius.com/api-clients](https://genius.com/api-clients). Note your Access Token.

- **MusicBrainz**: No account needed, but set `MUSICBRAINZ_CONTACT` in your `.env` to your real email address — their [rate limit policy](https://musicbrainz.org/doc/MusicBrainz_API/Rate_Limiting) requires it.

- **WhoSampled** (optional, DNA mode): A free account improves sample data quality for the DNA mode scraper.

- **setlist.fm** (optional, Setlist mode): Free API key at [api.setlist.fm](https://api.setlist.fm/docs/1.0/index.html). Register an account and request an API key — approval is usually instant.

### 2. Configure environment

```bash
cp .env.example .env   # Mac/Linux
copy .env.example .env  # Windows
```

Fill in at minimum:
```
SPOTIFY_CLIENT_ID=...
MUSICBRAINZ_CONTACT=your@email.com
```

**Recommended — store secrets in the OS keychain** (Windows Credential Manager / macOS Keychain):
```bash
uv run python setup_secrets.py
```
This prompts for each secret and stores them encrypted — nothing written to disk in plain text.

**Alternative — plain `.env`** (simpler, less secure):
```
SPOTIFY_CLIENT_SECRET=...
ANTHROPIC_API_KEY=...
GENIUS_ACCESS_TOKEN=...
SETLISTFM_API_KEY=...        # optional, required for Setlist mode
```
Both approaches work; the app checks the keychain first and falls back to `.env`.

### 3. Install uv (one-time)

[uv](https://docs.astral.sh/uv/) manages Python and all dependencies automatically.

**Windows:**
```powershell
powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
```
**Mac/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 4. Spotify Data Export (optional, for personalized ranking)

Enables completion rate, skip rate, and new-discovery signals. Supports per-user history when running the web app with multiple accounts.

1. Go to Spotify → Account → Privacy Settings → Download your data
2. Request **Extended streaming history** (takes a few days to prepare)
3. When ready, copy all `Streaming_History_Audio_*.json` files into `data/history/<username>/`
   - Example for user `brian`: `data/history/brian/Streaming_History_Audio_2024.json`
   - For CLI use, files can also go directly in `data/history/` (flat layout)

The history is loaded once per session on first use. The web app injects the logged-in username automatically so each user's history is isolated.

---

## Usage

### Web App

```bash
# Windows
run.bat web

# Mac/Linux
./run.sh web
```

Open http://localhost:8000. Create an account (credentials set via the `USERS` env var), connect your Spotify, and generate playlists from the browser. All four modes are available from the UI.

### CLI

```bash
# Sonic — describe the sound you want
run.bat "dark ambient for late-night coding" --count 25
run.bat "upbeat funk" --energy 0.7-1.0 --valence 0.6-1.0

# Connection — map an artist's network
run.bat --artist "Trent Reznor" --connections --new-only
run.bat --artist "Nine Inch Nails" --connections --match-sound --depth 2

# DNA — deep-dive a track's creative lineage
run.bat --dna "Hurt" --dna-artist "Nine Inch Nails"
run.bat --dna "Straight Outta Compton" --dna-artist "N.W.A"

# Rediscovery — forgotten favorites
run.bat --rediscovery --stale-days 120 --min-plays 5 --count 20

# Expand — discover new music from your history
run.bat --expand --count 20

# Setlist — live repertoire for an artist
run.bat --setlist --setlist-artist "Radiohead" --setlist-pages 3
```

Direct invocation with uv:
```bash
uv run python -m spotaify.agent "chill lo-fi hip-hop"
uv run python -m spotaify.agent --dna "Play Your Part Pt. 1" --dna-artist "Janelle Monáe"
```

**CLI authorize Spotify (one-time, CLI only):**
```bash
uv run python -c "from spotaify.clients.spotify_client import SpotifyClient; SpotifyClient(user_auth=True)"
```
Approve in the browser. The token is saved to `.spotify_token_cache` and future runs authenticate silently.

### Filters (Sonic and Connection modes)

| Flag | Description | Example |
|------|-------------|---------|
| `--count N` | Tracks in the final playlist (default 20) | `--count 30` |
| `--energy MIN-MAX` | Audio energy 0.0–1.0 | `--energy 0.7-1.0` |
| `--valence MIN-MAX` | Emotional positivity 0.0–1.0 | `--valence 0.3-0.6` |
| `--tempo MIN-MAX` | BPM range | `--tempo 120-140` |

---

## DNA Mode — How It Works

DNA mode builds a playlist with a **literal musical relationship** to the seed track:

- Songs the seed **directly samples** (source recordings resolved on Spotify)
- Songs that **directly sampled** the seed
- The original recording of any **confirmed melody interpolation** surfaced by Genius or MusicBrainz

The agent calls WhoSampled once for the seed track, resolves those exact recordings on Spotify, and stops. It does not recurse into samples of samples, does not include songs by the same producers, and does not include thematic cousins. Each included track gets an explicit narration: *"Firestarter samples the break from X"* or *"X was later sampled by Firestarter"*.

Playlist descriptions are ≤ 300 characters (Spotify's limit); full narration appears in the terminal or web output.

---

## MusicBrainz Concepts

Connection and DNA modes both rely on MusicBrainz data:

- **Artist**: A person or group. MusicBrainz tracks artist-to-artist relationships (member of, collaborator, etc.). Connection mode traverses this graph.
- **Recording**: A specific performance — the studio version of "Hurt" and a 1995 live recording are separate recordings. DNA mode analyzes recording-level credits.
- **Work**: The abstract composition. Many recordings can share the same work (studio, live, covers).

The code includes `time.sleep(1)` between MusicBrainz requests to stay within their one-request-per-second rate limit. This is intentional — removing it will get your IP throttled.

If the seed artist in Connection mode cannot be found on MusicBrainz, the agent stops with a clear error. DNA mode is more resilient — missing individual credits are noted and skipped rather than failing the whole run.

---

## Project Structure

```
spotAIfy/
├── cmd/web/main.go            # Go web server — auth, SSE streaming, Spotify OAuth
├── spotaify/
│   ├── agent.py               # Claude agent loop and CLI entrypoint
│   ├── agent_tools.py         # All 16 tools available to the agent
│   ├── config.py              # Credentials from .env / OS keychain
│   ├── clients/
│   │   ├── spotify_client.py  # Spotify API — search, recommendations, playlists
│   │   ├── genius_client.py   # Genius — track descriptions and credits (disk-cached)
│   │   ├── whosampled_client.py # WhoSampled scraper — sample relationships
│   │   └── setlistfm_client.py  # setlist.fm — live setlist data ranked by frequency
│   └── core/
│       ├── artist_graph.py    # MusicBrainz artist connection graph (BFS, disk-cached)
│       ├── history_profile.py # Parses Spotify history export → per-track listening signals
│       ├── taste_profile.py   # Audio fingerprint and era distribution (disk-cached)
│       ├── track_dna.py       # MusicBrainz recording-level credits for DNA mode
│       └── tf_scorer.py       # Keras scoring model (learning track — not yet wired in)
├── static/style.css           # UI styles — dark/light theme, dual-range sliders
├── templates/
│   ├── base.html              # Layout shell
│   ├── index.html             # Main playlist builder (six modes, vertical tab nav)
│   ├── profile.html           # Per-user listening stats, persona, recommendations
│   └── login.html             # Login page
├── tests/                     # Pytest suite
├── data/
│   ├── graph/                 # Cached MusicBrainz graphs (auto-created)
│   ├── history/               # Place Spotify Extended Streaming History files here
│   └── cache/                 # Genius and taste-profile caches (auto-created)
├── run.bat                    # Windows launcher
├── run.sh                     # Mac/Linux launcher
├── pyproject.toml             # Project metadata and dependencies
└── .env.example               # Environment variable template
```

---

## Developer Notes

### Lazy-loaded clients

`SpotifyClient` and other API clients are initialized on first use, not at import time. This prevents `OauthError` during `pytest` collection and lets test fixtures mock the client before it is ever constructed.

### Caching layers

| Layer | What | Location |
|-------|------|----------|
| MusicBrainz graphs | Per-artist JSON files | `data/graph/` |
| Genius track info | Per-track JSON (MD5-keyed) | `data/cache/genius/` |
| Taste profile | Audio fingerprint + era distribution | `data/cache/taste_profile.json` |
| History profile | In-memory, loaded once per session | `data/history/<username>/Streaming_History_Audio_*.json` |
| Agent tool results | In-memory per session | `_TOOL_CACHE` in `agent_tools.py` |

### Agent turn budget

The agent has a 12-turn limit (`_MAX_TURNS` in `agent.py`). At turn 9 a wrap-up nudge is injected to ensure `rank_and_select` and `create_spotify_playlist` are called before the budget runs out. Connection mode is the most expensive — keep `--depth` at 1 or 2.

### Spotify audio features deprecation

Spotify deprecated the `/v1/audio-features` endpoint for non-partner apps (returns 403). The app handles this gracefully: `fetch_audio_features` returns `{}` on 403, taste profile computation skips the fingerprint section but still computes era distribution, and any tool error returns `{"error": "..."}` to the agent rather than crashing the run.

### setlist.fm rate limits

setlist.fm does not publish rate limits. The client sleeps 0.5 s between requests. Scanning 5 pages hits the API 6–7 times (artist lookup + pages). Stay at 3–5 pages to avoid throttling.

### Running tests

```bash
uv run --with pytest --with pytest-mock pytest tests/
```
