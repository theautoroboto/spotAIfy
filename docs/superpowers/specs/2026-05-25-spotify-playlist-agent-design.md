# Spotify AI Playlist Agent — Design Spec
**Date:** 2026-05-25  
**Branch:** spotify  
**Status:** Approved

---

## Problem

Spotify's playlist and radio algorithms fail in two distinct ways:

1. **Genre-rigid sonic matching** — a track's genre label determines its neighbors, so "focused instrumental metal" and "focused jazz fusion" never mix even when their audio profiles are nearly identical.
2. **No human connection awareness** — Spotify has no concept of "everyone associated with Nine Inch Nails." It can't traverse the web of band members, side projects, and collaborations that connect Trent Reznor → How to Destroy Angels, Robin Finck → Guns N' Roses, Ilan Rubin → The New Regime.

The goal is a local AI agent that builds playlists two ways: by audio feature similarity across the full catalog (ignoring genre labels), and by human artist connection graphs (traversing real-world membership and collaboration relationships) — exposing you to artists you may not have known existed.

---

## Architecture Overview

Five layers, each with a single responsibility:

```
┌─────────────────────────────────────────────┐
│  Agent (Claude)                             │
│  • Interprets prompt / seeds / parameters   │
│  • Calls tools: search, embed, rank,        │
│    graph traversal, connection narration    │
│  • Assembles playlist, narrates reasoning   │
└─────────────┬───────────────────────────────┘
              │
┌─────────────▼───────────────────────────────┐
│  FiftyOne Dataset ("spotify_explorer")      │
│  • Persistent track cache                   │
│  • Audio feature embeddings + UMAP          │
│  • Personal play history signals            │
│  • Connection path metadata per track       │
│  • Visual browser for playlist review       │
└─────────────┬───────────────────────────────┘
              │
┌─────────────▼───────────────────────────────┐
│  Artist Connection Graph                    │
│  • MusicBrainz relationship data            │
│  • Nodes: artists  Edges: relationships     │
│  • Types: member, collaboration, supergroup │
│  • Cached locally (JSON) after first fetch  │
└─────────────┬───────────────────────────────┘
              │
┌─────────────▼───────────────────────────────┐
│  Spotify Client                             │
│  • Catalog search + audio features          │
│  • Liked songs, playlists, recent plays     │
│  • Full history export import (JSON)        │
│  • Push approved playlists                  │
└─────────────┬───────────────────────────────┘
              │
┌─────────────▼───────────────────────────────┐
│  Embeddings Engine                          │
│  • 13D normalized audio feature vector      │
│  • sentence-transformers text embedding     │
│    on "title artist genres"                 │
│  • 2D UMAP projection for FiftyOne          │
└─────────────────────────────────────────────┘
```

The agent never writes directly to Spotify. It proposes a playlist, opens FiftyOne for visual/audio review, and only pushes when the user runs `push.py`.

---

## FiftyOne Dataset Schema

Dataset name: `spotify_explorer`

Each sample represents one unique track:

```
filepath            → spotify:track:<id>  (Spotify URI, used as unique key)

# Identity
track_id            → Spotify track ID
title               → track name
artist              → primary artist name
album               → album name
duration_ms         → track length in milliseconds
release_year        → integer year from album release_date
explicit            → boolean
genres              → list[str] — from artist object
preview_url         → 30-second MP3 URL (plays in FiftyOne browser)

# Audio features (Spotify Audio Features API)
tempo               → BPM (float)
energy              → 0.0–1.0
valence             → 0.0–1.0  (happy=1.0, sad=0.0)
danceability        → 0.0–1.0
acousticness        → 0.0–1.0
instrumentalness    → 0.0–1.0
speechiness         → 0.0–1.0
liveness            → 0.0–1.0  (live recording detection)
loudness            → dB (float, typically -60 to 0)
key                 → int 0–11 (pitch class)
mode                → 0=minor, 1=major
time_signature      → int (beats per measure: 3, 4, 5, 6, 7)

# Personal signals (API + history export)
play_count          → int — total times played
total_ms_played     → int — total milliseconds listened
completion_rate     → float — avg (ms_played / duration_ms) across all plays
skip_rate           → float — fraction of plays ended via skip (fwdbtn / reason_end)
first_played_at     → datetime — earliest play timestamp (history export)
last_played_at      → datetime — most recent play timestamp
source              → str: "liked" | "playlist" | "history" | "catalog_search"
is_new_discovery    → bool — True if never appeared in personal history

# Artist connection metadata (populated during connection-mode queries)
artist_mbid         → str — MusicBrainz artist ID (for graph lookups)
connection_seed     → str — the seed artist this track was reached from
connection_path     → str — human-readable path e.g. "Nine Inch Nails → Trent Reznor → How to Destroy Angels"
connection_type     → str — "member" | "collaboration" | "supergroup" | "solo"
connection_depth    → int — hops from seed artist (1=direct, 2=one remove, etc.)

# Bateman mode metadata (populated during --bateman deep-dive queries)
recording_mbid      → str — MusicBrainz recording ID
composers           → list[str] — songwriter/composer names
lyricists           → list[str] — lyricist names
producers           → list[str] — producer names
session_musicians   → list[dict] — [{name, role}] e.g. {name: "Bootsy Collins", role: "bass"}
samples             → list[dict] — [{title, artist, recording_mbid}] recordings sampled
bateman_seed        → str — the seed track this was reached from in bateman mode
bateman_path        → str — e.g. "Fight the Power → samples Funky Drummer → James Brown"
bateman_link_type   → str — "composer" | "producer" | "session_musician" | "sample" | "sampled_by"

# Embeddings (computed locally)
audio_embedding     → list[float] — 13D normalized audio feature vector
text_embedding      → list[float] — sentence-transformer on "title artist genres"
embedding           → list[float] — 2D UMAP projection for visual browser
```

**Key insight:** `audio_embedding` encodes the 13 Spotify audio features as a normalized vector. Similarity search on this embedding finds tracks with matching sonic profiles regardless of genre — this is the core mechanism that breaks genre rigidity.

**Discovery insight:** `is_new_discovery` combined with `connection_path` lets the agent narrate introductions: "This is a new artist for you — How to Destroy Angels is a collaborative project of Trent Reznor from Nine Inch Nails with Atticus Ross and Mariqueen Maandig."

---

## Agent Tools

The agent (`agent.py`) is Claude-powered and has access to twelve tools across two modes:

**Sonic similarity tools:**

| Tool | Purpose |
|------|---------|
| `search_catalog` | Search Spotify catalog by keywords, return track IDs + metadata |
| `fetch_audio_features` | Get Spotify audio features for a batch of track IDs |
| `embed_tracks` | Compute audio + text embeddings for candidate tracks |
| `query_dataset` | Embedding similarity search against FiftyOne cache |
| `filter_by_features` | Hard filter on audio feature ranges (tempo, energy, valence, etc.) |
| `get_listening_history` | Pull personal signals from FiftyOne (completion_rate, play_count) |
| `rank_and_select` | Score + sort candidates, select final N tracks, generate reasoning |
| `open_in_fiftyone` | Load proposed playlist into FiftyOne browser for review |

**Artist connection tools:**

| Tool | Purpose |
|------|---------|
| `map_artist_connections` | Query MusicBrainz for all relationships of a seed artist (members, projects, collabs) |
| `traverse_artist_graph` | Walk the connection graph to a given depth, return all reachable artists + paths |
| `fetch_artist_top_tracks` | Get representative tracks from each artist in the constellation |
| `explain_connection` | Generate a human-readable narration of how a track connects back to the seed |

**Bateman mode tools:**

| Tool | Purpose |
|------|---------|
| `deep_dive_track` | Resolve a track to MusicBrainz recording; fetch all credits (composers, producers, session musicians, samples) |
| `find_tracks_by_person` | Given a person's name (writer, producer), find other recordings they worked on via MusicBrainz |
| `resolve_samples` | For each sampled recording, get the original track + its artist + related catalog |

**Typical agent flow for "everyone associated with Nine Inch Nails":**

```
1. map_artist_connections("Nine Inch Nails")
   → members: Trent Reznor, Robin Finck, Alessandro Cortini, Ilan Rubin
   → past members: Richard Patrick, Danny Lohner

2. traverse_artist_graph(seed="Nine Inch Nails", depth=2)
   → depth 1: How to Destroy Angels (Reznor)
              Filter (Richard Patrick)
   → depth 2: Guns N' Roses (Robin Finck)
              The New Regime (Ilan Rubin)
              Atticus Ross (How to Destroy Angels)

3. fetch_artist_top_tracks([all artists in constellation])
   → top 3-5 tracks per artist

4. fetch_audio_features([all candidate tracks])
5. embed_tracks(candidates)
6. get_listening_history()    → flag new discoveries vs. familiar tracks
7. rank_and_select(count=30)  → balanced across artists, not top-heavy
8. open_in_fiftyone(playlist) → color-coded by connection_depth + connection_path
```

**Typical agent flow for "energetic jazz-meets-metal focus music":**

```
1. search_catalog("jazz fusion energetic instrumental")  → 50 candidates
2. search_catalog("progressive metal instrumental")       → 50 candidates
3. fetch_audio_features([100 track IDs])
4. embed_tracks(candidates)
5. query_dataset(embedding)    → similar tracks already in local cache
6. filter_by_features(energy > 0.7, instrumentalness > 0.4)
7. get_listening_history()     → boost tracks with high completion_rate
8. rank_and_select(count=20)   → final playlist + per-track reasoning
9. open_in_fiftyone(playlist)  → visual browser opens for review
```

The agent narrates its reasoning in both modes: which cross-genre cluster or connection path each track came from, its audio feature profile, and a plain-English introduction for new discoveries ("You may not have heard How to Destroy Angels — it's a collaborative project formed by Trent Reznor of Nine Inch Nails with Atticus Ross and Mariqueen Maandig").

---

## Invocation Modes

The agent supports two primary modes — **Sonic** (audio feature matching) and **Connection** (artist relationship graph) — with all styles available in a single entrypoint:

### Sonic Mode

```bash
# Natural language prompt
python agent.py "energetic focus music crossing jazz and metal, 20 tracks"

# Prompt + structured audio feature constraints
python agent.py --energy 0.7-1.0 --valence 0.3-0.6 --tempo 120-180 "driving late night"

# Single seed track — expand outward by audio similarity
python -m spotaify.agent --seed "spotify:track:6eUW0wxWtzkFdaEFsTJto6" --count 25

# Multi-seed mood board — find what the seeds have in common sonically
python agent.py --seed "Meshuggah - Bleed" "Esbjörn Svensson Trio - Seven Days of Falling" --count 20
```

### Connection Mode

```bash
# Full artist constellation — everyone associated with a seed artist
python agent.py --artist "Nine Inch Nails" --connections

# Limit traversal depth (1=direct members/collabs only, 2=one remove)
python agent.py --artist "Nine Inch Nails" --connections --depth 1

# Constellation filtered by audio similarity to the seed (sounds like + connected to)
python agent.py --artist "Nine Inch Nails" --connections --match-sound

# Show how two artists are connected before building a playlist bridging them
python agent.py --connect "Nine Inch Nails" "How to Destroy Angels"

# Discovery mode — only include artists not in your listening history
python agent.py --artist "Nine Inch Nails" --connections --new-only
```

### Combined Mode

```bash
# Connection constellation + sonic filter: Nine Inch Nails associates that are heavy + industrial
python agent.py --artist "Nine Inch Nails" --connections --energy 0.8-1.0 --instrumentalness 0.5-1.0

# Seed track + connection graph: expand from a specific song AND its artist's network
python agent.py --seed "Nine Inch Nails - Hurt" --connections --count 30
```

### Bateman Mode

Named for Patrick Bateman's encyclopedic, obsessive track-level music analysis. Given a single song, the agent digs into its complete creative DNA — songwriters, producers, session musicians, and samples — then traces every person and recording outward to build a playlist connected by craft, not just sound or band membership.

```bash
# Deep dive a single track — follow every writer, producer, and sample outward
python agent.py --bateman "Nine Inch Nails - Hurt"
python agent.py --bateman "spotify:track:6eUW0wxWtzkFdaEFsTJto6"

# Bateman + depth: how far to follow each human connection
python agent.py --bateman "Kanye West - POWER" --depth 2

# Bateman + count
python agent.py --bateman "Public Enemy - Fight the Power" --count 40
```

**What Bateman mode resolves:**
- **Composers / lyricists** — who wrote the song; what else they wrote; what artists recorded their songs
- **Producers** — who produced it; what other albums/artists they produced
- **Session musicians / featured artists** — who played on it; what bands they're in; what else they've recorded
- **Samples** — what recordings were sampled; who made those originals; what else those artists made

**Example: "Fight the Power" by Public Enemy**
```
Samples "Funky Drummer" by James Brown → find all James Brown tracks + artists who sampled him
Samples "The Grunt" by JBs → Fred Wesley → New Power Generation (Prince)
Produced by The Bomb Squad (Hank Shocklee) → other Bomb Squad productions → Ice Cube, LL Cool J
Written by Chuck D + Flavor Flav → Chuck D solo → Public Enemy discography
Features Flavor Flav → Flav solo work
→ Playlist spans: James Brown, JBs, Prince, Ice Cube, LL Cool J, Chuck D solo, PE discography
```

Seed tracks are resolved by search if given as "Artist - Title" strings, or used directly if given as Spotify URIs. Artist names are resolved via MusicBrainz search; ambiguous names prompt for disambiguation.

---

## Artist Connection Graph

### Data Source: MusicBrainz

MusicBrainz is a free, open music encyclopedia with explicit relationship data between artists. Unlike Spotify's "related artists" (which is sonic similarity), MusicBrainz records actual human relationships: who was a member of what band, when, and in what role.

Accessed via the `musicbrainzngs` Python library. Rate-limited to 1 req/sec; responses are cached locally.

### Graph Structure

```
Nodes: artists
  artist_mbid     → MusicBrainz artist ID (stable, unique)
  artist_name     → canonical name
  spotify_id      → Spotify artist ID (resolved separately)
  known_to_user   → bool — appears in personal listening history

Edges: relationships
  from_artist     → artist_mbid
  to_artist       → artist_mbid
  relation_type   → "member of" | "past member" | "collaboration" |
                    "supergroup" | "solo" | "tribute" | "founded by"
  begin_date      → year membership/collaboration started
  end_date        → year ended (None = ongoing)
  attributes      → role details e.g. "vocals", "guitar", "drums"
```

### Example Graph: Nine Inch Nails Constellation (depth 2)

```
Nine Inch Nails
  ├── [member] Trent Reznor
  │     ├── [project] How to Destroy Angels
  │     │     └── [member] Atticus Ross → [film score collaborator]
  │     └── [solo] Trent Reznor (film scores, solo)
  ├── [member] Robin Finck → [member of] Guns N' Roses
  ├── [member] Alessandro Cortini
  └── [member] Ilan Rubin → [project] The New Regime
```

Depth 1 = How to Destroy Angels, Filter (Richard Patrick), solo Reznor  
Depth 2 = Guns N' Roses (Robin Finck), The New Regime (Ilan Rubin), Atticus Ross

### Graph Cache

The graph is serialized to `data/graph/` after the first MusicBrainz fetch. Subsequent runs load from cache. Cache is invalidated per-artist on demand (`--refresh-graph`).

---

## Data Pipeline

### index.py — build and update the FiftyOne dataset

```bash
python -m spotaify.core.index --liked          # index your Spotify liked songs
python -m spotaify.core.index --playlists      # index all your created playlists
python -m spotaify.core.index --recent         # pull last 50 played tracks (API limit)
python -m spotaify.core.index --history        # import full Spotify data export (JSON files)
python -m spotaify.core.index --resume         # skip tracks already in the dataset
```

The `--history` flag reads all JSON files from the `data/history/` directory (drop your Spotify export files there). It computes `first_played_at`, `last_played_at`, `play_count`, `total_ms_played`, `completion_rate`, and `skip_rate` per track, then upserts into FiftyOne.

### push.py — send approved playlist to Spotify

```bash
python -m spotaify.core.push --name "Focus: Jazz × Metal" --public false
```

Reads the current FiftyOne session's tagged playlist samples and creates the playlist in the user's Spotify account via the Web API.

---

## Project Structure

```
spotAIfy/
├── spotaify/
│   ├── agent.py                   # AI playlist builder — main entrypoint
│   ├── agent_tools.py             # All tool definitions + executor for Claude API
│   ├── config.py                  # credentials, dataset name, defaults
│   ├── clients/
│   │   ├── genius_client.py       # Genius API client (track descriptions, credits)
│   │   ├── jellyfin_client.py     # Jellyfin media server client
│   │   ├── spotify_client.py      # Spotify API wrapper (OAuth + client_credentials)
│   │   └── whosampled_client.py   # WhoSampled scraper (sample relationships)
│   └── core/
│       ├── artist_graph.py        # MusicBrainz client + artist graph traversal + cache
│       ├── color_mood.py          # Color/mood analysis utilities
│       ├── embeddings.py          # audio feature vectors + sentence-transformers + UMAP
│       ├── frame_extractor.py     # Video frame extraction utilities
│       ├── history_importer.py    # parses Spotify data export JSON files
│       ├── index.py               # data pipeline — run first
│       ├── push.py                # pushes approved playlist to Spotify
│       └── track_dna.py           # MusicBrainz recording credits + sample resolution (Bateman mode)
├── tests/                         # unit and integration tests
├── data/
│   ├── history/                   # drop Spotify export ZIP contents here
│   └── graph/                     # auto-created — cached MusicBrainz graph JSON files
├── docs/
├── requirements.txt
└── .env                           # SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, etc.
```

---

## Authentication

Two OAuth flows, both handled by `spotaify/clients/spotify_client.py`:

- **Client Credentials** (no user login): catalog search, audio features. Used by `agent.py` for catalog-only queries.
- **Authorization Code** (browser login, one-time): liked songs, playlists, recently played, push playlists. Token cached locally after first login. Required for `index.py --liked/--playlists/--recent` and `push.py`.

---

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `spotipy` | Spotify Web API client + OAuth handling |
| `fiftyone` | Dataset storage, visual browser, embedding visualization |
| `fiftyone-brain` | UMAP computation, similarity index |
| `sentence-transformers` | Text embedding for "title artist genres" |
| `scikit-learn` | Audio feature vector normalization |
| `numpy` | Vector math |
| `musicbrainzngs` | MusicBrainz API client for artist relationship data |
| `anthropic` | Claude API for the agent |
| `python-dotenv` | `.env` loading |

---

## What Makes This Better Than Spotify

1. **Cross-genre by audio profile** — embeddings on 13 audio features group tracks by sonic character, not genre label. Metal and jazz that share energy/instrumentalness/tempo profiles become neighbors.
2. **Human connection awareness** — MusicBrainz relationship traversal finds every band, supergroup, collaboration, and side project connected to a seed artist. Spotify has no equivalent.
3. **Discovery narration** — the agent introduces new artists with their connection path explained: "How to Destroy Angels is a collaborative project of Trent Reznor (Nine Inch Nails) + Atticus Ross and Mariqueen Maandig." Spotify just plays a track with no context.
4. **Your actual preferences as signal** — `completion_rate` and `skip_rate` from full listening history weight tracks you genuinely finish over tracks you impulsively saved.
5. **Track DNA deep-dive (Bateman mode)** — traces a single song's complete creative lineage: writers, producers, session musicians, samples. Builds a playlist from every human thread, narrating exactly why each track is there ("sampled by", "produced by", "written by").
6. **Transparent reasoning** — the agent explains every track choice: connection path, audio feature profile, whether it's a new discovery. Spotify's algorithm is a black box.
6. **Full history depth** — once the Spotify data export arrives, `first_played_at` and `completion_rate` span your entire listening history, not just recent plays.
7. **Visual review before commit** — FiftyOne shows the playlist as an interactive embedding map, color-coded by connection depth or audio cluster, with 30-second audio previews, before anything touches your Spotify account.
