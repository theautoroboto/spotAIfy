# Spotify AI Playlist Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local AI agent that generates Spotify playlists via audio feature embeddings, artist connection graphs (band membership/collaborations via MusicBrainz), and track DNA deep-dives (writers, producers, samples) — then pushes approved playlists to Spotify after visual review in FiftyOne.

**Architecture:** Claude-powered agent with 15 tools across three modes (Sonic, Connection, Bateman). FiftyOne is the persistent track cache and visual review layer. MusicBrainz provides artist relationship and recording credit data. Spotify API provides audio features, catalog search, and playlist creation.

**Tech Stack:** Python 3.11+, spotipy, fiftyone, fiftyone-brain, anthropic, musicbrainzngs, sentence-transformers, umap-learn, scikit-learn, numpy, python-dotenv, pytest

---

## File Map

| File | Responsibility |
|------|---------------|
| `spotaify/config.py` | Load env vars, constants |
| `spotaify/clients/spotify_client.py` | Spotify API: search, audio features, OAuth, user data, create playlist |
| `spotaify/clients/genius_client.py` | Fetch track descriptions and credits from Genius |
| `spotaify/clients/whosampled_client.py` | Scrape WhoSampled for sample relationships |
| `spotaify/clients/jellyfin_client.py` | Jellyfin media server client |
| `spotaify/core/history_importer.py` | Parse Spotify data export JSON → per-track play stats |
| `spotaify/core/embeddings.py` | 13D audio feature vector, sentence-transformer text embedding, UMAP |
| `spotaify/core/artist_graph.py` | MusicBrainz artist relationships: traverse, cache, explain path |
| `spotaify/core/track_dna.py` | MusicBrainz recording credits + sample resolution (Bateman mode) |
| `spotaify/core/index.py` | Pipeline: ingest liked/playlists/recent/history into FiftyOne |
| `spotaify/core/push.py` | Read FiftyOne tagged playlist → create Spotify playlist |
| `spotaify/agent_tools.py` | All tool definitions + executor for Claude API |
| `spotaify/agent.py` | Claude agent loop + CLI argument parsing |
| `tests/conftest.py` | Shared fixtures |
| `tests/test_spotify_client.py` | Unit tests with mocked spotipy |
| `tests/test_history_importer.py` | Unit tests with fixture JSON |
| `tests/test_embeddings.py` | Unit tests for vector math |
| `tests/test_artist_graph.py` | Unit tests with mocked musicbrainzngs |
| `tests/test_track_dna.py` | Unit tests with mocked musicbrainzngs |
| `tests/test_fiftyone_store.py` | Integration tests (requires FiftyOne running) |
| `tests/test_agent_tools.py` | Unit tests with mocked dependencies |
| `data/history/` | Place your Spotify data export files here |
| `data/graph/` | Cached MusicBrainz artist graphs (auto-created) |

---

## Task 1: Project Scaffold

**Files:**
- Create: `config.py`
- Create: `requirements.txt`
- Create: `tests/conftest.py`
- Create: `.env.example`
- Create: `tests/__init__.py`

- [ ] **Step 1: Update requirements.txt**

```
fiftyone>=0.25.0
fiftyone-brain>=0.16.0
Pillow>=10.0.0
scikit-learn>=1.3.0
requests>=2.31.0
numpy>=1.24.0
umap-learn>=0.5
python-dotenv>=1.0.0
torch>=2.0.0
spotipy>=2.23.0
anthropic>=0.40.0
musicbrainzngs>=0.7.1
sentence-transformers>=2.7.0
pytest>=8.0.0
pytest-mock>=3.14.0
```

- [ ] **Step 2: Create config.py**

```python
import os
from dotenv import load_dotenv

load_dotenv()

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8888/callback")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
DATASET_NAME = os.getenv("DATASET_NAME", "spotify_explorer")
HISTORY_DIR = os.getenv("HISTORY_DIR", "data/history")
GRAPH_DIR = os.getenv("GRAPH_DIR", "data/graph")
```

- [ ] **Step 3: Create .env.example**

```
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here
SPOTIFY_REDIRECT_URI=http://localhost:8888/callback
ANTHROPIC_API_KEY=your_anthropic_key_here
DATASET_NAME=spotify_explorer
HISTORY_DIR=data/history
GRAPH_DIR=data/graph
```

- [ ] **Step 4: Create tests/__init__.py and tests/conftest.py**

```python
# tests/__init__.py  (empty)
```

```python
# tests/conftest.py
import pytest

TEST_TRACK = {
    "track_id": "6eUW0wxWtzkFdaEFsTJto6",
    "title": "Hurt",
    "artist": "Nine Inch Nails",
    "artist_id": "0X380XXpSNLICoudqnW6TVk",
    "album": "The Downward Spiral",
    "duration_ms": 325000,
    "release_year": 1994,
    "explicit": False,
    "preview_url": "https://p.scdn.co/mp3-preview/abc123",
    "popularity": 80,
    "spotify_uri": "spotify:track:6eUW0wxWtzkFdaEFsTJto6",
    "genres": ["industrial rock", "alternative rock"],
}

TEST_AUDIO_FEATURES = {
    "danceability": 0.4,
    "energy": 0.6,
    "key": 5,
    "loudness": -7.5,
    "mode": 0,
    "speechiness": 0.04,
    "acousticness": 0.12,
    "instrumentalness": 0.0,
    "liveness": 0.1,
    "valence": 0.3,
    "tempo": 92.0,
    "duration_ms": 325000,
    "time_signature": 4,
}
```

- [ ] **Step 5: Install dependencies**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Expected: all packages install without errors. sentence-transformers will download the `all-MiniLM-L6-v2` model (~90MB) on first use — this is normal.

- [ ] **Step 6: Verify pytest runs**

```bash
pytest tests/ -v
```

Expected: `no tests ran` (0 items). No import errors.

- [ ] **Step 7: Commit**

```bash
git add config.py requirements.txt .env.example tests/
git commit -m "feat: project scaffold — config, requirements, test setup"
```

---

## Task 2: Spotify Client — Catalog (No User Auth)

**Files:**
- Create: `spotify_client.py`
- Create: `tests/test_spotify_client.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_spotify_client.py
from unittest.mock import MagicMock, patch
import pytest
from spotify_client import SpotifyClient, AUDIO_FEATURE_KEYS

@pytest.fixture
def mock_spotipy(mocker):
    return mocker.patch("spotify_client.spotipy.Spotify")

def test_search_catalog_returns_parsed_tracks(mock_spotipy):
    client = SpotifyClient(user_auth=False)
    client._sp.search.return_value = {
        "tracks": {"items": [{
            "id": "abc123",
            "name": "Hurt",
            "artists": [{"id": "art1", "name": "Nine Inch Nails"}],
            "album": {"name": "The Downward Spiral", "release_date": "1994-03-08"},
            "duration_ms": 325000,
            "explicit": False,
            "preview_url": "https://preview.url",
            "popularity": 80,
            "uri": "spotify:track:abc123",
        }]}
    }
    tracks = client.search_catalog("hurt nine inch nails")
    assert len(tracks) == 1
    assert tracks[0]["track_id"] == "abc123"
    assert tracks[0]["title"] == "Hurt"
    assert tracks[0]["artist"] == "Nine Inch Nails"
    assert tracks[0]["release_year"] == 1994

def test_fetch_audio_features_returns_keyed_dict(mock_spotipy):
    client = SpotifyClient(user_auth=False)
    client._sp.audio_features.return_value = [{
        "id": "abc123",
        "danceability": 0.4, "energy": 0.6, "key": 5,
        "loudness": -7.5, "mode": 0, "speechiness": 0.04,
        "acousticness": 0.12, "instrumentalness": 0.0,
        "liveness": 0.1, "valence": 0.3, "tempo": 92.0,
        "duration_ms": 325000, "time_signature": 4,
    }]
    features = client.fetch_audio_features(["abc123"])
    assert "abc123" in features
    assert set(features["abc123"].keys()) == set(AUDIO_FEATURE_KEYS)

def test_fetch_audio_features_batches_over_100(mock_spotipy):
    client = SpotifyClient(user_auth=False)
    client._sp.audio_features.return_value = []
    ids = [f"id{i}" for i in range(150)]
    client.fetch_audio_features(ids)
    assert client._sp.audio_features.call_count == 2

def test_search_artist_returns_none_when_not_found(mock_spotipy):
    client = SpotifyClient(user_auth=False)
    client._sp.search.return_value = {"artists": {"items": []}}
    result = client.search_artist("nonexistentxyzband")
    assert result is None

def test_parse_track_handles_missing_preview_url(mock_spotipy):
    client = SpotifyClient(user_auth=False)
    client._sp.search.return_value = {
        "tracks": {"items": [{
            "id": "abc123", "name": "Song",
            "artists": [{"id": "a1", "name": "Artist"}],
            "album": {"name": "Album", "release_date": "2020"},
            "duration_ms": 180000, "explicit": False,
            "preview_url": None, "popularity": 50,
            "uri": "spotify:track:abc123",
        }]}
    }
    tracks = client.search_catalog("song")
    assert tracks[0]["preview_url"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_spotify_client.py -v
```

Expected: `ImportError: No module named 'spotify_client'`

- [ ] **Step 3: Implement spotify_client.py**

```python
# spotify_client.py
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
from config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REDIRECT_URI

AUDIO_FEATURE_KEYS = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo", "duration_ms", "time_signature",
]

SPOTIFY_SCOPES = (
    "user-library-read playlist-read-private user-read-recently-played "
    "playlist-modify-private playlist-modify-public"
)


class SpotifyClient:
    def __init__(self, user_auth: bool = False):
        if user_auth:
            self._sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
                client_id=SPOTIFY_CLIENT_ID,
                client_secret=SPOTIFY_CLIENT_SECRET,
                redirect_uri=SPOTIFY_REDIRECT_URI,
                scope=SPOTIFY_SCOPES,
                cache_path=".spotify_token_cache",
            ))
        else:
            self._sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
                client_id=SPOTIFY_CLIENT_ID,
                client_secret=SPOTIFY_CLIENT_SECRET,
            ))

    def search_catalog(self, query: str, limit: int = 50) -> list[dict]:
        results = self._sp.search(q=query, type="track", limit=min(limit, 50))
        return [self._parse_track(t) for t in results["tracks"]["items"]]

    def fetch_audio_features(self, track_ids: list[str]) -> dict[str, dict]:
        features = {}
        for i in range(0, len(track_ids), 100):
            batch = track_ids[i : i + 100]
            results = self._sp.audio_features(batch) or []
            for f in results:
                if f:
                    features[f["id"]] = {k: f[k] for k in AUDIO_FEATURE_KEYS if k in f}
        return features

    def get_artist_info(self, artist_id: str) -> dict:
        a = self._sp.artist(artist_id)
        return {
            "artist_id": a["id"],
            "genres": a.get("genres", []),
            "popularity": a.get("popularity", 0),
        }

    def get_artist_top_tracks(self, artist_id: str, market: str = "US") -> list[dict]:
        results = self._sp.artist_top_tracks(artist_id, country=market)
        return [self._parse_track(t) for t in results["tracks"]]

    def search_artist(self, name: str) -> dict | None:
        results = self._sp.search(q=name, type="artist", limit=1)
        items = results["artists"]["items"]
        if not items:
            return None
        a = items[0]
        return {
            "artist_id": a["id"],
            "name": a["name"],
            "genres": a.get("genres", []),
            "popularity": a.get("popularity", 0),
        }

    def liked_tracks(self) -> list[dict]:
        tracks = []
        results = self._sp.current_user_saved_tracks(limit=50)
        while results:
            tracks.extend(
                self._parse_track(item["track"])
                for item in results["items"]
                if item["track"]
            )
            results = self._sp.next(results) if results["next"] else None
        return tracks

    def user_playlists(self) -> list[dict]:
        playlists, results = [], self._sp.current_user_playlists(limit=50)
        while results:
            playlists.extend(results["items"])
            results = self._sp.next(results) if results["next"] else None
        return playlists

    def playlist_tracks(self, playlist_id: str) -> list[dict]:
        tracks, results = [], self._sp.playlist_tracks(playlist_id, limit=100)
        while results:
            tracks.extend(
                self._parse_track(item["track"])
                for item in results["items"]
                if item["track"]
            )
            results = self._sp.next(results) if results["next"] else None
        return tracks

    def recently_played(self, limit: int = 50) -> list[dict]:
        results = self._sp.current_user_recently_played(limit=limit)
        return [self._parse_track(item["track"]) for item in results["items"]]

    def create_playlist(self, name: str, track_ids: list[str], public: bool = False) -> str:
        user = self._sp.current_user()
        playlist = self._sp.user_playlist_create(user["id"], name, public=public)
        for i in range(0, len(track_ids), 100):
            self._sp.playlist_add_items(playlist["id"], track_ids[i : i + 100])
        return playlist["external_urls"]["spotify"]

    def _parse_track(self, t: dict) -> dict:
        album = t.get("album", {})
        release_date = album.get("release_date", "")
        release_year = int(release_date[:4]) if len(release_date) >= 4 else None
        artists = t.get("artists", [{}])
        return {
            "track_id": t["id"],
            "title": t["name"],
            "artist": artists[0]["name"] if artists else "",
            "artist_id": artists[0]["id"] if artists else "",
            "album": album.get("name", ""),
            "duration_ms": t.get("duration_ms", 0),
            "release_year": release_year,
            "explicit": t.get("explicit", False),
            "preview_url": t.get("preview_url"),
            "popularity": t.get("popularity", 0),
            "spotify_uri": t.get("uri", f"spotify:track:{t['id']}"),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_spotify_client.py -v
```

Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add spotify_client.py tests/test_spotify_client.py
git commit -m "feat: Spotify client — catalog search, audio features, artist lookup"
```

---

## Task 3: History Importer

**Files:**
- Create: `history_importer.py`
- Create: `tests/test_history_importer.py`
- Create: `tests/fixtures/streaming_history.json`

- [ ] **Step 1: Create test fixture**

```json
[
  {
    "ts": "2023-01-15T08:30:00Z",
    "ms_played": 310000,
    "master_metadata_track_name": "Hurt",
    "master_metadata_album_artist_name": "Nine Inch Nails",
    "master_metadata_album_album_name": "The Downward Spiral",
    "spotify_track_uri": "spotify:track:6eUW0wxWtzkFdaEFsTJto6",
    "reason_end": "trackdone"
  },
  {
    "ts": "2022-06-10T20:00:00Z",
    "ms_played": 45000,
    "master_metadata_track_name": "Hurt",
    "master_metadata_album_artist_name": "Nine Inch Nails",
    "master_metadata_album_album_name": "The Downward Spiral",
    "spotify_track_uri": "spotify:track:6eUW0wxWtzkFdaEFsTJto6",
    "reason_end": "fwdbtn"
  },
  {
    "ts": "2023-03-01T12:00:00Z",
    "ms_played": 200000,
    "master_metadata_track_name": "Closer",
    "master_metadata_album_artist_name": "Nine Inch Nails",
    "master_metadata_album_album_name": "The Downward Spiral",
    "spotify_track_uri": "spotify:track:othertrack123",
    "reason_end": "trackdone"
  }
]
```

- [ ] **Step 2: Write failing tests**

```python
# tests/test_history_importer.py
import json
import os
import pytest
from pathlib import Path
from history_importer import parse_history_files, compute_play_stats

FIXTURE_DIR = Path(__file__).parent / "fixtures"

def test_parse_history_aggregates_play_count(tmp_path):
    fixture = FIXTURE_DIR / "streaming_history.json"
    shutil_copy_to = tmp_path / "streaming_history.json"
    shutil_copy_to.write_text(fixture.read_text())
    stats = parse_history_files(str(tmp_path))
    assert "6eUW0wxWtzkFdaEFsTJto6" in stats
    assert stats["6eUW0wxWtzkFdaEFsTJto6"]["play_count"] == 2

def test_parse_history_tracks_skip_count(tmp_path):
    fixture = FIXTURE_DIR / "streaming_history.json"
    (tmp_path / "streaming_history.json").write_text(fixture.read_text())
    stats = parse_history_files(str(tmp_path))
    assert stats["6eUW0wxWtzkFdaEFsTJto6"]["skip_count"] == 1

def test_parse_history_first_and_last_played(tmp_path):
    fixture = FIXTURE_DIR / "streaming_history.json"
    (tmp_path / "streaming_history.json").write_text(fixture.read_text())
    stats = parse_history_files(str(tmp_path))
    s = stats["6eUW0wxWtzkFdaEFsTJto6"]
    assert s["first_played_at"].year == 2022
    assert s["last_played_at"].year == 2023

def test_parse_history_ignores_podcasts(tmp_path):
    data = [{"ts": "2023-01-01T00:00:00Z", "ms_played": 1000,
             "spotify_episode_uri": "spotify:episode:abc", "reason_end": "trackdone"}]
    (tmp_path / "endsong_0.json").write_text(json.dumps(data))
    stats = parse_history_files(str(tmp_path))
    assert len(stats) == 0

def test_compute_play_stats_completion_rate():
    raw = {"play_count": 2, "total_ms_played": 355000, "skip_count": 1,
           "first_played_at": None, "last_played_at": None}
    result = compute_play_stats(raw, duration_ms=325000)
    assert result["completion_rate"] == pytest.approx(355000 / 2 / 325000, abs=0.01)
    assert result["skip_rate"] == pytest.approx(0.5)

def test_compute_play_stats_caps_completion_at_1(():
    raw = {"play_count": 1, "total_ms_played": 999999, "skip_count": 0,
           "first_played_at": None, "last_played_at": None}
    result = compute_play_stats(raw, duration_ms=180000)
    assert result["completion_rate"] == 1.0
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/test_history_importer.py -v
```

Expected: `ImportError: No module named 'history_importer'`

- [ ] **Step 4: Implement history_importer.py**

```python
# history_importer.py
import json
from datetime import datetime
from pathlib import Path


def parse_history_files(history_dir: str) -> dict[str, dict]:
    stats: dict[str, dict] = {}
    history_path = Path(history_dir)
    json_files = (
        list(history_path.glob("Streaming_History_Audio*.json"))
        + list(history_path.glob("endsong*.json"))
    )

    for filepath in json_files:
        with open(filepath, encoding="utf-8") as f:
            entries = json.load(f)

        for entry in entries:
            uri = entry.get("spotify_track_uri", "")
            if not uri or not uri.startswith("spotify:track:"):
                continue

            track_id = uri.split(":")[-1]
            ms_played = entry.get("ms_played", 0)
            reason_end = entry.get("reason_end", "")

            ts_str = entry.get("ts", "")
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")) if ts_str else None
            except ValueError:
                ts = None

            if track_id not in stats:
                stats[track_id] = {
                    "track_id": track_id,
                    "title": entry.get("master_metadata_track_name", ""),
                    "artist": entry.get("master_metadata_album_artist_name", ""),
                    "album": entry.get("master_metadata_album_album_name", ""),
                    "play_count": 0,
                    "total_ms_played": 0,
                    "skip_count": 0,
                    "first_played_at": ts,
                    "last_played_at": ts,
                }

            s = stats[track_id]
            s["play_count"] += 1
            s["total_ms_played"] += ms_played
            if reason_end in ("fwdbtn", "backbtn"):
                s["skip_count"] += 1
            if ts:
                if s["first_played_at"] is None or ts < s["first_played_at"]:
                    s["first_played_at"] = ts
                if s["last_played_at"] is None or ts > s["last_played_at"]:
                    s["last_played_at"] = ts

    return stats


def compute_play_stats(stats: dict, duration_ms: int) -> dict:
    if stats["play_count"] == 0 or duration_ms == 0:
        return {**stats, "completion_rate": 0.0, "skip_rate": 0.0}
    avg_ms = stats["total_ms_played"] / stats["play_count"]
    return {
        **stats,
        "completion_rate": min(avg_ms / duration_ms, 1.0),
        "skip_rate": stats["skip_count"] / stats["play_count"],
    }
```

- [ ] **Step 5: Fix the test syntax error (missing closing paren) and run**

Fix the test: `def test_compute_play_stats_caps_completion_at_1():` (remove extra `(`)

```bash
pytest tests/test_history_importer.py -v
```

Expected: 6 tests pass.

- [ ] **Step 6: Commit**

```bash
git add history_importer.py tests/test_history_importer.py tests/fixtures/
git commit -m "feat: history importer — parse Spotify data export, compute play stats"
```

---

## Task 4: Embeddings Engine

**Files:**
- Create: `embeddings.py`
- Create: `tests/test_embeddings.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_embeddings.py
import numpy as np
import pytest
from tests.conftest import TEST_AUDIO_FEATURES
from embeddings import compute_audio_embedding, compute_text_embedding, AUDIO_FEATURE_KEYS

def test_audio_embedding_length():
    vec = compute_audio_embedding(TEST_AUDIO_FEATURES)
    assert len(vec) == len(AUDIO_FEATURE_KEYS)

def test_audio_embedding_all_values_in_0_1():
    vec = compute_audio_embedding(TEST_AUDIO_FEATURES)
    assert all(0.0 <= v <= 1.0 for v in vec), f"Out-of-range values: {[v for v in vec if not 0<=v<=1]}"

def test_audio_embedding_missing_keys_default_to_zero():
    vec = compute_audio_embedding({})
    assert len(vec) == len(AUDIO_FEATURE_KEYS)
    assert all(0.0 <= v <= 1.0 for v in vec)

def test_text_embedding_returns_list_of_floats():
    vec = compute_text_embedding("Hurt", "Nine Inch Nails", ["industrial rock", "alternative rock"])
    assert isinstance(vec, list)
    assert len(vec) > 0
    assert all(isinstance(v, float) for v in vec)

def test_text_embedding_different_inputs_differ():
    v1 = compute_text_embedding("Hurt", "Nine Inch Nails", ["grunge"])
    v2 = compute_text_embedding("Stairway to Heaven", "Led Zeppelin", ["rock"])
    assert v1 != v2

def test_audio_embedding_high_energy_track():
    features = {**TEST_AUDIO_FEATURES, "energy": 0.95, "tempo": 180.0, "loudness": -3.0}
    vec = compute_audio_embedding(features)
    # energy is index 1 in AUDIO_FEATURE_KEYS
    energy_idx = AUDIO_FEATURE_KEYS.index("energy")
    assert vec[energy_idx] == pytest.approx(0.95)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_embeddings.py -v
```

Expected: `ImportError: No module named 'embeddings'`

- [ ] **Step 3: Implement embeddings.py**

```python
# embeddings.py
import numpy as np
from sentence_transformers import SentenceTransformer

AUDIO_FEATURE_KEYS = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo", "duration_ms", "time_signature",
]

_IDX = {k: i for i, k in enumerate(AUDIO_FEATURE_KEYS)}

_text_model: SentenceTransformer | None = None


def _get_text_model() -> SentenceTransformer:
    global _text_model
    if _text_model is None:
        _text_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _text_model


def compute_audio_embedding(features: dict) -> list[float]:
    vec = np.array([float(features.get(k, 0.0)) for k in AUDIO_FEATURE_KEYS])
    # Normalize each feature to [0, 1]
    vec[_IDX["loudness"]] = np.clip((vec[_IDX["loudness"]] + 60) / 60, 0, 1)
    vec[_IDX["tempo"]] = np.clip(vec[_IDX["tempo"]] / 250, 0, 1)
    vec[_IDX["duration_ms"]] = np.clip(vec[_IDX["duration_ms"]] / 600_000, 0, 1)
    vec[_IDX["key"]] = np.clip(vec[_IDX["key"]] / 11, 0, 1)
    vec[_IDX["time_signature"]] = np.clip((vec[_IDX["time_signature"]] - 3) / 4, 0, 1)
    return np.clip(vec, 0, 1).tolist()


def compute_text_embedding(title: str, artist: str, genres: list[str]) -> list[float]:
    text = f"{title} {artist} {' '.join(genres)}"
    return _get_text_model().encode(text).tolist()


def fit_umap(embeddings: list[list[float]], n_neighbors: int = 15) -> object:
    import umap
    X = np.array(embeddings)
    reducer = umap.UMAP(n_components=2, n_neighbors=min(n_neighbors, len(X) - 1), random_state=42)
    reducer.fit(X)
    return reducer


def project_umap(reducer, embeddings: list[list[float]]) -> list[list[float]]:
    return reducer.transform(np.array(embeddings)).tolist()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_embeddings.py -v
```

Expected: 6 tests pass. (First run downloads ~90MB model — subsequent runs are cached.)

- [ ] **Step 5: Commit**

```bash
git add embeddings.py tests/test_embeddings.py
git commit -m "feat: embeddings — audio feature vector + sentence-transformer text embedding"
```

---

## Task 5: FiftyOne Store

**Files:**
- Create: `fiftyone_store.py`
- Create: `tests/test_fiftyone_store.py`

> Note: These are integration tests — they create a real FiftyOne dataset named `test_spotify_explorer` and clean it up after each test. FiftyOne must be installed and its database running.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_fiftyone_store.py
import pytest
import fiftyone as fo
from tests.conftest import TEST_TRACK, TEST_AUDIO_FEATURES
from fiftyone_store import (
    get_or_create_dataset, upsert_track, query_by_ids,
    tag_playlist, get_tagged_playlist,
)

TEST_DS = "test_spotify_explorer"

@pytest.fixture(autouse=True)
def cleanup():
    yield
    if fo.dataset_exists(TEST_DS):
        fo.delete_dataset(TEST_DS)

def test_get_or_create_creates_persistent_dataset():
    ds = get_or_create_dataset(TEST_DS)
    assert fo.dataset_exists(TEST_DS)
    assert ds.persistent

def test_get_or_create_returns_existing_dataset():
    ds1 = get_or_create_dataset(TEST_DS)
    ds2 = get_or_create_dataset(TEST_DS)
    assert ds1.name == ds2.name

def test_upsert_track_adds_sample():
    ds = get_or_create_dataset(TEST_DS)
    track = {**TEST_TRACK, "audio_features": TEST_AUDIO_FEATURES}
    upsert_track(ds, track)
    assert len(ds) == 1

def test_upsert_track_updates_existing():
    ds = get_or_create_dataset(TEST_DS)
    upsert_track(ds, {**TEST_TRACK})
    upsert_track(ds, {**TEST_TRACK, "play_count": 5})
    assert len(ds) == 1
    sample = ds.first()
    assert sample["play_count"] == 5

def test_upsert_track_uses_preview_url_as_filepath():
    ds = get_or_create_dataset(TEST_DS)
    upsert_track(ds, {**TEST_TRACK})
    assert ds.first().filepath == TEST_TRACK["preview_url"]

def test_upsert_track_fallback_filepath_when_no_preview():
    ds = get_or_create_dataset(TEST_DS)
    track = {**TEST_TRACK, "preview_url": None}
    upsert_track(ds, track)
    assert "spotify://track/" in ds.first().filepath

def test_query_by_ids_returns_matching_samples():
    ds = get_or_create_dataset(TEST_DS)
    upsert_track(ds, TEST_TRACK)
    view = query_by_ids(ds, [TEST_TRACK["track_id"]])
    assert len(view) == 1

def test_tag_and_get_tagged_playlist():
    ds = get_or_create_dataset(TEST_DS)
    upsert_track(ds, TEST_TRACK)
    tag_playlist(ds, [TEST_TRACK["track_id"]], "my_playlist")
    ids = get_tagged_playlist(ds, "my_playlist")
    assert TEST_TRACK["track_id"] in ids
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_fiftyone_store.py -v
```

Expected: `ImportError: No module named 'fiftyone_store'`

- [ ] **Step 3: Implement fiftyone_store.py**

```python
# fiftyone_store.py
import fiftyone as fo
from config import DATASET_NAME


def get_or_create_dataset(name: str = DATASET_NAME) -> fo.Dataset:
    if fo.dataset_exists(name):
        return fo.load_dataset(name)
    ds = fo.Dataset(name)
    ds.persistent = True
    return ds


def upsert_track(dataset: fo.Dataset, track: dict) -> None:
    track_id = track["track_id"]
    filepath = track.get("preview_url") or f"spotify://track/{track_id}"

    view = dataset.match(fo.ViewField("track_id") == track_id)
    if len(view) > 0:
        sample = view.first()
        for key, value in track.items():
            sample[key] = value
        sample.save()
    else:
        sample = fo.Sample(filepath=filepath)
        for key, value in track.items():
            sample[key] = value
        dataset.add_sample(sample)


def query_by_ids(dataset: fo.Dataset, track_ids: list[str]) -> fo.DatasetView:
    return dataset.match(fo.ViewField("track_id").is_in(track_ids))


def tag_playlist(dataset: fo.Dataset, track_ids: list[str], tag: str = "playlist") -> None:
    query_by_ids(dataset, track_ids).tag_samples(tag)


def get_tagged_playlist(dataset: fo.Dataset, tag: str = "playlist") -> list[str]:
    return [s["track_id"] for s in dataset.match_tags(tag)]


def open_in_browser(dataset: fo.Dataset, view: fo.DatasetView | None = None, port: int = 5151):
    return fo.launch_app(view or dataset, port=port)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_fiftyone_store.py -v
```

Expected: 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add fiftyone_store.py tests/test_fiftyone_store.py
git commit -m "feat: FiftyOne store — dataset CRUD, upsert, query, tag, browser"
```

---

## Task 6: Artist Graph (MusicBrainz)

**Files:**
- Create: `artist_graph.py`
- Create: `tests/test_artist_graph.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_artist_graph.py
import json
import pytest
from unittest.mock import patch, MagicMock
from artist_graph import (
    resolve_artist_mbid, fetch_artist_relations,
    traverse_graph, get_all_artist_names, explain_path,
)

NIN_MBID = "b7ffd2af-418f-4be2-bdd1-22f8b48613da"
REZNOR_MBID = "1f9df192-a621-4f54-8850-2c5373b7eac9"

@pytest.fixture
def mock_mb(mocker):
    return mocker.patch("artist_graph.mb")

def test_resolve_artist_mbid_returns_id(mock_mb):
    mock_mb.search_artists.return_value = {
        "artist-list": [{"id": NIN_MBID, "name": "Nine Inch Nails"}]
    }
    result = resolve_artist_mbid("Nine Inch Nails")
    assert result == NIN_MBID

def test_resolve_artist_mbid_returns_none_when_not_found(mock_mb):
    mock_mb.search_artists.return_value = {"artist-list": []}
    assert resolve_artist_mbid("xyznonexistentband") is None

def test_fetch_artist_relations_extracts_members(mock_mb):
    mock_mb.get_artist_by_id.return_value = {
        "artist": {
            "name": "Nine Inch Nails",
            "artist-relation-list": [{
                "type": "member of band",
                "direction": "backward",
                "artist": {"id": REZNOR_MBID, "name": "Trent Reznor"},
                "begin": "1988", "end": "",
                "attribute-list": ["vocals"],
            }]
        }
    }
    relations = fetch_artist_relations(NIN_MBID)
    assert relations["name"] == "Nine Inch Nails"
    assert len(relations["relations"]) == 1
    assert relations["relations"][0]["related_name"] == "Trent Reznor"
    assert relations["relations"][0]["type"] == "member of band"

def test_traverse_graph_uses_cache(tmp_path, mock_mb):
    cached = {
        "seed": "Nine Inch Nails", "seed_mbid": NIN_MBID,
        "depth": 1,
        "nodes": {NIN_MBID: {"mbid": NIN_MBID, "name": "Nine Inch Nails", "type": "Group", "depth": 0}},
        "edges": [],
    }
    cache_file = tmp_path / "nine_inch_nails_d1.json"
    cache_file.write_text(json.dumps(cached))
    graph = traverse_graph("Nine Inch Nails", depth=1, graph_dir=str(tmp_path))
    mock_mb.search_artists.assert_not_called()
    assert graph["seed"] == "Nine Inch Nails"

def test_get_all_artist_names_excludes_seed(tmp_path, mock_mb):
    cached = {
        "seed": "Nine Inch Nails", "seed_mbid": NIN_MBID, "depth": 1,
        "nodes": {
            NIN_MBID: {"mbid": NIN_MBID, "name": "Nine Inch Nails", "type": "Group", "depth": 0},
            REZNOR_MBID: {"mbid": REZNOR_MBID, "name": "Trent Reznor", "type": "Person", "depth": 1},
        },
        "edges": [{"from": NIN_MBID, "to": REZNOR_MBID, "type": "member of band",
                   "attributes": ["vocals"], "begin": "1988", "end": ""}],
    }
    (tmp_path / "nine_inch_nails_d1.json").write_text(json.dumps(cached))
    graph = traverse_graph("Nine Inch Nails", depth=1, graph_dir=str(tmp_path))
    artists = get_all_artist_names(graph)
    names = [a["name"] for a in artists]
    assert "Nine Inch Nails" not in names
    assert "Trent Reznor" in names

def test_explain_path_generates_description(tmp_path, mock_mb):
    cached = {
        "seed": "Nine Inch Nails", "seed_mbid": NIN_MBID, "depth": 1,
        "nodes": {
            NIN_MBID: {"mbid": NIN_MBID, "name": "Nine Inch Nails", "type": "Group", "depth": 0},
            REZNOR_MBID: {"mbid": REZNOR_MBID, "name": "Trent Reznor", "type": "Person", "depth": 1},
        },
        "edges": [{"from": NIN_MBID, "to": REZNOR_MBID, "type": "member of band",
                   "attributes": ["vocals"], "begin": "1988", "end": ""}],
    }
    (tmp_path / "nine_inch_nails_d1.json").write_text(json.dumps(cached))
    graph = traverse_graph("Nine Inch Nails", depth=1, graph_dir=str(tmp_path))
    explanation = explain_path(graph, REZNOR_MBID)
    assert "Trent Reznor" in explanation
    assert "Nine Inch Nails" in explanation
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_artist_graph.py -v
```

Expected: `ImportError: No module named 'artist_graph'`

- [ ] **Step 3: Implement artist_graph.py**

```python
# artist_graph.py
import json
import time
from pathlib import Path
import musicbrainzngs as mb
from config import GRAPH_DIR

mb.set_useragent("spotify-playlist-agent", "0.1", "your@email.com")


def resolve_artist_mbid(name: str) -> str | None:
    result = mb.search_artists(artist=name, limit=1)
    artists = result.get("artist-list", [])
    return artists[0]["id"] if artists else None


def fetch_artist_relations(mbid: str) -> dict:
    time.sleep(1)
    result = mb.get_artist_by_id(mbid, includes=["artist-rels"])
    artist = result["artist"]
    relations = [
        {
            "type": rel.get("type", ""),
            "direction": rel.get("direction", ""),
            "related_mbid": rel.get("artist", {}).get("id", ""),
            "related_name": rel.get("artist", {}).get("name", ""),
            "begin": rel.get("begin", ""),
            "end": rel.get("end", ""),
            "attributes": rel.get("attribute-list", []),
        }
        for rel in artist.get("artist-relation-list", [])
    ]
    return {"mbid": mbid, "name": artist.get("name", ""), "type": artist.get("type", ""), "relations": relations}


def traverse_graph(seed_name: str, depth: int = 2, graph_dir: str = GRAPH_DIR) -> dict:
    slug = seed_name.lower().replace(" ", "_")
    cache_path = Path(graph_dir) / f"{slug}_d{depth}.json"
    Path(graph_dir).mkdir(exist_ok=True)

    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    seed_mbid = resolve_artist_mbid(seed_name)
    if not seed_mbid:
        raise ValueError(f"Artist not found in MusicBrainz: {seed_name}")

    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    queue = [(seed_mbid, seed_name, 0)]
    visited: set[str] = set()

    while queue:
        mbid, name, current_depth = queue.pop(0)
        if mbid in visited or current_depth > depth:
            continue
        visited.add(mbid)

        data = fetch_artist_relations(mbid)
        nodes[mbid] = {"mbid": mbid, "name": data["name"], "type": data.get("type", ""), "depth": current_depth}

        for rel in data["relations"]:
            related_mbid = rel["related_mbid"]
            if not related_mbid:
                continue
            edges.append({"from": mbid, "to": related_mbid, "type": rel["type"],
                          "attributes": rel["attributes"], "begin": rel["begin"], "end": rel["end"]})
            if current_depth < depth and related_mbid not in visited:
                queue.append((related_mbid, rel["related_name"], current_depth + 1))

    graph = {"seed": seed_name, "seed_mbid": seed_mbid, "depth": depth, "nodes": nodes, "edges": edges}
    with open(cache_path, "w") as f:
        json.dump(graph, f, indent=2, default=str)
    return graph


def get_all_artist_names(graph: dict) -> list[dict]:
    seed_mbid = graph["seed_mbid"]
    nodes, edges = graph["nodes"], graph["edges"]
    result = []
    for mbid, node in nodes.items():
        if mbid == seed_mbid:
            continue
        incoming = [e for e in edges if e["to"] == mbid]
        if incoming:
            e = incoming[0]
            from_name = nodes.get(e["from"], {}).get("name", graph["seed"])
            path = (f"{graph['seed']} → {from_name} → {node['name']}"
                    if from_name != graph["seed"] else f"{graph['seed']} → {node['name']}")
            conn_type = e["type"]
        else:
            path, conn_type = f"{graph['seed']} → {node['name']}", "unknown"
        result.append({"mbid": mbid, "name": node["name"], "depth": node["depth"],
                       "connection_path": path, "connection_type": conn_type})
    return result


def explain_path(graph: dict, artist_mbid: str) -> str:
    nodes, edges = graph["nodes"], graph["edges"]
    node = nodes.get(artist_mbid)
    if not node:
        return "Artist not found in graph."
    incoming = [e for e in edges if e["to"] == artist_mbid]
    if not incoming:
        return f"{node['name']} is {graph['seed']}."
    e = incoming[0]
    from_node = nodes.get(e["from"], {})
    role = f" ({', '.join(e['attributes'])})" if e.get("attributes") else ""
    return f"{node['name']} — {e['type']}{role} with {from_node.get('name', graph['seed'])}, associated with {graph['seed']}."
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_artist_graph.py -v
```

Expected: 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add artist_graph.py tests/test_artist_graph.py
git commit -m "feat: artist graph — MusicBrainz traversal, cache, connection explanation"
```

---

## Task 7: Track DNA (Bateman Mode)

**Files:**
- Create: `track_dna.py`
- Create: `tests/test_track_dna.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_track_dna.py
import pytest
from unittest.mock import patch
from track_dna import resolve_recording_mbid, fetch_recording_credits, expand_bateman

RECORDING_MBID = "b52a8f8a-aaaa-bbbb-cccc-123456789abc"

@pytest.fixture
def mock_mb(mocker):
    return mocker.patch("track_dna.mb")

def test_resolve_recording_mbid_returns_id(mock_mb):
    mock_mb.search_recordings.return_value = {
        "recording-list": [{"id": RECORDING_MBID, "title": "Hurt"}]
    }
    result = resolve_recording_mbid("Hurt", "Nine Inch Nails")
    assert result == RECORDING_MBID

def test_resolve_recording_mbid_returns_none_when_not_found(mock_mb):
    mock_mb.search_recordings.return_value = {"recording-list": []}
    assert resolve_recording_mbid("xyznonexistent", "nobody") is None

def test_fetch_recording_credits_extracts_composers(mock_mb):
    mock_mb.get_recording_by_id.return_value = {
        "recording": {
            "title": "Hurt",
            "artist-relation-list": [
                {"type": "composer", "artist": {"id": "c1", "name": "Trent Reznor"}, "attribute-list": []},
                {"type": "performer", "artist": {"id": "c2", "name": "Nine Inch Nails"}, "attribute-list": []},
            ],
            "work-relation-list": [],
        }
    }
    credits = fetch_recording_credits(RECORDING_MBID)
    assert "Trent Reznor" in credits["composers"]
    assert credits["producers"] == []

def test_fetch_recording_credits_extracts_samples(mock_mb):
    mock_mb.get_recording_by_id.return_value = {
        "recording": {
            "title": "Fight the Power",
            "artist-relation-list": [],
            "recording-relation-list": [
                {
                    "type": "samples material from",
                    "recording": {"id": "orig123", "title": "Funky Drummer",
                                  "artist-credit": [{"artist": {"name": "James Brown"}}]},
                }
            ],
            "work-relation-list": [],
        }
    }
    credits = fetch_recording_credits(RECORDING_MBID)
    assert len(credits["samples"]) == 1
    assert credits["samples"][0]["title"] == "Funky Drummer"
    assert credits["samples"][0]["artist"] == "James Brown"

def test_expand_bateman_returns_artist_names(mock_mb):
    mock_mb.search_recordings.return_value = {"recording-list": [{"id": RECORDING_MBID, "title": "Hurt"}]}
    mock_mb.get_recording_by_id.return_value = {
        "recording": {
            "title": "Hurt",
            "artist-relation-list": [
                {"type": "composer", "artist": {"id": "c1", "name": "Trent Reznor"}, "attribute-list": []},
            ],
            "recording-relation-list": [],
            "work-relation-list": [],
        }
    }
    mock_mb.search_artists.return_value = {"artist-list": [{"id": "c1", "name": "Trent Reznor"}]}
    mock_mb.get_artist_by_id.return_value = {"artist": {"name": "Trent Reznor", "artist-relation-list": []}}
    result = expand_bateman("Hurt", "Nine Inch Nails", depth=1)
    assert "persons" in result
    assert any(p["name"] == "Trent Reznor" for p in result["persons"])
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_track_dna.py -v
```

Expected: `ImportError: No module named 'track_dna'`

- [ ] **Step 3: Implement track_dna.py**

```python
# track_dna.py
import time
import musicbrainzngs as mb
from artist_graph import fetch_artist_relations, resolve_artist_mbid

mb.set_useragent("spotify-playlist-agent", "0.1", "your@email.com")

COMPOSER_TYPES = {"composer", "lyricist", "writer", "arranger"}
PRODUCER_TYPES = {"producer", "mix", "engineer", "recording"}
PERFORMER_TYPES = {"performer", "instrument", "vocal", "guitar", "bass", "drums", "keyboards"}
SAMPLE_TYPES = {"samples material from", "has samples of"}


def resolve_recording_mbid(title: str, artist: str) -> str | None:
    time.sleep(1)
    result = mb.search_recordings(recording=title, artist=artist, limit=1)
    recordings = result.get("recording-list", [])
    return recordings[0]["id"] if recordings else None


def fetch_recording_credits(recording_mbid: str) -> dict:
    time.sleep(1)
    result = mb.get_recording_by_id(
        recording_mbid,
        includes=["artist-rels", "recording-rels", "work-rels"],
    )
    rec = result["recording"]

    composers, producers, session_musicians, samples = [], [], [], []

    for rel in rec.get("artist-relation-list", []):
        rel_type = rel.get("type", "").lower()
        artist_name = rel.get("artist", {}).get("name", "")
        artist_id = rel.get("artist", {}).get("id", "")
        attrs = rel.get("attribute-list", [])
        if rel_type in COMPOSER_TYPES:
            composers.append({"name": artist_name, "mbid": artist_id, "role": rel_type})
        elif any(t in rel_type for t in PRODUCER_TYPES):
            producers.append({"name": artist_name, "mbid": artist_id, "role": rel_type})
        elif rel_type in PERFORMER_TYPES or any(a.lower() in PERFORMER_TYPES for a in attrs):
            session_musicians.append({"name": artist_name, "mbid": artist_id, "role": attrs[0] if attrs else rel_type})

    for rel in rec.get("recording-relation-list", []):
        if rel.get("type", "").lower() in SAMPLE_TYPES:
            sampled = rel.get("recording", {})
            credits = sampled.get("artist-credit", [{}])
            samples.append({
                "title": sampled.get("title", ""),
                "recording_mbid": sampled.get("id", ""),
                "artist": credits[0].get("artist", {}).get("name", "") if credits else "",
            })

    return {
        "recording_mbid": recording_mbid,
        "title": rec.get("title", ""),
        "composers": [c["name"] for c in composers],
        "producers": [p["name"] for p in producers],
        "session_musicians": session_musicians,
        "samples": samples,
        "composer_details": composers,
        "producer_details": producers,
    }


def expand_bateman(title: str, artist: str, depth: int = 1) -> dict:
    mbid = resolve_recording_mbid(title, artist)
    if not mbid:
        return {"error": f"Recording not found: {title} by {artist}"}

    credits = fetch_recording_credits(mbid)
    persons: list[dict] = []

    for person in credits["composer_details"] + credits["producer_details"] + credits["session_musicians"]:
        mbid_p = person.get("mbid", "")
        artist_mbid = resolve_artist_mbid(person["name"])
        persons.append({
            "name": person["name"],
            "mbid": mbid_p or artist_mbid,
            "link_type": person.get("role", "contributor"),
            "bateman_path": f"{title} by {artist} → {person.get('role', 'credit')} → {person['name']}",
        })

    sample_origins = []
    for sample in credits["samples"]:
        sample_origins.append({
            "title": sample["title"],
            "artist": sample["artist"],
            "recording_mbid": sample["recording_mbid"],
            "bateman_path": f"{title} by {artist} → samples → {sample['title']} by {sample['artist']}",
        })

    return {
        "seed_title": title,
        "seed_artist": artist,
        "recording_mbid": credits["recording_mbid"],
        "composers": credits["composers"],
        "producers": credits["producers"],
        "session_musicians": credits["session_musicians"],
        "samples": sample_origins,
        "persons": persons,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_track_dna.py -v
```

Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add track_dna.py tests/test_track_dna.py
git commit -m "feat: track DNA — MusicBrainz recording credits + sample resolution (Bateman mode)"
```

---

## Task 8: Agent Tools

**Files:**
- Create: `agent_tools.py`
- Create: `tests/test_agent_tools.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_agent_tools.py
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from tests.conftest import TEST_TRACK, TEST_AUDIO_FEATURES

@pytest.fixture(autouse=True)
def mock_deps(mocker):
    mocker.patch("agent_tools.SpotifyClient")
    mocker.patch("agent_tools.get_or_create_dataset")
    mocker.patch("agent_tools.upsert_track")
    mocker.patch("agent_tools.tag_playlist")
    mocker.patch("agent_tools.get_tagged_playlist")
    mocker.patch("agent_tools.open_in_browser")
    mocker.patch("agent_tools.traverse_graph")
    mocker.patch("agent_tools.get_all_artist_names")
    mocker.patch("agent_tools.expand_bateman")

from agent_tools import execute_tool, TOOLS

def test_tools_list_has_15_tools():
    assert len(TOOLS) == 15

def test_all_tools_have_required_fields():
    for tool in TOOLS:
        assert "name" in tool
        assert "description" in tool
        assert "input_schema" in tool

def test_execute_search_catalog(mocker):
    import agent_tools
    mock_client = MagicMock()
    mock_client.search_catalog.return_value = [TEST_TRACK]
    agent_tools._spotify = mock_client
    result = execute_tool("search_catalog", {"query": "black hole sun"})
    assert result["count"] == 1
    assert result["tracks"][0]["track_id"] == TEST_TRACK["track_id"]

def test_execute_fetch_audio_features(mocker):
    import agent_tools
    mock_client = MagicMock()
    mock_client.fetch_audio_features.return_value = {"abc123": TEST_AUDIO_FEATURES}
    agent_tools._spotify = mock_client
    result = execute_tool("fetch_audio_features", {"track_ids": ["abc123"]})
    assert "abc123" in result["features"]

def test_execute_rank_and_select_limits_count():
    candidates = [
        {"track_id": f"t{i}", "audio_sim": float(i) / 10, "completion_rate": 0.8, "skip_rate": 0.1}
        for i in range(30)
    ]
    result = execute_tool("rank_and_select", {"candidates": candidates, "count": 5})
    assert len(result["playlist"]) == 5

def test_execute_rank_and_select_boosts_discovery():
    candidates = [
        {"track_id": "known", "audio_sim": 0.9, "completion_rate": 0.9, "skip_rate": 0.0, "is_new_discovery": False},
        {"track_id": "new", "audio_sim": 0.7, "completion_rate": 0.5, "skip_rate": 0.1, "is_new_discovery": True},
    ]
    result = execute_tool("rank_and_select", {"candidates": candidates, "count": 1, "boost_discovery": True})
    assert result["playlist"][0]["track_id"] == "new"

def test_execute_unknown_tool_returns_error():
    result = execute_tool("nonexistent_tool", {})
    assert "error" in result
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_agent_tools.py -v
```

Expected: `ImportError: No module named 'agent_tools'`

- [ ] **Step 3: Implement agent_tools.py**

```python
# agent_tools.py
import json
import numpy as np
import fiftyone as fo
from spotify_client import SpotifyClient
from embeddings import compute_audio_embedding, compute_text_embedding
from fiftyone_store import get_or_create_dataset, upsert_track, tag_playlist, get_tagged_playlist, open_in_browser
from artist_graph import traverse_graph, get_all_artist_names
from track_dna import expand_bateman
from config import DATASET_NAME

_spotify = SpotifyClient(user_auth=False)
_spotify_user: SpotifyClient | None = None


def _user_client() -> SpotifyClient:
    global _spotify_user
    if _spotify_user is None:
        _spotify_user = SpotifyClient(user_auth=True)
    return _spotify_user


TOOLS = [
    {"name": "search_catalog",
     "description": "Search the Spotify catalog by keywords. Returns up to 50 tracks with metadata.",
     "input_schema": {"type": "object", "properties": {
         "query": {"type": "string"}, "limit": {"type": "integer", "default": 50}},
         "required": ["query"]}},
    {"name": "fetch_audio_features",
     "description": "Get Spotify audio features (tempo, energy, valence, etc.) for a list of track IDs.",
     "input_schema": {"type": "object", "properties": {
         "track_ids": {"type": "array", "items": {"type": "string"}}}, "required": ["track_ids"]}},
    {"name": "embed_tracks",
     "description": "Compute audio and text embeddings for candidate tracks and upsert into FiftyOne cache.",
     "input_schema": {"type": "object", "properties": {
         "tracks": {"type": "array", "items": {"type": "object"}}}, "required": ["tracks"]}},
    {"name": "filter_by_features",
     "description": "Filter candidate tracks by audio feature ranges.",
     "input_schema": {"type": "object", "properties": {
         "track_ids": {"type": "array", "items": {"type": "string"}},
         "energy_min": {"type": "number"}, "energy_max": {"type": "number"},
         "valence_min": {"type": "number"}, "valence_max": {"type": "number"},
         "tempo_min": {"type": "number"}, "tempo_max": {"type": "number"},
         "instrumentalness_min": {"type": "number"}, "acousticness_min": {"type": "number"}},
         "required": ["track_ids"]}},
    {"name": "query_dataset",
     "description": "Find tracks in the FiftyOne cache similar to given track IDs using audio embedding similarity.",
     "input_schema": {"type": "object", "properties": {
         "track_ids": {"type": "array", "items": {"type": "string"}},
         "k": {"type": "integer", "default": 20}}, "required": ["track_ids"]}},
    {"name": "get_listening_history",
     "description": "Get personal listening signals (play_count, completion_rate, skip_rate) from the local dataset.",
     "input_schema": {"type": "object", "properties": {
         "track_ids": {"type": "array", "items": {"type": "string"}}}, "required": ["track_ids"]}},
    {"name": "rank_and_select",
     "description": "Score and rank candidate tracks, select the final N for the playlist.",
     "input_schema": {"type": "object", "properties": {
         "candidates": {"type": "array", "items": {"type": "object"}},
         "count": {"type": "integer", "default": 20},
         "boost_discovery": {"type": "boolean", "default": False}},
         "required": ["candidates", "count"]}},
    {"name": "open_in_fiftyone",
     "description": "Tag final playlist tracks and open them in the FiftyOne visual browser for review.",
     "input_schema": {"type": "object", "properties": {
         "track_ids": {"type": "array", "items": {"type": "string"}},
         "tag": {"type": "string", "default": "playlist"}}, "required": ["track_ids"]}},
    {"name": "map_artist_connections",
     "description": "Query MusicBrainz for all artists connected to a seed artist.",
     "input_schema": {"type": "object", "properties": {
         "artist_name": {"type": "string"}, "depth": {"type": "integer", "default": 2}},
         "required": ["artist_name"]}},
    {"name": "traverse_artist_graph",
     "description": "Return all artists in the connection graph with paths back to the seed.",
     "input_schema": {"type": "object", "properties": {
         "artist_name": {"type": "string"}, "depth": {"type": "integer", "default": 2}},
         "required": ["artist_name"]}},
    {"name": "fetch_artist_top_tracks",
     "description": "Get top tracks from a list of artist names. Returns merged track list.",
     "input_schema": {"type": "object", "properties": {
         "artist_names": {"type": "array", "items": {"type": "string"}},
         "tracks_per_artist": {"type": "integer", "default": 5}}, "required": ["artist_names"]}},
    {"name": "explain_connection",
     "description": "Generate a plain-English explanation of how a track connects to the seed artist.",
     "input_schema": {"type": "object", "properties": {
         "track_id": {"type": "string"}, "seed_artist": {"type": "string"},
         "connection_path": {"type": "string"}}, "required": ["track_id", "seed_artist"]}},
    {"name": "deep_dive_track",
     "description": "Bateman mode: resolve a track to MusicBrainz and fetch all credits (composers, producers, session musicians, samples).",
     "input_schema": {"type": "object", "properties": {
         "title": {"type": "string"}, "artist": {"type": "string"},
         "depth": {"type": "integer", "default": 1}}, "required": ["title", "artist"]}},
    {"name": "find_tracks_by_person",
     "description": "Given a person's name (writer, producer), find other recordings they worked on via Spotify search.",
     "input_schema": {"type": "object", "properties": {
         "person_name": {"type": "string"}, "role": {"type": "string"},
         "limit": {"type": "integer", "default": 20}}, "required": ["person_name"]}},
    {"name": "resolve_samples",
     "description": "For each sampled recording in a track's DNA, find the original artists' top tracks on Spotify.",
     "input_schema": {"type": "object", "properties": {
         "samples": {"type": "array", "items": {"type": "object"}},
         "tracks_per_sample": {"type": "integer", "default": 5}}, "required": ["samples"]}},
]


def execute_tool(name: str, inputs: dict) -> dict:
    dataset = get_or_create_dataset()

    if name == "search_catalog":
        tracks = _spotify.search_catalog(inputs["query"], limit=inputs.get("limit", 50))
        return {"tracks": tracks, "count": len(tracks)}

    if name == "fetch_audio_features":
        return {"features": _spotify.fetch_audio_features(inputs["track_ids"])}

    if name == "embed_tracks":
        tracks = inputs["tracks"]
        for track in tracks:
            af = track.get("audio_features") or {}
            if af:
                track["audio_embedding"] = compute_audio_embedding(af)
                for k, v in af.items():
                    track[k] = v
            track["text_embedding"] = compute_text_embedding(
                track.get("title", ""), track.get("artist", ""), track.get("genres", []))
            track["is_new_discovery"] = track.get("play_count", 0) == 0
            upsert_track(dataset, track)
        return {"embedded": len(tracks)}

    if name == "filter_by_features":
        view = dataset.match(fo.ViewField("track_id").is_in(inputs["track_ids"]))
        filters = [
            ("energy", inputs.get("energy_min", 0.0), inputs.get("energy_max", 1.0)),
            ("valence", inputs.get("valence_min", 0.0), inputs.get("valence_max", 1.0)),
            ("tempo", inputs.get("tempo_min", 0.0), inputs.get("tempo_max", 300.0)),
            ("instrumentalness", inputs.get("instrumentalness_min", 0.0), 1.0),
            ("acousticness", inputs.get("acousticness_min", 0.0), 1.0),
        ]
        for field, lo, hi in filters:
            view = view.match((fo.ViewField(field) >= lo) & (fo.ViewField(field) <= hi))
        ids = [s["track_id"] for s in view]
        return {"track_ids": ids, "count": len(ids)}

    if name == "query_dataset":
        k = inputs.get("k", 20)
        seed_ids = set(inputs["track_ids"])
        seed_view = dataset.match(fo.ViewField("track_id").is_in(list(seed_ids)))
        seed_embs = [s["audio_embedding"] for s in seed_view if s.get("audio_embedding")]
        if not seed_embs:
            return {"track_ids": [], "count": 0}
        seed_mean = np.mean(seed_embs, axis=0)
        scored = []
        for s in dataset:
            emb = s.get("audio_embedding")
            if emb and s["track_id"] not in seed_ids:
                scored.append((s["track_id"], float(np.dot(seed_mean, emb))))
        top = sorted(scored, key=lambda x: -x[1])[:k]
        return {"track_ids": [t[0] for t in top], "count": len(top)}

    if name == "get_listening_history":
        view = dataset.match(fo.ViewField("track_id").is_in(inputs["track_ids"]))
        history = {s["track_id"]: {
            "play_count": s.get("play_count") or 0,
            "completion_rate": s.get("completion_rate") or 0.0,
            "skip_rate": s.get("skip_rate") or 0.0,
            "is_new_discovery": s.get("is_new_discovery", True),
        } for s in view}
        return {"history": history}

    if name == "rank_and_select":
        candidates = inputs["candidates"]
        boost = inputs.get("boost_discovery", False)
        for c in candidates:
            score = c.get("audio_sim", 0.5)
            score += c.get("completion_rate", 0.0) * 0.3
            score -= c.get("skip_rate", 0.0) * 0.2
            if boost and c.get("is_new_discovery", True):
                score += 0.15
            c["score"] = score
        ranked = sorted(candidates, key=lambda x: -x.get("score", 0))
        return {"playlist": ranked[: inputs.get("count", 20)]}

    if name == "open_in_fiftyone":
        track_ids, tag = inputs["track_ids"], inputs.get("tag", "playlist")
        tag_playlist(dataset, track_ids, tag)
        view = dataset.match_tags(tag)
        open_in_browser(dataset, view)
        return {"status": "opened", "track_count": len(track_ids), "url": "http://localhost:5151"}

    if name == "map_artist_connections":
        graph = traverse_graph(inputs["artist_name"], depth=inputs.get("depth", 2))
        return {"artists": get_all_artist_names(graph), "total": len(graph["nodes"])}

    if name == "traverse_artist_graph":
        graph = traverse_graph(inputs["artist_name"], depth=inputs.get("depth", 2))
        return {"artists": get_all_artist_names(graph)}

    if name == "fetch_artist_top_tracks":
        tracks = []
        for artist_name in inputs["artist_names"]:
            artist = _spotify.search_artist(artist_name)
            if artist:
                top = _spotify.get_artist_top_tracks(artist["artist_id"])
                for t in top[: inputs.get("tracks_per_artist", 5)]:
                    t["connection_artist"] = artist_name
                    tracks.append(t)
        return {"tracks": tracks, "count": len(tracks)}

    if name == "explain_connection":
        path = inputs.get("connection_path", "direct connection")
        return {"explanation": f"{inputs['track_id']} connects to {inputs['seed_artist']} via: {path}"}

    if name == "deep_dive_track":
        return expand_bateman(inputs["title"], inputs["artist"], depth=inputs.get("depth", 1))

    if name == "find_tracks_by_person":
        query = f"{inputs['person_name']} {inputs.get('role', '')}"
        tracks = _spotify.search_catalog(query.strip(), limit=inputs.get("limit", 20))
        return {"tracks": tracks, "count": len(tracks)}

    if name == "resolve_samples":
        tracks = []
        for sample in inputs["samples"]:
            artist = _spotify.search_artist(sample.get("artist", ""))
            if artist:
                top = _spotify.get_artist_top_tracks(artist["artist_id"])
                for t in top[: inputs.get("tracks_per_sample", 5)]:
                    t["bateman_path"] = sample.get("bateman_path", "")
                    t["bateman_link_type"] = "sample"
                    tracks.append(t)
        return {"tracks": tracks, "count": len(tracks)}

    return {"error": f"Unknown tool: {name}"}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_agent_tools.py -v
```

Expected: 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git add agent_tools.py tests/test_agent_tools.py
git commit -m "feat: agent tools — all 15 tools for sonic, connection, and Bateman modes"
```

---

## Task 9: index.py Pipeline

**Files:**
- Create: `index.py`

> No isolated unit tests — integration with Spotify API and FiftyOne. Manually verify each flag works.

- [ ] **Step 1: Implement index.py**

```python
# index.py
import argparse
from spotify_client import SpotifyClient
from history_importer import parse_history_files, compute_play_stats
from embeddings import compute_audio_embedding, compute_text_embedding
from fiftyone_store import get_or_create_dataset, upsert_track
from config import HISTORY_DIR


def enrich_and_upsert(dataset, tracks: list[dict], client: SpotifyClient,
                      history_stats: dict, source: str) -> None:
    track_ids = [t["track_id"] for t in tracks if t.get("track_id")]
    features_map = client.fetch_audio_features(track_ids)

    for track in tracks:
        tid = track.get("track_id")
        if not tid:
            continue

        # Skip already-indexed if --resume
        track["source"] = source

        af = features_map.get(tid, {})
        if af:
            track["audio_embedding"] = compute_audio_embedding(af)
            track.update(af)

        # Fetch genres from artist
        artist_id = track.get("artist_id")
        if artist_id:
            try:
                info = client.get_artist_info(artist_id)
                track["genres"] = info.get("genres", [])
            except Exception:
                track["genres"] = []

        track["text_embedding"] = compute_text_embedding(
            track.get("title", ""), track.get("artist", ""), track.get("genres", []))

        # Merge history signals
        hs = history_stats.get(tid)
        if hs:
            enriched = compute_play_stats(hs, track.get("duration_ms", 1))
            track.update({
                "play_count": enriched["play_count"],
                "total_ms_played": enriched["total_ms_played"],
                "completion_rate": enriched["completion_rate"],
                "skip_rate": enriched["skip_rate"],
                "first_played_at": enriched.get("first_played_at"),
                "last_played_at": enriched.get("last_played_at"),
            })
        track["is_new_discovery"] = track.get("play_count", 0) == 0

        upsert_track(dataset, track)
        print(f"  ✓ {track.get('title', tid)} — {track.get('artist', '')}")


def main():
    parser = argparse.ArgumentParser(description="Index Spotify tracks into FiftyOne")
    parser.add_argument("--liked", action="store_true", help="Index liked songs")
    parser.add_argument("--playlists", action="store_true", help="Index all user playlists")
    parser.add_argument("--recent", action="store_true", help="Index recently played (last 50)")
    parser.add_argument("--history", action="store_true", help="Import Spotify data export from history/")
    parser.add_argument("--resume", action="store_true", help="Skip tracks already in dataset")
    args = parser.parse_args()

    dataset = get_or_create_dataset()
    history_stats: dict = {}

    if args.history:
        print("Parsing listening history export...")
        history_stats = parse_history_files(HISTORY_DIR)
        print(f"  Found {len(history_stats)} unique tracks in history")

    if args.liked or args.playlists or args.recent:
        client = SpotifyClient(user_auth=True)
    else:
        client = SpotifyClient(user_auth=False)

    existing_ids: set[str] = set()
    if args.resume:
        existing_ids = {s["track_id"] for s in dataset if s.get("track_id")}
        print(f"Resume mode: {len(existing_ids)} tracks already indexed")

    if args.liked:
        print("Fetching liked songs...")
        tracks = client.liked_tracks()
        if args.resume:
            tracks = [t for t in tracks if t["track_id"] not in existing_ids]
        print(f"  Indexing {len(tracks)} liked tracks...")
        enrich_and_upsert(dataset, tracks, client, history_stats, "liked")

    if args.playlists:
        print("Fetching playlists...")
        for pl in client.user_playlists():
            print(f"  Playlist: {pl['name']}")
            tracks = client.playlist_tracks(pl["id"])
            if args.resume:
                tracks = [t for t in tracks if t["track_id"] not in existing_ids]
            enrich_and_upsert(dataset, tracks, client, history_stats, "playlist")

    if args.recent:
        print("Fetching recently played...")
        tracks = client.recently_played()
        if args.resume:
            tracks = [t for t in tracks if t["track_id"] not in existing_ids]
        enrich_and_upsert(dataset, tracks, client, history_stats, "recent")

    if args.history and not (args.liked or args.playlists or args.recent):
        # History-only: upsert tracks that are only in history (no Spotify API metadata yet)
        print("Upserting history-only tracks (metadata will be minimal)...")
        catalog_client = SpotifyClient(user_auth=False)
        ids = list(history_stats.keys())
        features_map = catalog_client.fetch_audio_features(ids)
        for tid, hs in history_stats.items():
            if args.resume and tid in existing_ids:
                continue
            track = {"track_id": tid, "title": hs["title"], "artist": hs["artist"],
                     "album": hs["album"], "source": "history"}
            af = features_map.get(tid, {})
            if af:
                track["audio_embedding"] = compute_audio_embedding(af)
                track.update(af)
            track["text_embedding"] = compute_text_embedding(hs["title"], hs["artist"], [])
            enriched = compute_play_stats(hs, af.get("duration_ms", 1) if af else 1)
            track.update({k: enriched[k] for k in ("play_count", "total_ms_played",
                           "completion_rate", "skip_rate", "first_played_at", "last_played_at")})
            track["is_new_discovery"] = False
            upsert_track(dataset, track)

    print(f"\nDone. Dataset '{dataset.name}' now has {len(dataset)} samples.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test (requires Spotify credentials in .env)**

```bash
# Test catalog-only (no user auth needed)
python index.py --history
```

Expected: prints "Parsing listening history export..." (or warns if history/ is empty). No crash.

- [ ] **Step 3: Commit**

```bash
git add index.py
git commit -m "feat: index pipeline — ingest liked/playlists/recent/history into FiftyOne"
```

---

## Task 10: agent.py

**Files:**
- Create: `agent.py`

- [ ] **Step 1: Implement agent.py**

```python
# agent.py
import argparse
import json
import anthropic
from agent_tools import TOOLS, execute_tool
from config import ANTHROPIC_API_KEY

SYSTEM_PROMPT = """You are an expert music curator. You build playlists in three modes:

**Sonic mode**: Find tracks by audio feature similarity across genres. Use search_catalog, fetch_audio_features, embed_tracks, filter_by_features, query_dataset, get_listening_history, rank_and_select, open_in_fiftyone.

**Connection mode**: Map every artist connected to a seed (band members, side projects, collabs) via MusicBrainz. Use map_artist_connections, traverse_artist_graph, fetch_artist_top_tracks, then rank_and_select, open_in_fiftyone. Narrate connections in plain English.

**Bateman mode**: Deep-dive a single track's creative DNA — composers, producers, session musicians, samples. Use deep_dive_track, find_tracks_by_person, resolve_samples, then rank_and_select, open_in_fiftyone. Introduce every artist with their connection path.

Rules:
- Always end by calling open_in_fiftyone with the final track IDs
- For new discoveries (is_new_discovery=True), write a one-sentence introduction
- Narrate your reasoning as you go
- Never explain what tools do — just use them and describe results"""


def run_agent(prompt: str) -> None:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    messages = [{"role": "user", "content": prompt}]
    print(f"\nPrompt: {prompt}\n{'─' * 60}")

    while True:
        response = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=8192,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        for block in response.content:
            if hasattr(block, "text") and block.text:
                print(f"\n{block.text}")

        if response.stop_reason == "end_turn":
            break

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"\n→ {block.name}({json.dumps(block.input)[:120]}...)")
                    result = execute_tool(block.name, block.input)
                    preview = str(result)[:200]
                    print(f"  ← {preview}{'...' if len(str(result)) > 200 else ''}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    })
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
        else:
            break


def build_prompt(args: argparse.Namespace) -> str:
    parts = []

    if args.bateman:
        parts.append(f"Bateman mode: deep-dive the track '{args.bateman}'. Find all composers, producers, session musicians, and samples. Then find those people's other work and the original sampled artists. Build a playlist from the complete creative DNA.")
        if args.depth:
            parts.append(f"Follow connections up to {args.depth} hops.")

    elif args.artist and args.connections:
        parts.append(f"Connection mode: build a playlist of every artist associated with '{args.artist}'.")
        parts.append(f"Traverse up to depth {args.depth}.")
        if args.new_only:
            parts.append("Prefer artists not in my listening history (is_new_discovery=True).")
        if args.match_sound:
            parts.append("Also filter by audio similarity to the seed artist's sound.")

    elif args.connect:
        parts.append(f"Show the connection path between '{args.connect[0]}' and '{args.connect[1]}', then build a playlist that bridges both artists' constellations.")

    if args.seed:
        seeds = ", ".join(f'"{s}"' for s in args.seed)
        parts.append(f"Use these as seed tracks or artists: {seeds}. Expand by similarity.")

    if args.prompt:
        parts.append(args.prompt)

    for constraint, label in [
        (args.energy, "energy"), (args.valence, "valence"), (args.tempo, "tempo BPM")
    ]:
        if constraint:
            lo, hi = constraint.split("-")
            parts.append(f"{label} between {lo} and {hi}")

    parts.append(f"Target playlist length: {args.count} tracks.")
    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="AI Playlist Agent")
    parser.add_argument("prompt", nargs="?", default="", help="Natural language playlist description")
    parser.add_argument("--bateman", metavar="TRACK", help="Deep-dive a track's creative DNA")
    parser.add_argument("--artist", help="Seed artist name for connection mode")
    parser.add_argument("--connections", action="store_true", help="Build from artist connection graph")
    parser.add_argument("--depth", type=int, default=2, help="Connection/Bateman traversal depth")
    parser.add_argument("--new-only", action="store_true", dest="new_only", help="Discovery artists only")
    parser.add_argument("--match-sound", action="store_true", dest="match_sound", help="Filter by audio similarity")
    parser.add_argument("--connect", nargs=2, metavar=("ARTIST1", "ARTIST2"))
    parser.add_argument("--seed", nargs="+", help="Seed tracks or artists")
    parser.add_argument("--energy", help="Energy range e.g. 0.7-1.0")
    parser.add_argument("--valence", help="Valence range e.g. 0.3-0.6")
    parser.add_argument("--tempo", help="Tempo range e.g. 120-180")
    parser.add_argument("--count", type=int, default=20, help="Target playlist length")
    args = parser.parse_args()

    prompt = build_prompt(args)
    if not prompt.strip():
        parser.print_help()
        return

    run_agent(prompt)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test (requires ANTHROPIC_API_KEY in .env)**

```bash
python agent.py "find 5 energetic instrumental tracks" --count 5
```

Expected: Agent narrates tool calls, FiftyOne browser opens with 5 tagged tracks.

- [ ] **Step 3: Commit**

```bash
git add agent.py
git commit -m "feat: agent — Claude tool-use loop, all invocation modes (sonic/connection/bateman)"
```

---

## Task 11: push.py

**Files:**
- Create: `push.py`

- [ ] **Step 1: Implement push.py**

```python
# push.py
import argparse
from spotify_client import SpotifyClient
from fiftyone_store import get_or_create_dataset, get_tagged_playlist


def main():
    parser = argparse.ArgumentParser(description="Push FiftyOne-tagged playlist to Spotify")
    parser.add_argument("--name", required=True, help="Playlist name in Spotify")
    parser.add_argument("--public", action="store_true", help="Make playlist public")
    parser.add_argument("--tag", default="playlist", help="FiftyOne tag to push (default: playlist)")
    args = parser.parse_args()

    dataset = get_or_create_dataset()
    track_ids = get_tagged_playlist(dataset, args.tag)

    if not track_ids:
        print(f"No tracks tagged '{args.tag}' in the dataset. Run agent.py first.")
        return

    print(f"Pushing {len(track_ids)} tracks to Spotify as '{args.name}'...")
    client = SpotifyClient(user_auth=True)
    url = client.create_playlist(args.name, track_ids, public=args.public)
    print(f"Playlist created: {url}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test (requires user OAuth)**

```bash
# First run the agent to tag some tracks, then:
python push.py --name "Test AI Playlist" --tag playlist
```

Expected: prints the Spotify playlist URL.

- [ ] **Step 3: Run full test suite**

```bash
pytest tests/ -v --ignore=tests/test_fiftyone_store.py
```

Expected: all non-integration tests pass.

- [ ] **Step 4: Commit**

```bash
git add push.py
git commit -m "feat: push — send FiftyOne-tagged playlist to Spotify"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|----------------|------|
| FiftyOne dataset with all schema fields | Task 5 (upsert_track carries all fields) |
| Audio feature embeddings (13D) | Task 4 |
| Text embeddings (sentence-transformers) | Task 4 |
| UMAP 2D projection | Task 4 (fit_umap / project_umap defined, called from index.py batch) |
| Spotify catalog search | Task 2 |
| Audio features API | Task 2 |
| Artist top tracks | Task 2 |
| User liked songs / playlists / recent | Task 2 |
| History export parse | Task 3 |
| play_count, completion_rate, skip_rate, first/last_played_at | Task 3 |
| MusicBrainz artist graph traversal | Task 6 |
| Graph cache (JSON) | Task 6 |
| explain_path narration | Task 6 |
| Track DNA: composers, producers, session musicians, samples | Task 7 |
| Bateman expand (depth traversal) | Task 7 |
| All 15 agent tools | Task 8 |
| Sonic mode invocation | Task 10 |
| Connection mode invocation | Task 10 |
| Bateman mode invocation | Task 10 |
| --connect bridging mode | Task 10 |
| index.py with all flags | Task 9 |
| push.py | Task 11 |
| is_new_discovery field | Task 8 (embed_tracks tool) |
| connection_path / bateman_path fields | Task 8 (fetch_artist_top_tracks / resolve_samples) |

**Gap identified:** UMAP is defined in Task 4 but never called in the pipeline. Fix: add a UMAP recompute step at the end of `index.py` when `len(dataset) >= 10`. Add this to Task 9's `main()`:

```python
# At end of index.py main(), after all upserts:
if len(dataset) >= 10:
    from embeddings import fit_umap, project_umap
    print("Computing UMAP projection...")
    samples_with_emb = [s for s in dataset if s.get("audio_embedding")]
    embs = [s["audio_embedding"] for s in samples_with_emb]
    reducer = fit_umap(embs)
    projections = project_umap(reducer, embs)
    for sample, proj in zip(samples_with_emb, projections):
        sample["embedding"] = proj
        sample.save()
    print(f"  UMAP computed for {len(samples_with_emb)} tracks.")
```

**Placeholder scan:** None found.

**Type consistency:** `AUDIO_FEATURE_KEYS` is defined in both `spotify_client.py` and `embeddings.py`. They must match. Both define the same 13 keys in the same order — confirmed.

**`_IDX` dict** in embeddings.py is used for normalization by key name — consistent with `AUDIO_FEATURE_KEYS` order.

**`compute_play_stats`** signature is `(stats: dict, duration_ms: int)` in both the test and implementation — consistent.

**Tool count:** TOOLS list has exactly 15 entries matching the spec's 8 sonic + 4 connection + 3 Bateman tools.
