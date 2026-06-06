"""
Builds two user taste signals by cross-referencing listening history with
the Spotify API:

  audio_fingerprint   mean audio-feature vector across the user's top-played tracks
                      (energy, valence, tempo, danceability, etc.)

  era_distribution    fraction of top-100 tracks by release decade
                      e.g. {"1990s": 0.31, "2000s": 0.24, ...}

Results are cached to data/cache/taste_profile.json so the batch of Spotify
API calls only happens once (or when the user explicitly requests a rebuild).
"""

from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

_CACHE_PATH = Path("data/cache/taste_profile.json")

_FLOAT_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]

_TOP_N = 100


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_cached() -> dict | None:
    """Return the cached taste profile, or None if not yet built."""
    try:
        return json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def build_taste_profile(spotify_client) -> dict:
    """
    Fetch audio features and release metadata for the user's top 100 played
    tracks, compute the mean audio-feature vector and decade distribution,
    cache to disk, and return the result.
    """
    from spotaify.core.history_profile import get_top_track_ids

    track_ids = get_top_track_ids(_TOP_N)
    if not track_ids:
        return {
            "error": "No listening history found",
            "fingerprint": {},
            "era_distribution": {},
        }

    features = spotify_client.fetch_audio_features(track_ids)
    metadata = spotify_client.get_tracks_metadata(track_ids)

    # Mean audio-feature vector over tracks that have feature data
    buckets: dict[str, list[float]] = {k: [] for k in _FLOAT_FEATURES}
    for tid in track_ids:
        f = features.get(tid)
        if not f:
            continue
        for k in _FLOAT_FEATURES:
            if k in f:
                buckets[k].append(f[k])

    fingerprint = {k: round(_mean(v), 4) for k, v in buckets.items()}

    # Era distribution weighted by track count in top-100
    decade_counts: dict[str, int] = {}
    for tid in track_ids:
        m = metadata.get(tid)
        if m and m.get("release_year"):
            decade = (m["release_year"] // 10) * 10
            label = f"{decade}s"
            decade_counts[label] = decade_counts.get(label, 0) + 1

    total = sum(decade_counts.values())
    era_distribution = (
        {k: round(v / total, 3) for k, v in sorted(decade_counts.items())}
        if total else {}
    )

    result = {
        "fingerprint": fingerprint,
        "era_distribution": era_distribution,
        "top_track_count": len(track_ids),
        "features_computed": len(features),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }

    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
