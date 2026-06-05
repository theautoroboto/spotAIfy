"""
Builds a per-track listening profile from Spotify's Extended Streaming History
export (data/history/Streaming_History_Audio_*.json).

Signals computed per track:
  completion_rate  — fraction of plays where reason_end == "trackdone"
  skip_rate        — fraction of plays where skipped == True

Artist-level signal:
  is_new_discovery — True when a candidate's artist has never appeared in history

All data is loaded lazily on first use and held in memory for the session.
"""

from __future__ import annotations
import glob
import json
from pathlib import Path

_HISTORY_DIR = Path(__file__).parent.parent.parent / "data" / "history"
_GLOB = "Streaming_History_Audio_*.json"

_profile: dict[str, dict] | None = None   # track_id → stats
_known_artists: set[str] | None = None


def _load() -> None:
    global _profile, _known_artists
    track_stats: dict[str, dict] = {}
    artists: set[str] = set()

    for path in sorted(_HISTORY_DIR.glob(_GLOB)):
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        for r in records:
            uri = r.get("spotify_track_uri") or ""
            if not uri.startswith("spotify:track:"):
                continue
            track_id = uri.split(":")[-1]

            artist = (r.get("master_metadata_album_artist_name") or "").strip()
            if artist:
                artists.add(artist.lower())

            stats = track_stats.setdefault(track_id, {
                "play_count": 0, "completion_count": 0, "skip_count": 0
            })
            stats["play_count"] += 1
            if r.get("reason_end") == "trackdone":
                stats["completion_count"] += 1
            if r.get("skipped"):
                stats["skip_count"] += 1

    # Collapse counts → rates
    for stats in track_stats.values():
        n = stats["play_count"]
        stats["completion_rate"] = stats["completion_count"] / n
        stats["skip_rate"] = stats["skip_count"] / n

    _profile = track_stats
    _known_artists = artists


def _get_profile() -> tuple[dict[str, dict], set[str]]:
    if _profile is None:
        _load()
    return _profile, _known_artists  # type: ignore[return-value]


def enrich_candidates(candidates: list[dict]) -> None:
    """
    Annotate each candidate dict with completion_rate, skip_rate, and
    is_new_discovery derived from the local Spotify history export.

    Mutates candidates in place; safe to call even if history files are missing
    (signals remain at their existing values or defaults).
    """
    try:
        profile, known_artists = _get_profile()
    except Exception:
        return

    for c in candidates:
        track_id = c.get("track_id", "")
        artist = (c.get("artist") or "").strip().lower()

        stats = profile.get(track_id)
        if stats:
            c["completion_rate"] = stats["completion_rate"]
            c["skip_rate"] = stats["skip_rate"]
            c.setdefault("is_new_discovery", False)
        else:
            c.setdefault("completion_rate", 0.0)
            c.setdefault("skip_rate", 0.0)
            c["is_new_discovery"] = artist not in known_artists if artist else False
