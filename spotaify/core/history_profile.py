"""
Builds a per-track listening profile from Spotify's Extended Streaming History
export (data/history/Streaming_History_Audio_*.json).

Signals computed per track:
  completion_rate  — fraction of plays where reason_end == "trackdone"
  skip_rate        — fraction of plays where skipped == True
  ms_played_total  — total milliseconds listened across all plays

Artist-level signal:
  is_new_discovery — True when a candidate's artist has never appeared in history

All data is loaded lazily on first use and held in memory for the session.
"""

from __future__ import annotations
from datetime import datetime, timedelta, timezone
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
                "play_count": 0, "completion_count": 0, "skip_count": 0,
                "ms_played_total": 0, "last_played_ts": "",
            })
            stats["play_count"] += 1
            stats["ms_played_total"] += r.get("ms_played") or 0
            ts = r.get("ts") or ""
            if ts > stats["last_played_ts"]:
                stats["last_played_ts"] = ts
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


def get_top_track_ids(n: int = 100) -> list[str]:
    """Return the top-N track IDs ranked by total milliseconds played."""
    profile, _ = _get_profile()
    ranked = sorted(profile.items(), key=lambda kv: -kv[1]["ms_played_total"])
    return [tid for tid, _ in ranked[:n]]


def get_forgotten_favorites(
    min_plays: int = 3,
    min_completion: float = 0.4,
    stale_days: int = 90,
) -> list[str]:
    """
    Return track IDs the user used to love but hasn't played recently.

    Criteria: play_count >= min_plays AND completion_rate >= min_completion
    AND last played more than stale_days ago.
    Sorted by ms_played_total descending (most-loved first).
    """
    profile, _ = _get_profile()
    cutoff_ts = (
        datetime.now(timezone.utc) - timedelta(days=stale_days)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    candidates = [
        (tid, stats)
        for tid, stats in profile.items()
        if (stats["play_count"] >= min_plays
            and stats["completion_rate"] >= min_completion
            and stats.get("last_played_ts", "") < cutoff_ts)
    ]
    candidates.sort(key=lambda kv: -kv[1]["ms_played_total"])
    return [tid for tid, _ in candidates]


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
            if artist:
                c["is_new_discovery"] = artist not in known_artists
            else:
                c.setdefault("is_new_discovery", False)
