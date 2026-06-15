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
    if stats["play_count"] == 0 or duration_ms <= 0:
        return {**stats, "completion_rate": 0.0, "skip_rate": 0.0}
    avg_ms = stats["total_ms_played"] / stats["play_count"]
    return {
        **stats,
        "completion_rate": min(avg_ms / duration_ms, 1.0),
        "skip_rate": stats["skip_count"] / stats["play_count"],
    }