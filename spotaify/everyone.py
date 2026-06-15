"""Builds the "Everyone's Top Songs" playlist.

Combines the most-played tracks from every user who has imported a Spotify
Extended Streaming History export (a subdirectory of HISTORY_DIR containing
Streaming_History_Audio_*.json files) into one shared playlist on the
logged-in user's account.

Merge strategy: interleaved round-robin with dedupe + backfill. Users take
turns contributing their next-ranked track; a track already added by someone
else is skipped and that user's next-ranked track is used instead, so each
user contributes exactly `per_user` unique tracks (fewer only when their
history is too small).

Deterministic — no AI involved. Output lines follow the agent's `-> ` /
`  <- ` convention so the web UI streams and renders them identically,
including the final "Done — {url}" playlist link.
"""

from datetime import datetime
from pathlib import Path

from spotaify.clients.spotify_client import SpotifyClient
from spotaify.config import HISTORY_DIR
from spotaify.core.history_profile import top_tracks_for_dir

_GLOB = "Streaming_History_Audio_*.json"


def _users_with_history(base: Path) -> list[Path]:
    """Subdirectories of base that contain at least one history export file."""
    if not base.is_dir():
        return []
    return sorted(
        (d for d in base.iterdir() if d.is_dir() and any(d.glob(_GLOB))),
        key=lambda d: d.name.lower(),
    )


def _interleave(pools: dict[str, list[dict]], per_user: int) -> tuple[list[dict], dict[str, int]]:
    """Round-robin merge with dedupe + backfill.

    Returns the merged track list and a per-user count of contributed tracks.
    """
    seen: set[str] = set()
    cursors = {name: 0 for name in pools}
    contributed = {name: 0 for name in pools}
    merged: list[dict] = []

    progressed = True
    while progressed:
        progressed = False
        for name, pool in pools.items():
            if contributed[name] >= per_user:
                continue
            i = cursors[name]
            while i < len(pool) and pool[i]["track_id"] in seen:
                i += 1
            cursors[name] = i + 1
            if i >= len(pool):
                continue  # this user's history is exhausted
            track = pool[i]
            seen.add(track["track_id"])
            merged.append(track)
            contributed[name] += 1
            progressed = True
    return merged, contributed


def build_everyone_top(per_user: int = 100) -> None:
    base = Path(HISTORY_DIR)
    user_dirs = _users_with_history(base)
    if not user_dirs:
        print(f"[error] No imported listening history found under {base}", flush=True)
        return

    names = [d.name for d in user_dirs]
    print(f"Building a shared playlist from {len(names)} listeners: {', '.join(names)}.", flush=True)

    # Rank deeper than per_user so dedupe backfill has candidates to draw from:
    # in the worst case a user needs per_user picks plus one replacement for
    # every track another user claimed first.
    depth = per_user * (len(user_dirs) + 1)
    pools: dict[str, list[dict]] = {}
    for d in user_dirs:
        print(f"-> Ranking {d.name}'s most-played tracks", flush=True)
        pool = top_tracks_for_dir(d, n=depth)
        pools[d.name] = pool
        print(f"  <- {len(pool)} tracks ranked by listening time", flush=True)

    merged, contributed = _interleave(pools, per_user)
    if not merged:
        print("[error] No playable tracks found in any history export", flush=True)
        return

    counts = ", ".join(f"{name} {contributed[name]}" for name in names)
    print(f"Selected {len(merged)} unique tracks ({counts}), interleaved between listeners.", flush=True)
    for name in names:
        if contributed[name] < per_user:
            print(f"Note: {name}'s history only provided {contributed[name]} unique tracks.", flush=True)

    playlist_name = "Everyone's Top Songs — " + datetime.now().strftime("%b %Y")
    description = (
        f"The {per_user} most-played songs from each of: {', '.join(names)}. "
        "Built from imported Spotify listening history."
    )

    print(f'-> Creating playlist: "{playlist_name}"', flush=True)
    sp = SpotifyClient(user_auth=True)
    url = sp.create_playlist(
        playlist_name,
        [t["track_id"] for t in merged],
        public=False,
        description=description,
    )
    print(f"  <- Done — {url}", flush=True)
