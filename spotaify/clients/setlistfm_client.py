# setlistfm_client.py
import time
import requests
from spotaify.config import SETLISTFM_API_KEY

_BASE = "https://api.setlist.fm/rest/1.0"
_DELAY = 0.5  # be polite to the API


def _headers() -> dict:
    return {
        "x-api-key": SETLISTFM_API_KEY,
        "Accept": "application/json",
    }


def _get(path: str, params: dict | None = None) -> dict:
    time.sleep(_DELAY)
    resp = requests.get(f"{_BASE}{path}", headers=_headers(), params=params or {}, timeout=15)
    resp.raise_for_status()
    return resp.json()


def search_artist(name: str) -> dict | None:
    """Return {mbid, name} for the best-matching artist on setlist.fm."""
    if not SETLISTFM_API_KEY:
        raise RuntimeError("SETLISTFM_API_KEY not configured — add it to .env or keyring")
    data = _get("/search/artists", {"artistName": name, "sort": "relevance"})
    items = data.get("artist", [])
    if not items:
        return None
    a = items[0]
    return {"mbid": a["mbid"], "name": a["name"]}


def get_setlist_songs(mbid: str, pages: int = 3) -> list[dict]:
    """
    Fetch up to `pages` pages of setlists and return songs ranked by live
    frequency: [{title, artist, appearances, last_date}].
    Tape tracks (pre-recorded playback) are excluded.
    """
    song_counts: dict[str, dict] = {}

    for p in range(1, pages + 1):
        try:
            data = _get(f"/artist/{mbid}/setlists", {"p": p})
        except Exception:
            break
        setlists = data.get("setlist", [])
        if not setlists:
            break

        for sl in setlists:
            date = sl.get("eventDate", "")
            artist_name = sl.get("artist", {}).get("name", "")
            for set_block in sl.get("sets", {}).get("set", []):
                for song in set_block.get("song", []):
                    if song.get("tape"):
                        continue  # skip pre-recorded intros / outros
                    title = song.get("name", "").strip()
                    if not title:
                        continue
                    key = title.lower()
                    if key not in song_counts:
                        song_counts[key] = {
                            "title": title,
                            "artist": artist_name,
                            "appearances": 0,
                            "last_date": "",
                        }
                    song_counts[key]["appearances"] += 1
                    if date > song_counts[key]["last_date"]:
                        song_counts[key]["last_date"] = date

    return sorted(song_counts.values(), key=lambda x: -x["appearances"])
