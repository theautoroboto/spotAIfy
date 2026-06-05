# genius_client.py
import re
import time
import requests
from spotaify.config import GENIUS_ACCESS_TOKEN

_BASE = "https://api.genius.com"
_DELAY = 1.0
_ABOUT_MAX_CHARS = 800


def _headers() -> dict:
    return {"Authorization": f"Bearer {GENIUS_ACCESS_TOKEN}"}


def _get(path: str, params: dict | None = None) -> dict | None:
    time.sleep(_DELAY)
    try:
        resp = requests.get(
            f"{_BASE}{path}",
            headers=_headers(),
            params=params or {},
            timeout=15,
        )
        if resp.status_code == 429:
            retry = resp.headers.get("Retry-After", "?")
            raise RuntimeError(f"Genius rate limit — retry after {retry}s")
        if resp.status_code == 200:
            return resp.json()
    except RuntimeError:
        raise
    except requests.RequestException:
        pass
    return None


def _normalize(text: str) -> str:
    """Lowercase and strip punctuation for fuzzy comparison."""
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def _artist_matches(query_artist: str, hit_artist: str) -> bool:
    """True if any token from a multi-artist query string appears in the hit artist name."""
    q = _normalize(query_artist)
    h = _normalize(hit_artist)
    if q in h or h in q:
        return True
    for token in re.split(r"[,&]+", q):
        token = token.strip()
        if token and token in h:
            return True
    return False


def _find_song_id(title: str, artist: str) -> tuple[int | None, str]:
    """Return (genius_song_id, genius_url) for the best matching hit."""
    data = _get("/search", params={"q": f"{title} {artist}"})
    if not data:
        return None, ""

    hits = data.get("response", {}).get("hits", [])
    title_norm = _normalize(title)

    # First pass: match on both title and primary artist
    for hit in hits:
        result = hit.get("result", {})
        hit_title = _normalize(result.get("title", ""))
        hit_artist = result.get("primary_artist", {}).get("name", "")
        if title_norm in hit_title and _artist_matches(artist, hit_artist):
            return result["id"], result.get("url", "")

    # Second pass: title match only (handles alternate artist spellings on Genius)
    for hit in hits:
        result = hit.get("result", {})
        hit_title = _normalize(result.get("title", ""))
        if title_norm in hit_title:
            return result["id"], result.get("url", "")

    return None, ""


def fetch_song_info(title: str, artist: str) -> dict:
    """
    Fetch Genius metadata for a track.

    Returns:
        about      — plain-text "About" description (capped at 800 chars)
        url        — Genius song page URL
        writers    — list of writer names from Genius credits
        producers  — list of producer names from Genius credits
        release_date
    """
    if not GENIUS_ACCESS_TOKEN:
        return {"error": "GENIUS_ACCESS_TOKEN not set in .env", "about": "", "url": ""}

    song_id, genius_url = _find_song_id(title, artist)
    if not song_id:
        return {"error": "Song not found on Genius", "about": "", "url": ""}

    detail = _get(f"/songs/{song_id}", params={"text_format": "plain"})
    if not detail:
        return {"error": "Could not fetch song detail from Genius", "about": "", "url": genius_url}

    song = detail.get("response", {}).get("song", {})
    about_raw = song.get("description", {}).get("plain", "").strip()
    about = about_raw[:_ABOUT_MAX_CHARS] if about_raw else ""

    writers = [a.get("name", "") for a in song.get("writer_artists", []) if a.get("name")]
    producers = [a.get("name", "") for a in song.get("producer_artists", []) if a.get("name")]

    return {
        "about": about,
        "url": genius_url,
        "writers": writers,
        "producers": producers,
        "release_date": song.get("release_date_for_display", ""),
    }
