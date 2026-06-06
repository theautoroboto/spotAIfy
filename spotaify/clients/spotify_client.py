# spotify_client.py
import os
import spotipy
from spotipy.exceptions import SpotifyException
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
from spotaify.config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REDIRECT_URI


def _guard(fn):
    """Raise a labelled RuntimeError on Spotify 429 instead of a raw SpotifyException."""
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except SpotifyException as e:
            if e.http_status == 429:
                retry = (e.headers or {}).get("Retry-After", "?")
                raise RuntimeError(f"Spotify rate limit — retry after {retry}s") from e
            raise
    return wrapper

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
                cache_path=os.getenv("SPOTIFY_CACHE_PATH", ".spotify_token_cache"),
            ))
        else:
            self._sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
                client_id=SPOTIFY_CLIENT_ID,
                client_secret=SPOTIFY_CLIENT_SECRET,
            ))

    @_guard
    def search_catalog(self, query: str, limit: int = 10, market: str = "US") -> list[dict]:
        results = self._sp.search(q=query, type="track", limit=min(limit, 10), market=market)
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

    def get_tracks_metadata(self, track_ids: list[str]) -> dict[str, dict]:
        """Fetch track metadata for up to 500 IDs. Returns dict keyed by track_id."""
        result = {}
        for i in range(0, len(track_ids), 50):
            batch = track_ids[i : i + 50]
            try:
                resp = self._sp.tracks(batch)
                for t in (resp.get("tracks") or []):
                    if t:
                        result[t["id"]] = self._parse_track(t)
            except Exception:
                pass
        return result

    def get_artist_info(self, artist_id: str) -> dict:
        a = self._sp.artist(artist_id)
        return {
            "artist_id": a["id"],
            "genres": a.get("genres", []),
            "popularity": a.get("popularity", 0),
        }

    def get_artist_top_tracks(self, artist_id: str, market: str = "US") -> list[dict]:
        # The /top-tracks endpoint requires user OAuth; use search instead.
        try:
            info = self._sp.artist(artist_id)
            name = info.get("name", "")
            if name:
                r = self._sp.search(q=f'artist:"{name}"', type="track", limit=10, market=market)
                return [self._parse_track(t) for t in r["tracks"]["items"]]
        except Exception:
            pass
        return []

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

    def create_playlist(self, name: str, track_ids: list[str], public: bool = False, description: str = "") -> str:
        playlist = self._sp.current_user_playlist_create(name, public=public, description=description[:300])
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