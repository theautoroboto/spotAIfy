# tests/conftest.py
import pytest

TEST_TRACK = {
    "track_id": "6eUW0wxWtzkFdaEFsTJto6",
    "title": "Hurt",
    "artist": "Nine Inch Nails",
    "artist_id": "0X380XXpSNLICoudqnW6TVk",
    "album": "The Downward Spiral",
    "duration_ms": 237000,
    "release_year": 1994,
    "explicit": False,
    "preview_url": "https://p.scdn.co/mp3-preview/abc123",
    "popularity": 82,
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