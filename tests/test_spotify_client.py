# tests/test_spotify_client.py
import os
from unittest.mock import MagicMock, patch
import pytest
from spotaify.clients.spotify_client import SpotifyClient, AUDIO_FEATURE_KEYS

@pytest.fixture(autouse=True)
def mock_config(mocker):
    """Mock config values to avoid environment dependency during tests."""
    mocker.patch("spotaify.clients.spotify_client.SPOTIFY_CLIENT_ID", "test_client_id")
    mocker.patch("spotaify.clients.spotify_client.SPOTIFY_CLIENT_SECRET", "test_client_secret")

@pytest.fixture
def mock_spotipy(mocker):
    return mocker.patch("spotaify.clients.spotify_client.spotipy.Spotify")

def test_search_catalog_returns_parsed_tracks(mock_spotipy):
    client = SpotifyClient(user_auth=False)
    client._sp.search.return_value = {
        "tracks": {"items": [{
            "id": "abc123",
            "name": "Hurt",
            "artists": [{"id": "art1", "name": "Nine Inch Nails"}],
            "album": {"name": "The Downward Spiral", "release_date": "1994-03-08"},
            "duration_ms": 237000,
            "explicit": False,
            "preview_url": "https://preview.url",
            "popularity": 82,
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