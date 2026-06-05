# tests/test_track_dna.py
import pytest
from unittest.mock import patch
from spotaify.core.track_dna import resolve_recording_mbid, fetch_recording_credits, expand_bateman

RECORDING_MBID = "b52a8f8a-aaaa-bbbb-cccc-123456789abc"

@pytest.fixture
def mock_mb(mocker):
    return mocker.patch("spotaify.core.track_dna.mb")

def test_resolve_recording_mbid_returns_id(mock_mb):
    mock_mb.search_recordings.return_value = {
        "recording-list": [{"id": RECORDING_MBID, "title": "Hurt"}]
    }
    result = resolve_recording_mbid("Hurt", "Nine Inch Nails")
    assert result == RECORDING_MBID

def test_resolve_recording_mbid_returns_none_when_not_found(mock_mb):
    mock_mb.search_recordings.return_value = {"recording-list": []}
    assert resolve_recording_mbid("xyznonexistent", "nobody") is None

def test_fetch_recording_credits_extracts_composers(mock_mb):
    mock_mb.get_recording_by_id.return_value = {
        "recording": {
            "title": "Hurt",
            "artist-relation-list": [
                {"type": "composer", "artist": {"id": "c1", "name": "Trent Reznor"}, "attribute-list": []},
                {"type": "performer", "artist": {"id": "c2", "name": "Nine Inch Nails"}, "attribute-list": []},
            ],
            "recording-relation-list": [],
            "work-relation-list": [],
        }
    }
    credits = fetch_recording_credits(RECORDING_MBID)
    assert "Trent Reznor" in credits["composers"]
    assert credits["producers"] == []

def test_fetch_recording_credits_extracts_samples(mock_mb):
    mock_mb.get_recording_by_id.return_value = {
        "recording": {
            "title": "Fight the Power",
            "artist-relation-list": [],
            "recording-relation-list": [
                {
                    "type": "samples material from",
                    "recording": {"id": "orig123", "title": "Funky Drummer",
                                  "artist-credit": [{"artist": {"name": "James Brown"}}]},
                }
            ],
            "work-relation-list": [],
        }
    }
    credits = fetch_recording_credits(RECORDING_MBID)
    assert len(credits["samples"]) == 1
    assert credits["samples"][0]["title"] == "Funky Drummer"
    assert credits["samples"][0]["artist"] == "James Brown"

def test_expand_bateman_returns_artist_names(mock_mb):
    mock_mb.search_recordings.return_value = {"recording-list": [{"id": RECORDING_MBID, "title": "Hurt"}]}
    mock_mb.get_recording_by_id.return_value = {
        "recording": {
            "title": "Hurt",
            "artist-relation-list": [
                {"type": "composer", "artist": {"id": "c1", "name": "Trent Reznor"}, "attribute-list": []},
            ],
            "recording-relation-list": [],
            "work-relation-list": [],
        }
    }
    mock_mb.search_artists.return_value = {"artist-list": [{"id": "c1", "name": "Trent Reznor"}]}
    mock_mb.get_artist_by_id.return_value = {"artist": {"name": "Trent Reznor", "artist-relation-list": []}}
    result = expand_bateman("Hurt", "Nine Inch Nails", depth=1)
    assert "persons" in result
    assert any(p["name"] == "Trent Reznor" for p in result["persons"])