# tests/test_artist_graph.py
import json
import pytest
from unittest.mock import patch, MagicMock
from spotaify.core.artist_graph import (
    resolve_artist_mbid, fetch_artist_relations,
    traverse_graph, get_all_artist_names, explain_path,
)

NIN_MBID = "b7ffd2af-418f-4be2-bdd1-22f8b48613da"
REZNOR_MBID = "1f9df192-a621-4f54-8850-2c5373b7eac9"

@pytest.fixture
def mock_mb(mocker):
    return mocker.patch("spotaify.core.artist_graph.mb")

def test_resolve_artist_mbid_returns_id(mock_mb):
    mock_mb.search_artists.return_value = {
        "artist-list": [{"id": NIN_MBID, "name": "Nine Inch Nails"}]
    }
    result = resolve_artist_mbid("Nine Inch Nails")
    assert result == NIN_MBID

def test_resolve_artist_mbid_returns_none_when_not_found(mock_mb):
    mock_mb.search_artists.return_value = {"artist-list": []}
    assert resolve_artist_mbid("xyznonexistentband") is None

def test_fetch_artist_relations_extracts_members(mock_mb):
    mock_mb.get_artist_by_id.return_value = {
        "artist": {
            "name": "Nine Inch Nails",
            "artist-relation-list": [{
                "type": "member of band",
                "direction": "backward",
                "artist": {"id": REZNOR_MBID, "name": "Trent Reznor"},
                "begin": "1988", "end": "",
                "attribute-list": ["vocals"],
            }]
        }
    }
    relations = fetch_artist_relations(NIN_MBID)
    assert relations["name"] == "Nine Inch Nails"
    assert len(relations["relations"]) == 1
    assert relations["relations"][0]["related_name"] == "Trent Reznor"
    assert relations["relations"][0]["type"] == "member of band"

def test_traverse_graph_uses_cache(tmp_path, mock_mb):
    cached = {
        "seed": "Nine Inch Nails", "seed_mbid": NIN_MBID,
        "depth": 1,
        "nodes": {NIN_MBID: {"mbid": NIN_MBID, "name": "Nine Inch Nails", "type": "Group", "depth": 0}},
        "edges": [],
    }
    cache_file = tmp_path / "nine_inch_nails_d1.json"
    cache_file.write_text(json.dumps(cached))
    graph = traverse_graph("Nine Inch Nails", depth=1, graph_dir=str(tmp_path))
    mock_mb.search_artists.assert_not_called()
    assert graph["seed"] == "Nine Inch Nails"

def test_get_all_artist_names_excludes_seed(tmp_path, mock_mb):
    cached = {
        "seed": "Nine Inch Nails", "seed_mbid": NIN_MBID, "depth": 1,
        "nodes": {
            NIN_MBID: {"mbid": NIN_MBID, "name": "Nine Inch Nails", "type": "Group", "depth": 0},
            REZNOR_MBID: {"mbid": REZNOR_MBID, "name": "Trent Reznor", "type": "Person", "depth": 1},
        },
        "edges": [{"from": NIN_MBID, "to": REZNOR_MBID, "type": "member of band",
                   "attributes": ["vocals"], "begin": "1988", "end": ""}],
    }
    (tmp_path / "nine_inch_nails_d1.json").write_text(json.dumps(cached))
    graph = traverse_graph("Nine Inch Nails", depth=1, graph_dir=str(tmp_path))
    artists = get_all_artist_names(graph)
    names = [a["name"] for a in artists]
    assert "Nine Inch Nails" not in names
    assert "Trent Reznor" in names

def test_explain_path_generates_description(tmp_path, mock_mb):
    cached = {
        "seed": "Nine Inch Nails", "seed_mbid": NIN_MBID, "depth": 1,
        "nodes": {
            NIN_MBID: {"mbid": NIN_MBID, "name": "Nine Inch Nails", "type": "Group", "depth": 0},
            REZNOR_MBID: {"mbid": REZNOR_MBID, "name": "Trent Reznor", "type": "Person", "depth": 1},
        },
        "edges": [{"from": NIN_MBID, "to": REZNOR_MBID, "type": "member of band",
                   "attributes": ["vocals"], "begin": "1988", "end": ""}],
    }
    (tmp_path / "nine_inch_nails_d1.json").write_text(json.dumps(cached))
    graph = traverse_graph("Nine Inch Nails", depth=1, graph_dir=str(tmp_path))
    explanation = explain_path(graph, REZNOR_MBID)
    assert "Trent Reznor" in explanation
    assert "Nine Inch Nails" in explanation
