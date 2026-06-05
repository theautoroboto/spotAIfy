# tests/test_agent_tools.py
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from tests.conftest import TEST_TRACK, TEST_AUDIO_FEATURES
import spotaify.agent_tools as agent_tools

@pytest.fixture(autouse=True)
def mock_deps(mocker):
    # Reset lazy-loaded clients before each test to prevent state leakage
    agent_tools._spotify = None
    agent_tools._spotify_user = None

    mocker.patch("spotaify.agent_tools.SpotifyClient")
    mocker.patch("spotaify.agent_tools.get_or_create_dataset", create=True)
    mocker.patch("spotaify.agent_tools.upsert_track", create=True)
    mocker.patch("spotaify.agent_tools.tag_playlist", create=True)
    mocker.patch("spotaify.agent_tools.get_tagged_playlist", create=True)
    mocker.patch("spotaify.agent_tools.open_in_browser", create=True)
    mocker.patch("spotaify.agent_tools.traverse_graph")
    mocker.patch("spotaify.agent_tools.get_all_artist_names")
    mocker.patch("spotaify.agent_tools.expand_track_dna")

def test_tools_list_has_correct_count():
    assert len(agent_tools.TOOLS) == 12

def test_all_tools_have_required_fields():
    for tool in agent_tools.TOOLS:
        assert "name" in tool
        assert "description" in tool
        assert "input_schema" in tool

def test_execute_search_catalog():
    mock_instance = agent_tools.SpotifyClient.return_value
    mock_instance.search_catalog.return_value = [TEST_TRACK]
    result = agent_tools.execute_tool("search_catalog", {"query": "hurt"})
    assert result["count"] == 1
    assert result["tracks"][0]["track_id"] == TEST_TRACK["track_id"]
    mock_instance.search_catalog.assert_called_with("hurt", limit=50)


def test_execute_rank_and_select_limits_count():
    candidates = [
        {"track_id": f"t{i}", "audio_sim": float(i) / 10, "completion_rate": 0.8, "skip_rate": 0.1}
        for i in range(30)
    ]
    result = agent_tools.execute_tool("rank_and_select", {"candidates": candidates, "count": 5})
    assert len(result["playlist"]) == 5

def test_execute_rank_and_select_boosts_discovery():
    # known: 0.7 + 0.0*0.3 - 0.0*0.2 = 0.70
    # new:   0.6 + 0.0*0.3 - 0.0*0.2 + 0.15 = 0.75  (boost tips it above known)
    candidates = [
        {"track_id": "known", "audio_sim": 0.7, "completion_rate": 0.0, "skip_rate": 0.0, "is_new_discovery": False},
        {"track_id": "new", "audio_sim": 0.6, "completion_rate": 0.0, "skip_rate": 0.0, "is_new_discovery": True},
    ]
    result = agent_tools.execute_tool("rank_and_select", {"candidates": candidates, "count": 1, "boost_discovery": True})
    assert result["playlist"][0]["track_id"] == "new"

def test_execute_unknown_tool_returns_error():
    result = agent_tools.execute_tool("nonexistent_tool", {})
    assert "error" in result