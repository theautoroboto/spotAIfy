# tests/test_history_importer.py
import json
import pytest
from pathlib import Path
from spotaify.core.history_importer import parse_history_files, compute_play_stats

FIXTURE_DIR = Path(__file__).parent

@pytest.fixture
def history_files(tmp_path):
    """Copies the fixture file to a temporary directory for testing."""
    fixture_path = FIXTURE_DIR / "streaming_history.json"
    target_path = tmp_path / "endsong_0.json"
    target_path.write_text(fixture_path.read_text())
    return str(tmp_path)

def test_parse_history_aggregates_play_count(history_files):
    stats = parse_history_files(history_files)
    assert "6eUW0wxWtzkFdaEFsTJto6" in stats
    assert stats["6eUW0wxWtzkFdaEFsTJto6"]["play_count"] == 2

def test_parse_history_tracks_skip_count(history_files):
    stats = parse_history_files(history_files)
    assert stats["6eUW0wxWtzkFdaEFsTJto6"]["skip_count"] == 1

def test_parse_history_first_and_last_played(history_files):
    stats = parse_history_files(history_files)
    s = stats["6eUW0wxWtzkFdaEFsTJto6"]
    assert s["first_played_at"].year == 2022
    assert s["last_played_at"].year == 2023

def test_parse_history_ignores_podcasts(tmp_path):
    data = [{"ts": "2023-01-01T00:00:00Z", "ms_played": 1000,
             "spotify_episode_uri": "spotify:episode:abc", "reason_end": "trackdone"}]
    (tmp_path / "endsong_0.json").write_text(json.dumps(data))
    stats = parse_history_files(str(tmp_path))
    assert len(stats) == 0

def test_compute_play_stats_completion_rate():
    raw = {"play_count": 2, "total_ms_played": 355000, "skip_count": 1}
    result = compute_play_stats(raw, duration_ms=325000)
    avg_ms_played = 355000 / 2
    assert result["completion_rate"] == pytest.approx(avg_ms_played / 325000, abs=0.01)
    assert result["skip_rate"] == pytest.approx(0.5)

def test_compute_play_stats_caps_completion_at_1():
    raw = {"play_count": 1, "total_ms_played": 999999, "skip_count": 0}
    result = compute_play_stats(raw, duration_ms=180000)
    assert result["completion_rate"] == 1.0