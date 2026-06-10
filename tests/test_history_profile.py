# tests/test_history_profile.py
import json

from spotaify.core.history_profile import top_tracks_for_dir


def _record(track_id: str, ms: int, title: str = "T", artist: str = "A") -> dict:
    return {
        "spotify_track_uri": f"spotify:track:{track_id}",
        "master_metadata_track_name": title,
        "master_metadata_album_artist_name": artist,
        "ms_played": ms,
        "ts": "2024-01-01T00:00:00Z",
        "reason_end": "trackdone",
        "skipped": False,
    }


def _write(d, records, filename="Streaming_History_Audio_2024.json"):
    d.mkdir(parents=True, exist_ok=True)
    (d / filename).write_text(json.dumps(records), encoding="utf-8")


def test_ranks_by_total_ms_played(tmp_path):
    _write(tmp_path, [
        _record("aaa", 1000),
        _record("bbb", 5000),
        _record("aaa", 1500),  # aaa totals 2500 — still below bbb
        _record("ccc", 3000),
    ])

    top = top_tracks_for_dir(tmp_path)
    assert [t["track_id"] for t in top] == ["bbb", "ccc", "aaa"]
    assert top[2]["ms_played_total"] == 2500
    assert top[2]["play_count"] == 2


def test_respects_n_limit(tmp_path):
    _write(tmp_path, [_record(f"id{i}", (i + 1) * 100) for i in range(10)])

    top = top_tracks_for_dir(tmp_path, n=3)
    assert len(top) == 3
    assert top[0]["track_id"] == "id9"


def test_skips_non_track_records(tmp_path):
    _write(tmp_path, [
        _record("real", 1000),
        {"spotify_episode_uri": "spotify:episode:xyz", "ms_played": 99999},
        {"spotify_track_uri": None, "ms_played": 99999},
    ])

    top = top_tracks_for_dir(tmp_path)
    assert [t["track_id"] for t in top] == ["real"]


def test_aggregates_across_multiple_files(tmp_path):
    _write(tmp_path, [_record("aaa", 1000)], "Streaming_History_Audio_2023.json")
    _write(tmp_path, [_record("aaa", 2000)], "Streaming_History_Audio_2024.json")

    top = top_tracks_for_dir(tmp_path)
    assert top[0]["ms_played_total"] == 3000
    assert top[0]["play_count"] == 2


def test_tolerates_malformed_file(tmp_path):
    _write(tmp_path, [_record("good", 1000)])
    (tmp_path / "Streaming_History_Audio_2020.json").write_text("{not json", encoding="utf-8")

    top = top_tracks_for_dir(tmp_path)
    assert [t["track_id"] for t in top] == ["good"]


def test_empty_dir_returns_empty(tmp_path):
    assert top_tracks_for_dir(tmp_path) == []
