# tests/test_everyone.py
import json

from spotaify.everyone import _interleave, _users_with_history


def _pool(name: str, n: int, prefix: str | None = None) -> list[dict]:
    """Ranked track pool for a fake user; track ids are unique per prefix."""
    prefix = prefix if prefix is not None else name
    return [
        {"track_id": f"{prefix}-{i}", "title": f"Track {i}", "artist": name}
        for i in range(n)
    ]


# ── _interleave ───────────────────────────────────────────────────────────────

def test_interleave_round_robin_alternates_users():
    pools = {"ann": _pool("ann", 4), "bob": _pool("bob", 4)}
    merged, contributed = _interleave(pools, per_user=2)

    assert [t["track_id"] for t in merged] == ["ann-0", "bob-0", "ann-1", "bob-1"]
    assert contributed == {"ann": 2, "bob": 2}


def test_interleave_dedupes_and_backfills():
    # bob's #1 track is the same as ann's #1 — bob should skip it and
    # contribute his next-ranked track instead.
    ann = _pool("ann", 3)
    bob = _pool("bob", 3)
    bob[0]["track_id"] = ann[0]["track_id"]

    merged, contributed = _interleave({"ann": ann, "bob": bob}, per_user=2)

    ids = [t["track_id"] for t in merged]
    assert len(ids) == len(set(ids)), "merged playlist must not contain duplicates"
    assert contributed == {"ann": 2, "bob": 2}
    assert ids == ["ann-0", "bob-1", "ann-1", "bob-2"]


def test_interleave_exhausted_history_contributes_what_it_has():
    pools = {"ann": _pool("ann", 5), "bob": _pool("bob", 1)}
    merged, contributed = _interleave(pools, per_user=3)

    assert contributed == {"ann": 3, "bob": 1}
    assert len(merged) == 4


def test_interleave_per_user_caps_contributions():
    pools = {"ann": _pool("ann", 10)}
    merged, contributed = _interleave(pools, per_user=4)

    assert contributed == {"ann": 4}
    assert len(merged) == 4


def test_interleave_empty_pools():
    merged, contributed = _interleave({}, per_user=5)
    assert merged == []
    assert contributed == {}


# ── _users_with_history ───────────────────────────────────────────────────────

def _write_export(d, filename="Streaming_History_Audio_2024.json"):
    d.mkdir(parents=True, exist_ok=True)
    (d / filename).write_text("[]", encoding="utf-8")


def test_users_with_history_finds_only_dirs_with_exports(tmp_path):
    _write_export(tmp_path / "alice")
    _write_export(tmp_path / "bob")
    (tmp_path / "empty").mkdir()
    (tmp_path / "videos_only").mkdir()
    (tmp_path / "videos_only" / "Streaming_History_Video_2024.json").write_text("[]")
    (tmp_path / "loose_file.json").write_text("[]")

    users = _users_with_history(tmp_path)
    assert [d.name for d in users] == ["alice", "bob"]


def test_users_with_history_sorted_case_insensitive(tmp_path):
    for name in ("Zoe", "amy", "Bob"):
        _write_export(tmp_path / name)

    users = _users_with_history(tmp_path)
    assert [d.name for d in users] == ["amy", "Bob", "Zoe"]


def test_users_with_history_missing_base_dir(tmp_path):
    assert _users_with_history(tmp_path / "nope") == []
