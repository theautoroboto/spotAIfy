# tests/test_whosampled_cache.py
import spotaify.clients.whosampled_client as ws


def test_cache_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "_CACHE_DIR", tmp_path)
    result = {"url": "u", "samples": [{"title": "T"}], "sampled_by": [],
              "sample_count": 1, "sampled_by_count": 0}

    ws._cache_save("Straight Outta Compton", "N.W.A.", result)
    assert ws._cache_load("Straight Outta Compton", "N.W.A.") == result
    # Key is case/whitespace-insensitive.
    assert ws._cache_load("  straight outta compton ", "n.w.a.") == result


def test_errors_are_never_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "_CACHE_DIR", tmp_path)
    ws._cache_save("Song", "Artist", {"error": "WhoSampled search blocked (HTTP 403)"})
    assert ws._cache_load("Song", "Artist") is None
    assert list(tmp_path.iterdir()) == []


def test_fetch_samples_serves_from_cache_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "_CACHE_DIR", tmp_path)
    cached = {"url": "u", "samples": [], "sampled_by": [],
              "sample_count": 0, "sampled_by_count": 0}
    ws._cache_save("Hurt", "Nine Inch Nails", cached)

    def boom(*a, **k):
        raise AssertionError("network must not be touched on cache hit")
    monkeypatch.setattr(ws, "_get", boom)

    assert ws.fetch_samples("Hurt", "Nine Inch Nails") == cached
