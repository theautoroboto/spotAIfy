# tests/test_fiftyone_store.py
import pytest
import fiftyone as fo
from tests.conftest import TEST_TRACK, TEST_AUDIO_FEATURES
from spotaify.core.fiftyone_store import (
    get_or_create_dataset, upsert_track, query_by_ids,
    tag_playlist, get_tagged_playlist,
)

TEST_DS = "test_spotify_explorer"

@pytest.fixture(autouse=True)
def cleanup():
    yield
    if fo.dataset_exists(TEST_DS):
        fo.delete_dataset(TEST_DS)

def test_get_or_create_creates_persistent_dataset():
    ds = get_or_create_dataset(TEST_DS)
    assert fo.dataset_exists(TEST_DS)
    assert ds.persistent

def test_get_or_create_returns_existing_dataset():
    ds1 = get_or_create_dataset(TEST_DS)
    ds2 = get_or_create_dataset(TEST_DS)
    assert ds1.name == ds2.name

def test_upsert_track_adds_sample():
    ds = get_or_create_dataset(TEST_DS)
    track = {**TEST_TRACK, "audio_features": TEST_AUDIO_FEATURES}
    upsert_track(ds, track)
    assert len(ds) == 1

def test_upsert_track_updates_existing():
    ds = get_or_create_dataset(TEST_DS)
    upsert_track(ds, {**TEST_TRACK})
    upsert_track(ds, {**TEST_TRACK, "play_count": 5})
    assert len(ds) == 1
    sample = ds.first()
    assert sample["play_count"] == 5

def test_upsert_track_uses_preview_url_as_filepath():
    ds = get_or_create_dataset(TEST_DS)
    upsert_track(ds, {**TEST_TRACK})
    # On Windows, FiftyOne may incorrectly resolve URLs as local paths.
    # This checks if the core part of the URL is present in the resolved path.
    assert "p.scdn.co/mp3-preview/abc123" in ds.first().filepath.replace("\\", "/")

def test_upsert_track_fallback_filepath_when_no_preview():
    ds = get_or_create_dataset(TEST_DS)
    track = {**TEST_TRACK, "preview_url": None}
    upsert_track(ds, track)
    # Normalize path separators for cross-platform compatibility
    normalized_path = ds.first().filepath.replace("\\", "/")
    assert normalized_path.endswith(f"/{TEST_TRACK['track_id']}.media")

def test_query_by_ids_returns_matching_samples():
    ds = get_or_create_dataset(TEST_DS)
    upsert_track(ds, TEST_TRACK)
    view = query_by_ids(ds, [TEST_TRACK["track_id"]])
    assert len(view) == 1

def test_tag_and_get_tagged_playlist():
    ds = get_or_create_dataset(TEST_DS)
    upsert_track(ds, TEST_TRACK)
    tag_playlist(ds, [TEST_TRACK["track_id"]], "my_playlist")
    ids = get_tagged_playlist(ds, "my_playlist")
    assert TEST_TRACK["track_id"] in ids