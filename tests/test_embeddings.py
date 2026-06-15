# tests/test_embeddings.py
import numpy as np
import pytest
from tests.conftest import TEST_AUDIO_FEATURES
from spotaify.core.embeddings import compute_audio_embedding, compute_text_embedding, AUDIO_FEATURE_KEYS

def test_audio_embedding_length():
    vec = compute_audio_embedding(TEST_AUDIO_FEATURES)
    assert len(vec) == len(AUDIO_FEATURE_KEYS)

def test_audio_embedding_all_values_in_0_1():
    vec = compute_audio_embedding(TEST_AUDIO_FEATURES)
    assert all(0.0 <= v <= 1.0 for v in vec), f"Out-of-range values: {[v for v in vec if not 0<=v<=1]}"

def test_audio_embedding_missing_keys_default_to_zero():
    vec = compute_audio_embedding({})
    assert len(vec) == len(AUDIO_FEATURE_KEYS)
    assert all(0.0 <= v <= 1.0 for v in vec)

def test_text_embedding_returns_list_of_floats():
    vec = compute_text_embedding("Hurt", "Nine Inch Nails", ["industrial rock", "alternative rock"])
    assert isinstance(vec, list)
    assert len(vec) > 0
    assert all(isinstance(v, float) for v in vec)

def test_text_embedding_different_inputs_differ():
    v1 = compute_text_embedding("Hurt", "Nine Inch Nails", ["industrial rock"])
    v2 = compute_text_embedding("Stairway to Heaven", "Led Zeppelin", ["rock"])
    assert v1 != v2

def test_audio_embedding_high_energy_track():
    features = {**TEST_AUDIO_FEATURES, "energy": 0.95, "tempo": 180.0, "loudness": -3.0}
    vec = compute_audio_embedding(features)
    # energy is index 1 in AUDIO_FEATURE_KEYS
    energy_idx = AUDIO_FEATURE_KEYS.index("energy")
    assert vec[energy_idx] == pytest.approx(0.95)