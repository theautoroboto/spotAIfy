# embeddings.py
import io
import os
import tempfile

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

AUDIO_FEATURE_KEYS = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo", "duration_ms", "time_signature",
]

_IDX = {k: i for i, k in enumerate(AUDIO_FEATURE_KEYS)}

_text_model: SentenceTransformer | None = None


def _get_text_model() -> SentenceTransformer:
    global _text_model
    if _text_model is None:
        _text_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _text_model


def compute_audio_embedding(features: dict) -> list[float]:
    vec = np.array([float(features.get(k, 0.0)) for k in AUDIO_FEATURE_KEYS])
    # Normalize each feature to [0, 1]
    vec[_IDX["loudness"]] = np.clip((vec[_IDX["loudness"]] + 60) / 60, 0, 1)
    vec[_IDX["tempo"]] = np.clip(vec[_IDX["tempo"]] / 250, 0, 1)
    vec[_IDX["duration_ms"]] = np.clip(vec[_IDX["duration_ms"]] / 600_000, 0, 1)
    vec[_IDX["key"]] = np.clip(vec[_IDX["key"]] / 11, 0, 1)
    vec[_IDX["time_signature"]] = np.clip((vec[_IDX["time_signature"]] - 3) / 4, 0, 1)
    return np.clip(vec, 0, 1).tolist()


def analyze_preview(preview_url: str, duration_ms_hint: int = 0) -> dict:
    """Download a 30-second preview and extract Spotify-compatible audio features via librosa."""
    import librosa

    resp = requests.get(preview_url, timeout=15)
    resp.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(resp.content)
        tmp_path = f.name

    try:
        y, sr = librosa.load(tmp_path, sr=22050, mono=True)
    finally:
        os.unlink(tmp_path)

    duration_ms = int(len(y) / sr * 1000) or duration_ms_hint

    # Tempo and beat strength
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    danceability = float(np.clip(np.mean(onset_env) / 10.0, 0.0, 1.0))

    # Energy / loudness
    rms = librosa.feature.rms(y=y)[0]
    energy = float(np.clip(np.mean(rms) * 10.0, 0.0, 1.0))
    loudness = float(np.clip(librosa.amplitude_to_db(np.array([np.mean(rms)]))[0], -60.0, 0.0))

    # Spectral
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    acousticness = float(np.clip(1.0 - np.mean(centroid) / (sr / 2), 0.0, 1.0))
    speechiness = float(np.clip(np.mean(zcr) * 10.0, 0.0, 1.0))
    instrumentalness = float(np.clip(1.0 - speechiness, 0.0, 1.0))
    valence = float(np.clip(np.mean(rolloff) / (sr / 2), 0.0, 1.0))
    liveness = float(np.clip(np.std(rms) / (np.mean(rms) + 1e-6), 0.0, 1.0))

    # Key and mode via chroma
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    key = int(np.argmax(chroma_mean))
    major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    rolled = np.roll(chroma_mean, -key)
    mode = 1 if np.dot(rolled, major) >= np.dot(rolled, minor) else 0

    return {
        "danceability": danceability,
        "energy": energy,
        "key": key,
        "loudness": loudness,
        "mode": mode,
        "speechiness": speechiness,
        "acousticness": acousticness,
        "instrumentalness": instrumentalness,
        "liveness": liveness,
        "valence": valence,
        "tempo": tempo,
        "duration_ms": duration_ms,
        "time_signature": 4,
    }


def compute_text_embedding(title: str, artist: str, genres: list[str]) -> list[float]:
    text = f"{title} {artist} {' '.join(genres)}"
    return _get_text_model().encode(text).tolist()


def fit_umap(embeddings: list[list[float]], n_neighbors: int = 15) -> object:
    import umap
    X = np.array(embeddings)
    reducer = umap.UMAP(n_components=2, n_neighbors=min(n_neighbors, len(X) - 1), random_state=42)
    reducer.fit(X)
    return reducer


def project_umap(reducer, embeddings: list[list[float]]) -> list[list[float]]:
    return reducer.transform(np.array(embeddings)).tolist()