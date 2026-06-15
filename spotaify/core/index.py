# index.py
import argparse
from spotaify.clients.spotify_client import SpotifyClient
from spotaify.core.history_importer import parse_history_files, compute_play_stats
from spotaify.core.embeddings import compute_audio_embedding, compute_text_embedding
from spotaify.core.fiftyone_store import get_or_create_dataset, upsert_track
from spotaify.config import history_dir_for


def enrich_and_upsert(dataset, tracks: list[dict], client: SpotifyClient,
                      history_stats: dict, source: str) -> None:
    track_ids = [t["track_id"] for t in tracks if t.get("track_id")]
    features_map = client.fetch_audio_features(track_ids)

    for track in tracks:
        tid = track.get("track_id")
        if not tid:
            continue

        track["source"] = source

        af = features_map.get(tid, {})
        if af:
            track["audio_embedding"] = compute_audio_embedding(af)
            track.update(af)

        artist_id = track.get("artist_id")
        if artist_id:
            try:
                info = client.get_artist_info(artist_id)
                track["genres"] = info.get("genres", [])
            except Exception:
                track["genres"] = []

        track["text_embedding"] = compute_text_embedding(
            track.get("title", ""), track.get("artist", ""), track.get("genres", []))

        hs = history_stats.get(tid)
        if hs:
            enriched = compute_play_stats(hs, track.get("duration_ms", 1))
            track.update({
                "play_count": enriched["play_count"],
                "total_ms_played": enriched["total_ms_played"],
                "completion_rate": enriched["completion_rate"],
                "skip_rate": enriched["skip_rate"],
                "first_played_at": enriched.get("first_played_at"),
                "last_played_at": enriched.get("last_played_at"),
            })
        track["is_new_discovery"] = track.get("play_count", 0) == 0

        upsert_track(dataset, track)
        print(f"  ✓ {track.get('title', tid)} — {track.get('artist', '')}")


def main():
    parser = argparse.ArgumentParser(description="Index Spotify tracks into FiftyOne")
    parser.add_argument("--liked", action="store_true", help="Index liked songs")
    parser.add_argument("--playlists", action="store_true", help="Index all user playlists")
    parser.add_argument("--recent", action="store_true", help="Index recently played (last 50)")
    parser.add_argument("--history", action="store_true", help="Import Spotify data export from history/")
    parser.add_argument("--username", default="", help="User subdirectory inside HISTORY_DIR (required when history is per-user)")
    parser.add_argument("--resume", action="store_true", help="Skip tracks already in dataset")
    args = parser.parse_args()

    dataset = get_or_create_dataset()
    history_stats: dict = {}

    if args.history:
        hist_dir = history_dir_for(args.username)
        print(f"Parsing listening history export from {hist_dir}...")
        history_stats = parse_history_files(hist_dir)
        print(f"  Found {len(history_stats)} unique tracks in history")

    needs_user_auth = args.liked or args.playlists or args.recent
    client = SpotifyClient(user_auth=needs_user_auth)

    existing_ids: set[str] = set()
    if args.resume:
        existing_ids = {s["track_id"] for s in dataset if s.get("track_id")}
        print(f"Resume mode: {len(existing_ids)} tracks already indexed")

    if args.liked:
        print("Fetching liked songs...")
        tracks = client.liked_tracks()
        if args.resume:
            tracks = [t for t in tracks if t["track_id"] not in existing_ids]
        print(f"  Indexing {len(tracks)} liked tracks...")
        enrich_and_upsert(dataset, tracks, client, history_stats, "liked")

    if args.playlists:
        print("Fetching playlists...")
        for pl in client.user_playlists():
            print(f"  Playlist: {pl['name']}")
            tracks = client.playlist_tracks(pl["id"])
            if args.resume:
                tracks = [t for t in tracks if t["track_id"] not in existing_ids]
            enrich_and_upsert(dataset, tracks, client, history_stats, "playlist")

    if args.recent:
        print("Fetching recently played...")
        tracks = client.recently_played()
        if args.resume:
            tracks = [t for t in tracks if t["track_id"] not in existing_ids]
        enrich_and_upsert(dataset, tracks, client, history_stats, "recent")

    if args.history and not needs_user_auth:
        print("Upserting history-only tracks (metadata will be minimal)...")
        ids = list(history_stats.keys())
        features_map = client.fetch_audio_features(ids)
        for tid, hs in history_stats.items():
            if args.resume and tid in existing_ids:
                continue
            track = {"track_id": tid, "title": hs["title"], "artist": hs["artist"],
                     "album": hs["album"], "source": "history"}
            af = features_map.get(tid, {})
            if af:
                track["audio_embedding"] = compute_audio_embedding(af)
                track.update(af)
            track["text_embedding"] = compute_text_embedding(hs["title"], hs["artist"], [])
            enriched = compute_play_stats(hs, af.get("duration_ms", 1) if af else 1)
            track.update({k: enriched[k] for k in ("play_count", "total_ms_played",
                           "completion_rate", "skip_rate", "first_played_at", "last_played_at")})
            track["is_new_discovery"] = False
            upsert_track(dataset, track)

    print(f"\nDone. Dataset '{dataset.name}' now has {len(dataset)} samples.")

    if len(dataset) >= 10:
        from embeddings import fit_umap, project_umap
        print("Computing UMAP projection...")
        samples_with_emb = [s for s in dataset if s.get("audio_embedding")]
        if len(samples_with_emb) >= 10:
            embs = [s["audio_embedding"] for s in samples_with_emb]
            reducer = fit_umap(embs)
            projections = project_umap(reducer, embs)
            for sample, proj in zip(samples_with_emb, projections):
                sample["embedding"] = proj
                sample.save()
            print(f"  UMAP computed for {len(samples_with_emb)} tracks.")
        else:
            print("  Not enough samples with embeddings to compute UMAP.")


if __name__ == "__main__":
    main()
