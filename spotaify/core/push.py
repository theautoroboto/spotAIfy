# push.py
import argparse
from spotaify.clients.spotify_client import SpotifyClient
from spotaify.core.fiftyone_store import get_or_create_dataset, get_tagged_playlist


def main():
    parser = argparse.ArgumentParser(description="Push FiftyOne-tagged playlist to Spotify")
    parser.add_argument("--name", required=True, help="Playlist name in Spotify")
    parser.add_argument("--public", action="store_true", help="Make playlist public")
    parser.add_argument("--tag", default="playlist", help="FiftyOne tag to push (default: playlist)")
    args = parser.parse_args()

    dataset = get_or_create_dataset()
    track_ids = get_tagged_playlist(dataset, args.tag)

    if not track_ids:
        print(f"No tracks tagged '{args.tag}' in the dataset. Run agent.py first.")
        return

    print(f"Pushing {len(track_ids)} tracks to Spotify as '{args.name}'...")
    client = SpotifyClient(user_auth=True)
    url = client.create_playlist(args.name, track_ids, public=args.public)
    print(f"Playlist created: {url}")


if __name__ == "__main__":
    main()