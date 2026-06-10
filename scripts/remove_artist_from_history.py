"""Remove all plays by an exact artist name from a user's imported
Spotify Extended Streaming History files.

Usage: python scripts/remove_artist_from_history.py <history_dir> <artist_name>

Only files that contain the artist are rewritten. Back up first —
this edits files in place.
"""

import json
import sys
from pathlib import Path


def main() -> None:
    history_dir = Path(sys.argv[1])
    artist = sys.argv[2]

    for path in sorted(history_dir.glob("Streaming_History_*.json")):
        records = json.loads(path.read_text(encoding="utf-8"))
        kept = [r for r in records if r.get("master_metadata_album_artist_name") != artist]
        removed = len(records) - len(kept)
        if removed == 0:
            continue
        path.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{path.name}: removed {removed}, kept {len(kept)}")


if __name__ == "__main__":
    main()
