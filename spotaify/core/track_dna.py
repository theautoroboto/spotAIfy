# track_dna.py
import time
import musicbrainzngs as mb
from spotaify.core.artist_graph import fetch_artist_relations, resolve_artist_mbid
from spotaify.config import MUSICBRAINZ_CONTACT

mb.set_useragent("spotify-playlist-agent", "0.1", MUSICBRAINZ_CONTACT or "spotify-playlist-agent")

COMPOSER_TYPES = {"composer", "lyricist", "writer", "arranger"}
PRODUCER_TYPES = {"producer", "mix", "engineer", "recording"}
PERFORMER_TYPES = {"performer", "instrument", "vocal", "guitar", "bass", "drums", "keyboards"}
SAMPLE_TYPES = {"samples material from", "has samples of"}


def resolve_recording_mbid(title: str, artist: str) -> str | None:
    time.sleep(1)
    try:
        result = mb.search_recordings(recording=title, artist=artist, limit=1)
    except mb.ResponseError as e:
        raise RuntimeError(f"MusicBrainz rate limit: {e}") from e
    recordings = result.get("recording-list", [])
    return recordings[0]["id"] if recordings else None


def fetch_recording_credits(recording_mbid: str) -> dict:
    time.sleep(1)
    try:
        result = mb.get_recording_by_id(
            recording_mbid,
            includes=["artist-rels", "recording-rels", "work-rels"],
        )
    except mb.ResponseError as e:
        raise RuntimeError(f"MusicBrainz rate limit: {e}") from e
    rec = result["recording"]

    composers, producers, session_musicians, samples = [], [], [], []

    for rel in rec.get("artist-relation-list", []):
        rel_type = rel.get("type", "").lower()
        artist_name = rel.get("artist", {}).get("name", "")
        artist_id = rel.get("artist", {}).get("id", "")
        attrs = rel.get("attribute-list", [])
        if rel_type in COMPOSER_TYPES:
            composers.append({"name": artist_name, "mbid": artist_id, "role": rel_type})
        elif any(t in rel_type for t in PRODUCER_TYPES):
            producers.append({"name": artist_name, "mbid": artist_id, "role": rel_type})
        elif rel_type in PERFORMER_TYPES or any(a.lower() in PERFORMER_TYPES for a in attrs):
            session_musicians.append({"name": artist_name, "mbid": artist_id, "role": attrs[0] if attrs else rel_type})

    for rel in rec.get("recording-relation-list", []):
        if rel.get("type", "").lower() in SAMPLE_TYPES:
            sampled = rel.get("recording", {})
            credits = sampled.get("artist-credit", [{}])
            samples.append({
                "title": sampled.get("title", ""),
                "recording_mbid": sampled.get("id", ""),
                "artist": credits[0].get("artist", {}).get("name", "") if credits else "",
            })

    return {
        "recording_mbid": recording_mbid,
        "title": rec.get("title", ""),
        "composers": [c["name"] for c in composers],
        "producers": [p["name"] for p in producers],
        "session_musicians": session_musicians,
        "samples": samples,
        "composer_details": composers,
        "producer_details": producers,
    }


def expand_bateman(title: str, artist: str, depth: int = 1) -> dict:
    mbid = resolve_recording_mbid(title, artist)
    if not mbid:
        return {"error": f"Recording not found: {title} by {artist}"}

    credits = fetch_recording_credits(mbid)
    _MAX_PERSONS = 8
    persons: list[dict] = []

    # Prioritize composers and producers; session musicians fill remaining slots
    for person in credits["composer_details"] + credits["producer_details"] + credits["session_musicians"]:
        if len(persons) >= _MAX_PERSONS:
            break
        mbid_p = person.get("mbid", "")
        artist_mbid = resolve_artist_mbid(person["name"]) if not mbid_p else mbid_p
        persons.append({
            "name": person["name"],
            "mbid": mbid_p or artist_mbid,
            "link_type": person.get("role", "contributor"),
            "bateman_path": f"{title} by {artist} → {person.get('role', 'credit')} → {person['name']}",
        })

    sample_origins = []
    for sample in credits["samples"][:4]:
        sample_origins.append({
            "title": sample["title"],
            "artist": sample["artist"],
            "recording_mbid": sample["recording_mbid"],
            "bateman_path": f"{title} by {artist} → samples → {sample['title']} by {sample['artist']}",
        })

    return {
        "seed_title": title,
        "seed_artist": artist,
        "recording_mbid": credits["recording_mbid"],
        "composers": credits["composers"],
        "producers": credits["producers"],
        "session_musicians": [m["name"] for m in credits["session_musicians"]],
        "samples": sample_origins,
        "persons": persons,
    }