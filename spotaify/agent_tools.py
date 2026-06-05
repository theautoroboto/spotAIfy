# agent_tools.py
from spotaify.clients.spotify_client import SpotifyClient
from spotaify.core.artist_graph import traverse_graph, get_all_artist_names
from spotaify.core.track_dna import expand_track_dna
from spotaify.core.history_profile import enrich_candidates
from spotaify.clients.whosampled_client import fetch_samples as whosampled_fetch
from spotaify.clients.genius_client import fetch_song_info as genius_fetch

_spotify: SpotifyClient | None = None
_spotify_user: SpotifyClient | None = None


def _get_spotify_client() -> SpotifyClient:
    global _spotify
    if _spotify is None:
        _spotify = SpotifyClient(user_auth=False)
    return _spotify


def _user_client() -> SpotifyClient:
    global _spotify_user
    if _spotify_user is None:
        _spotify_user = SpotifyClient(user_auth=True)
    return _spotify_user


TOOLS = [
    {"name": "search_catalog",
     "description": "Search the Spotify catalog by keywords. Returns up to 10 tracks with metadata.",
     "input_schema": {"type": "object", "properties": {
         "query": {"type": "string"}, "limit": {"type": "integer", "default": 10, "maximum": 10}},
         "required": ["query"]}},
    {"name": "rank_and_select",
     "description": "Score and rank candidate tracks, select the final N for the playlist.",
     "input_schema": {"type": "object", "properties": {
         "candidates": {"type": "array", "items": {"type": "object"}},
         "count": {"type": "integer", "default": 20},
         "boost_discovery": {"type": "boolean", "default": False}},
         "required": ["candidates", "count"]}},
    {"name": "map_artist_connections",
     "description": "Query MusicBrainz for all artists connected to a seed artist.",
     "input_schema": {"type": "object", "properties": {
         "artist_name": {"type": "string"}, "depth": {"type": "integer", "default": 1, "maximum": 3}},
         "required": ["artist_name"]}},
    {"name": "traverse_artist_graph",
     "description": "Return all artists in the connection graph with paths back to the seed.",
     "input_schema": {"type": "object", "properties": {
         "artist_name": {"type": "string"}, "depth": {"type": "integer", "default": 1, "maximum": 3}},
         "required": ["artist_name"]}},
    {"name": "fetch_artist_top_tracks",
     "description": "Get top tracks from a list of artist names. Returns merged track list.",
     "input_schema": {"type": "object", "properties": {
         "artist_names": {"type": "array", "items": {"type": "string"}},
         "tracks_per_artist": {"type": "integer", "default": 3}}, "required": ["artist_names"]}},
    {"name": "explain_connection",
     "description": "Generate a plain-English explanation of how a track connects to the seed artist.",
     "input_schema": {"type": "object", "properties": {
         "track_id": {"type": "string"}, "seed_artist": {"type": "string"},
         "connection_path": {"type": "string"}}, "required": ["track_id", "seed_artist"]}},
    {"name": "deep_dive_track",
     "description": "DNA mode: resolve a track to MusicBrainz and fetch all credits (composers, producers, session musicians, samples).",
     "input_schema": {"type": "object", "properties": {
         "title": {"type": "string"}, "artist": {"type": "string"},
         "depth": {"type": "integer", "default": 1}}, "required": ["title", "artist"]}},
    {"name": "find_tracks_by_person",
     "description": "Given a person's name (writer, producer), find other recordings they worked on via Spotify search.",
     "input_schema": {"type": "object", "properties": {
         "person_name": {"type": "string"}, "role": {"type": "string"},
         "limit": {"type": "integer", "default": 20}}, "required": ["person_name"]}},
    {"name": "resolve_samples",
     "description": "Find specific tracks on Spotify from a list of {title, artist, dna_path, dna_link_type} objects. Pass items from fetch_whosampled `samples` with dna_link_type='samples_from', and items from `sampled_by` with dna_link_type='sampled_by'. Returns only those exact tracks — no additional artist tracks.",
     "input_schema": {"type": "object", "properties": {
         "samples": {"type": "array", "items": {"type": "object"}}}, "required": ["samples"]},
     "cache_control": {"type": "ephemeral"}},
    {"name": "fetch_whosampled",
     "description": "Scrape WhoSampled.com for a track's sample relationships: what it samples and what samples it.",
     "input_schema": {"type": "object", "properties": {
         "title": {"type": "string"}, "artist": {"type": "string"}},
         "required": ["title", "artist"]}},
    {"name": "fetch_genius_info",
     "description": "Fetch a track's About description, writers, and producers from Genius.com.",
     "input_schema": {"type": "object", "properties": {
         "title": {"type": "string"},
         "artist": {"type": "string"},
         "track_id": {"type": "string", "description": "Spotify track ID (optional)"}},
         "required": ["title", "artist"]}},
    {"name": "create_spotify_playlist",
     "description": "Save the final playlist to the user's Spotify account. Always call this as the last step.",
     "input_schema": {"type": "object", "properties": {
         "name": {"type": "string", "description": "Playlist name"},
         "description": {"type": "string", "description": "Playlist description explaining the theme and connections"},
         "track_ids": {"type": "array", "items": {"type": "string"}, "description": "Spotify track IDs"},
         "public": {"type": "boolean", "default": False}},
         "required": ["name", "description", "track_ids"]}},
]


_TOOL_CACHE: dict[tuple, dict] = {}
_CACHED_TOOLS = {"fetch_genius_info", "fetch_whosampled", "deep_dive_track",
                 "map_artist_connections", "traverse_artist_graph"}

_SLIM_KEEP = {"track_id", "title", "artist", "artist_id", "popularity",
              "is_new_discovery", "connection_artist", "dna_path", "dna_link_type",
              "genius_about", "genius_url"}


def _slim(tracks: list[dict]) -> list[dict]:
    """Drop bulky fields not needed for agent reasoning."""
    return [{k: v for k, v in t.items() if k in _SLIM_KEEP} for t in tracks]


def execute_tool(name: str, inputs: dict) -> dict:
    if name in _CACHED_TOOLS:
        cache_key = (name, tuple(sorted(inputs.items())))
        if cache_key in _TOOL_CACHE:
            print(f"  [cache hit]", flush=True)
            return _TOOL_CACHE[cache_key]

    try:
        result = _execute_tool_inner(name, inputs)
    except Exception as e:
        raise RuntimeError(f"[{name}] {e}") from e

    if name in _CACHED_TOOLS and "error" not in result:
        _TOOL_CACHE[(name, tuple(sorted(inputs.items())))] = result
    return result


def _execute_tool_inner(name: str, inputs: dict) -> dict:
    if name == "search_catalog":
        tracks = _get_spotify_client().search_catalog(inputs["query"], limit=inputs.get("limit", 10))
        return {"tracks": _slim(tracks), "count": len(tracks)}

    if name == "rank_and_select":
        candidates = inputs["candidates"]
        boost = inputs.get("boost_discovery", False)
        enrich_candidates(candidates)
        for c in candidates:
            score = c.get("audio_sim", 0.5)
            score += c.get("completion_rate", 0.0) * 0.3
            score -= c.get("skip_rate", 0.0) * 0.2
            if boost and c.get("is_new_discovery", False):
                score += 0.15
            c["score"] = score
        ranked = sorted(candidates, key=lambda x: -x.get("score", 0))
        return {"playlist": ranked[: inputs.get("count", 20)]}

    if name == "create_spotify_playlist":
        url = _user_client().create_playlist(
            inputs["name"],
            inputs["track_ids"],
            public=inputs.get("public", False),
            description=inputs.get("description", ""),
        )
        return {"status": "created", "url": url}

    if name == "map_artist_connections":
        graph = traverse_graph(inputs["artist_name"], depth=inputs.get("depth", 1))
        artists = get_all_artist_names(graph)[:15]
        return {"artists": artists, "total": len(artists)}

    if name == "traverse_artist_graph":
        graph = traverse_graph(inputs["artist_name"], depth=inputs.get("depth", 1))
        return {"artists": get_all_artist_names(graph)[:15]}

    if name == "fetch_artist_top_tracks":
        tracks = []
        client = _get_spotify_client()
        max_total = 60
        for artist_name in inputs["artist_names"]:
            if len(tracks) >= max_total:
                break
            artist = client.search_artist(artist_name)
            if artist:
                top = client.get_artist_top_tracks(artist["artist_id"])
                for t in top[: inputs.get("tracks_per_artist", 5)]:
                    t["connection_artist"] = artist_name
                    tracks.append(t)
        return {"tracks": _slim(tracks), "count": len(tracks)}

    if name == "explain_connection":
        path = inputs.get("connection_path", "direct connection")
        return {"explanation": f"{inputs['track_id']} connects to {inputs['seed_artist']} via: {path}"}

    if name == "deep_dive_track":
        return expand_track_dna(inputs["title"], inputs["artist"], depth=inputs.get("depth", 1))

    if name == "find_tracks_by_person":
        client = _get_spotify_client()
        query = f"{inputs['person_name']} {inputs.get('role', '')}"
        limit = min(inputs.get("limit", 10), 10)
        tracks = client.search_catalog(query.strip(), limit=limit)
        return {"tracks": _slim(tracks), "count": len(tracks)}

    if name == "resolve_samples":
        tracks = []
        client = _get_spotify_client()
        for sample in inputs["samples"]:
            title = sample.get("title", "")
            artist_name = sample.get("artist", "")
            dna_path = sample.get("dna_path", "")
            if title and artist_name:
                results = client.search_catalog(f"{title} {artist_name}", limit=1)
                if results:
                    results[0]["dna_path"] = dna_path
                    results[0]["dna_link_type"] = "sample"
                    tracks.append(results[0])
        return {"tracks": _slim(tracks), "count": len(tracks)}

    if name == "fetch_whosampled":
        return whosampled_fetch(inputs["title"], inputs["artist"])

    if name == "fetch_genius_info":
        return genius_fetch(inputs["title"], inputs["artist"])

    return {"error": f"Unknown tool: {name}"}
