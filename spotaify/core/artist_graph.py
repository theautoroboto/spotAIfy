# artist_graph.py
import json
import time
from collections import deque
from pathlib import Path
import musicbrainzngs as mb
from spotaify.config import GRAPH_DIR, MUSICBRAINZ_CONTACT

mb.set_useragent("spotify-playlist-agent", "0.1", MUSICBRAINZ_CONTACT or "spotify-playlist-agent")


def resolve_artist_mbid(name: str) -> str | None:
    result = mb.search_artists(artist=name, limit=1)
    artists = result.get("artist-list", [])
    return artists[0]["id"] if artists else None


def fetch_artist_relations(mbid: str) -> dict:
    time.sleep(1)
    try:
        result = mb.get_artist_by_id(mbid, includes=["artist-rels"])
    except mb.ResponseError as e:
        raise RuntimeError(f"MusicBrainz rate limit: {e}") from e
    artist = result["artist"]
    relations = [
        {
            "type": rel.get("type", ""),
            "direction": rel.get("direction", ""),
            "related_mbid": rel.get("artist", {}).get("id", ""),
            "related_name": rel.get("artist", {}).get("name", ""),
            "begin": rel.get("begin", ""),
            "end": rel.get("end", ""),
            "attributes": rel.get("attribute-list", []),
        }
        for rel in artist.get("artist-relation-list", [])
    ]
    return {"mbid": mbid, "name": artist.get("name", ""), "type": artist.get("type", ""), "relations": relations}


def traverse_graph(seed_name: str, depth: int = 2, max_nodes: int = 30, graph_dir: str = GRAPH_DIR) -> dict:
    slug = seed_name.lower().replace(" ", "_")
    cache_path = Path(graph_dir) / f"{slug}_d{depth}_n{max_nodes}.json"
    Path(graph_dir).mkdir(exist_ok=True)

    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    seed_mbid = resolve_artist_mbid(seed_name)
    if not seed_mbid:
        raise ValueError(f"Artist not found in MusicBrainz: {seed_name}")

    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    queue: deque = deque([(seed_mbid, seed_name, 0)])
    visited: set[str] = set()

    while queue:
        if len(nodes) >= max_nodes:
            break
        mbid, name, current_depth = queue.popleft()
        if mbid in visited or current_depth > depth:
            continue
        visited.add(mbid)

        data = fetch_artist_relations(mbid)
        nodes[mbid] = {"mbid": mbid, "name": data["name"], "type": data.get("type", ""), "depth": current_depth}

        for rel in data["relations"]:
            related_mbid = rel["related_mbid"]
            if not related_mbid:
                continue
            edges.append({"from": mbid, "to": related_mbid, "type": rel["type"],
                          "attributes": rel["attributes"], "begin": rel["begin"], "end": rel["end"]})
            if current_depth < depth and related_mbid not in visited:
                queue.append((related_mbid, rel["related_name"], current_depth + 1))

    graph = {"seed": seed_name, "seed_mbid": seed_mbid, "depth": depth, "nodes": nodes, "edges": edges}
    with open(cache_path, "w") as f:
        json.dump(graph, f, indent=2, default=str)
    return graph


def get_all_artist_names(graph: dict) -> list[dict]:
    seed_mbid = graph["seed_mbid"]
    nodes, edges = graph["nodes"], graph["edges"]
    result = []
    for mbid, node in nodes.items():
        if mbid == seed_mbid:
            continue
        incoming = [e for e in edges if e["to"] == mbid]
        if incoming:
            e = incoming[0]
            from_name = nodes.get(e["from"], {}).get("name", graph["seed"])
            path = (f"{graph['seed']} → {from_name} → {node['name']}"
                    if from_name != graph["seed"] else f"{graph['seed']} → {node['name']}")
            conn_type = e["type"]
        else:
            path, conn_type = f"{graph['seed']} → {node['name']}", "unknown"
        result.append({"mbid": mbid, "name": node["name"], "depth": node["depth"],
                       "connection_path": path, "connection_type": conn_type})
    return result


def explain_path(graph: dict, artist_mbid: str) -> str:
    nodes, edges = graph["nodes"], graph["edges"]
    node = nodes.get(artist_mbid)
    if not node:
        return "Artist not found in graph."
    incoming = [e for e in edges if e["to"] == artist_mbid]
    if not incoming:
        return f"{node['name']} is {graph['seed']}."
    e = incoming[0]
    from_node = nodes.get(e["from"], {})
    role = f" ({', '.join(e['attributes'])})" if e.get("attributes") else ""
    return f"{node['name']} — {e['type']}{role} with {from_node.get('name', graph['seed'])}, associated with {graph['seed']}."