# agent.py
import argparse
import json
import re
import sys
import time
import anthropic

sys.stdout.reconfigure(encoding="utf-8")
print("Initializing...", flush=True)
from spotaify.agent_tools import TOOLS, execute_tool
from spotaify.config import ANTHROPIC_API_KEY

_SYSTEM_PROMPT_TEXT = """You are an expert music curator. You build playlists in three modes:

**Sonic mode**: Find tracks by similarity across genres. Use search_catalog across multiple targeted keyword queries to build a candidate pool, then rank_and_select. Refine via keyword searches.

**Connection mode**: Map every artist connected to a seed (band members, side projects, collabs) via MusicBrainz. Call map_artist_connections at depth=1 (expand to depth=2 only if fewer than 8 artists are returned), then fetch_artist_top_tracks on at most 15 artists, then rank_and_select. Narrate connections in plain English.

**DNA mode**: Deep-dive a single track's creative DNA. Call fetch_whosampled(queried track) exactly once. Its response has two fields:
- `samples`: recordings the queried track pulled material from → call resolve_samples on this list → include those tracks (the queried track borrows from them)
- `sampled_by`: recordings that pulled material from the queried track → call resolve_samples on this list → include those tracks (they borrow from the queried track)

CRITICAL: Do NOT call fetch_whosampled again on any of the resolved tracks. Their own sample history is irrelevant — you are mapping connections to the queried track only, not building a sample graph. Do not include songs simply because they share a common source with the queried track.

Also call fetch_genius_info and deep_dive_track on the queried track to surface any interpolated melodies; include the original recording of a confirmed interpolation (not covers, not thematic similarities).

Do NOT include: other songs by the same producer/composer/musician, songs that sample the same source as the queried track, or any track without a direct musical-DNA link to the queried track. For each included track, state exactly how it connects (e.g. "Firestarter samples the break from X" or "X was later sampled by Firestarter").

Rules:
- Always include the seed/queried track as the first track in the playlist
- Always end by calling create_spotify_playlist with the final track IDs and a descriptive name and description
- The description field in create_spotify_playlist must be ≤300 characters and a complete sentence — no truncation. Write a tight, self-contained summary. Put all verbose explanation in your narration text to stdout instead.
- For new discoveries (is_new_discovery=True), write a one-sentence introduction
- Narrate your reasoning as you go
- Never explain what tools do — just use them and describe results
- Write all narrative text as plain prose. No markdown: no *, **, _, #, or em-dashes. The output is displayed as plain text."""

SYSTEM_PROMPT = [{"type": "text", "text": _SYSTEM_PROMPT_TEXT, "cache_control": {"type": "ephemeral"}}]


def _strip_embeddings(obj):
    if isinstance(obj, dict):
        return {k: _strip_embeddings(v) for k, v in obj.items()
                if k not in ("audio_embedding", "text_embedding")}
    if isinstance(obj, list):
        return [_strip_embeddings(i) for i in obj]
    return obj


_MAX_RESULT_CHARS = 12_000


def _cap_result(result: dict) -> dict:
    """If a result has a large list, trim it to keep token count in check."""
    for key in ("tracks", "artists", "playlist"):
        if isinstance(result.get(key), list) and len(result[key]) > 15:
            result = dict(result)
            result[key] = result[key][:15]
            result["_truncated"] = True
    return result


_MAX_TURNS = 12
_WRAP_UP_TURN = 9  # inject wrap-up nudge at this turn so the agent finishes cleanly

_WRAP_UP_MSG = (
    "You have gathered enough material. Do NOT call any more search or info tools. "
    "Call rank_and_select with all candidate tracks collected so far, then create_spotify_playlist. "
    "Finish the playlist now."
)


def _fmt_call(name: str, inp: dict) -> str:
    if name == "search_catalog":          return f'Searching: "{inp["query"]}"'
    if name == "fetch_whosampled":        return f'WhoSampled: "{inp["title"]}" by {inp["artist"]}'
    if name == "fetch_genius_info":       return f'Genius: "{inp["title"]}" by {inp["artist"]}'
    if name == "deep_dive_track":         return f'Deep dive: "{inp["title"]}" by {inp["artist"]}'
    if name == "map_artist_connections":  return f'Artist graph: {inp["artist_name"]} (depth {inp.get("depth", 2)})'
    if name == "traverse_artist_graph":   return f'Traversing graph: {inp["artist_name"]}'
    if name == "fetch_artist_top_tracks":
        artists = inp.get("artist_names", [])
        label = ", ".join(artists[:3]) + (f" +{len(artists)-3} more" if len(artists) > 3 else "")
        return f'Top tracks: {label}'
    if name == "resolve_samples":
        n = len(inp.get("samples", []))
        return f'Resolving {n} sample{"s" if n != 1 else ""} on Spotify'
    if name == "rank_and_select":
        return f'Ranking {len(inp.get("candidates", []))} candidates → selecting {inp.get("count", 20)}'
    if name == "find_tracks_by_person":
        role = inp.get("role", "")
        return f'Tracks by {inp["person_name"]}{f" ({role})" if role else ""}'
    if name == "create_spotify_playlist": return f'Creating playlist: "{inp["name"]}"'
    return f'{name}({json.dumps(inp)[:80]})'


def _fmt_result(name: str, result: dict) -> str:
    if "error" in result:               return f'Error: {result["error"]}'
    if name == "search_catalog":
        n = result.get("count", 0);     return f'{n} track{"s" if n != 1 else ""} found'
    if name == "fetch_whosampled":
        sc = result.get("sample_count", 0); sbc = result.get("sampled_by_count", 0)
        return f'{sc} sample{"s" if sc != 1 else ""} · {sbc} sampled-by'
    if name == "fetch_genius_info":
        about = result.get("about", "").strip()
        if about and about != "?":
            return (about[:120] + "…") if len(about) > 120 else about
        parts = []
        if result.get("writers"):   parts.append("Writers: " + ", ".join(result["writers"][:3]))
        if result.get("producers"): parts.append("Producers: " + ", ".join(result["producers"][:3]))
        return " · ".join(parts) if parts else "No description"
    if name in ("map_artist_connections", "traverse_artist_graph"):
        n = result.get("total") or len(result.get("artists", []))
        return f'{n} artists in graph'
    if name == "fetch_artist_top_tracks":
        n = result.get("count", 0);     return f'{n} track{"s" if n != 1 else ""}'
    if name == "resolve_samples":
        n = result.get("count", 0);     return f'{n} resolved on Spotify'
    if name == "rank_and_select":
        n = len(result.get("playlist", [])); return f'{n} tracks selected'
    if name == "find_tracks_by_person":
        n = result.get("count", 0);     return f'{n} track{"s" if n != 1 else ""} found'
    if name == "create_spotify_playlist":
        return f'Done — {result.get("url", "")}'
    if name == "deep_dive_track":
        parts = []
        for key, label in [("composers", "composer"), ("producers", "producer"),
                            ("session_musicians", "musician"), ("samples", "sample")]:
            n = len(result.get(key, []))
            if n: parts.append(f'{n} {label}{"s" if n != 1 else ""}')
        return ", ".join(parts) if parts else "No credits found"
    return str(result)[:120]


def _prune_messages(messages: list, keep_turns: int = 3) -> list:
    """Keep the first user message plus the most recent keep_turns assistant/user pairs.

    Always starts the tail from an assistant message so that tool_result blocks
    are never left without their corresponding tool_use block.
    """
    if len(messages) <= keep_turns * 2 + 1:
        return messages
    tail = messages[-(keep_turns * 2):]
    # Drop leading user messages whose tool_use partner was pruned away.
    while tail and tail[0]["role"] == "user":
        content = tail[0].get("content", "")
        if isinstance(content, list) and any(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in content
        ):
            tail = tail[1:]
        else:
            break
    return [messages[0]] + tail


_MAX_RETRY_WAIT = 90  # seconds — don't burn time waiting on daily/usage limits


def _retry_after(e: anthropic.RateLimitError) -> int | None:
    """Extract Retry-After seconds from headers or the error message body."""
    try:
        val = e.response.headers.get("retry-after") or e.response.headers.get("x-ratelimit-reset-requests")
        if val:
            return int(float(val))
    except Exception:
        pass
    # Anthropic sometimes embeds the wait in the message body rather than headers.
    m = re.search(r"retry.*?after[:\s]+(\d+)\s*s", str(e), re.I)
    return int(m.group(1)) if m else None


def _call_with_retry(client, **kwargs) -> object:
    """Call client.messages.create with backoff on rate limit errors.

    If the API's retry-after exceeds _MAX_RETRY_WAIT (i.e. a daily/usage cap,
    not a per-minute burst limit), raise immediately so the caller sees a clear
    message rather than silently burning retries.
    """
    delay = 65  # Start past the 1-minute rate-limit window
    for attempt in range(4):
        try:
            return client.messages.create(**kwargs)
        except anthropic.RateLimitError as e:
            wait = _retry_after(e)
            if wait is not None and wait > _MAX_RETRY_WAIT:
                raise RuntimeError(
                    f"Anthropic usage/daily limit hit — retry after {wait}s (~{wait // 60} min). "
                    "Check your usage at console.anthropic.com."
                ) from e
            if attempt == 3:
                raise
            actual_delay = wait if (wait and wait <= _MAX_RETRY_WAIT) else delay
            print(f"\nRate limit hit — waiting {actual_delay}s before retry {attempt + 1}/3...", flush=True)
            time.sleep(actual_delay)
            delay = min(delay + 30, 120)


def run_agent(prompt: str) -> None:
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not found in .env file.")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    messages = [{"role": "user", "content": prompt}]
    print(f"\nPrompt: {prompt}\n{'─' * 60}", flush=True)

    turn = 0
    while turn < _MAX_TURNS:
        response = _call_with_retry(
            client,
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        text_blocks = [block.text for block in response.content if hasattr(block, "text")]
        if text_blocks:
            print("\n" + "".join(text_blocks), flush=True)

        if response.stop_reason == "end_turn":
            print("\nAgent has finished.", flush=True)
            break

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    print(f"\n-> {_fmt_call(tool_name, tool_input)}", flush=True)
                    result = execute_tool(tool_name, tool_input)
                    print(f"  <- {_fmt_result(tool_name, result)}", flush=True)
                    payload = json.dumps(_strip_embeddings(_cap_result(result)))
                    if len(payload) > _MAX_RESULT_CHARS:
                        payload = json.dumps({"error": "result truncated", "preview": payload[:500]})
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": payload,
                    })
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
            messages = _prune_messages(messages)
            turn += 1
            if turn == _WRAP_UP_TURN:
                print(f"\n[Turn {turn}] Injecting wrap-up nudge.", flush=True)
                messages.append({"role": "user", "content": _WRAP_UP_MSG})
        else:
            break

    if turn >= _MAX_TURNS:
        print(f"\nReached {_MAX_TURNS}-turn limit — stopping.", flush=True)


def build_prompt(args: argparse.Namespace) -> str:
    parts = []

    if args.dna:
        track_ref = f"'{args.dna}' by '{args.dna_artist}'" if args.dna_artist else f"'{args.dna}'"
        parts.append(f"DNA mode: deep-dive the track {track_ref}. Include: (1) every recording it directly samples, (2) every recording that directly sampled it, (3) any original whose melody it confirmed interpolates. Nothing else.")

    elif args.artist and args.connections:
        parts.append(f"Connection mode: build a playlist of every artist associated with '{args.artist}'.")
        parts.append(f"Traverse up to depth {args.depth}.")
        if args.new_only:
            parts.append("Prefer artists not in my listening history (is_new_discovery=True).")
        if args.match_sound:
            parts.append("Also filter by audio similarity to the seed artist's sound.")

    elif args.connect:
        parts.append(f"Show the connection path between '{args.connect[0]}' and '{args.connect[1]}', then build a playlist that bridges both artists' constellations.")

    if args.seed:
        seeds = ", ".join(f'"{s}"' for s in args.seed)
        parts.append(f"Use these as seed tracks or artists: {seeds}. Expand by similarity.")

    if args.prompt:
        parts.append(args.prompt)

    for constraint, label in [
        (args.energy, "energy"), (args.valence, "valence"), (args.tempo, "tempo BPM")
    ]:
        if constraint and "-" in constraint:
            lo, hi = constraint.split("-", 1)
            parts.append(f"{label} between {lo} and {hi}")

    parts.append(f"Target playlist length: {args.count} tracks.")
    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="AI Playlist Agent")
    parser.add_argument("prompt", nargs="?", default="", help="Natural language playlist description")
    parser.add_argument("--dna", metavar="TRACK", help="Deep-dive a track's creative DNA")
    parser.add_argument("--dna-artist", metavar="ARTIST", default="", dest="dna_artist", help="Artist for DNA track disambiguation")
    parser.add_argument("--artist", help="Seed artist name for connection mode")
    parser.add_argument("--connections", action="store_true", help="Build from artist connection graph")
    parser.add_argument("--depth", type=int, default=2, help="Connection/DNA traversal depth")
    parser.add_argument("--new-only", action="store_true", dest="new_only", help="Discovery artists only")
    parser.add_argument("--match-sound", action="store_true", dest="match_sound", help="Filter by audio similarity")
    parser.add_argument("--connect", nargs=2, metavar=("ARTIST1", "ARTIST2"))
    parser.add_argument("--seed", nargs="+", help="Seed tracks or artists")
    parser.add_argument("--energy", help="Energy range e.g. 0.7-1.0")
    parser.add_argument("--valence", help="Valence range e.g. 0.3-0.6")
    parser.add_argument("--tempo", help="Tempo range e.g. 120-180")
    parser.add_argument("--count", type=int, default=20, help="Target playlist length")
    args = parser.parse_args()

    prompt = build_prompt(args)
    if not prompt.strip():
        parser.print_help()
        return

    try:
        run_agent(prompt)
    except Exception as e:
        print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    main()