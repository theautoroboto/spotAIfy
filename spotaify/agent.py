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

**Rediscovery mode**: Build a playlist from tracks the user historically loved but hasn't played recently. Call forgotten_favorites (passing stale_days and min_plays from the prompt), then rank_and_select (boost_discovery=False), then create_spotify_playlist. Optionally call get_taste_profile first to surface the user's audio fingerprint for context.

**Setlist mode**: Build a playlist from an artist's actual live repertoire.
1. Call get_setlist_tracks(artist_name) — returns songs ranked by `appearances` (how many shows featured them).
2. Call rank_and_select on the returned tracks. Weight high-appearances tracks as fan favourites; treat low-appearances tracks as rarities worth highlighting.
3. Call create_spotify_playlist.
Narrate which songs are live staples vs. rarities based on appearance count.

**Expand mode**: Discover new music beyond the user's current listening. Steps:
1. Call get_history_top_artists(n=8) to find the most-played artists.
2. Call map_artist_connections(depth=1) on the top 3-5 of those artists.
3. Call get_spotify_recommendations to get Spotify's recommendation engine output seeded from history.
4. Call fetch_artist_top_tracks on all unique connected artists from step 2.
5. Merge both candidate pools and call rank_and_select with boost_discovery=True, prioritising is_new_discovery=True tracks.
6. Call create_spotify_playlist.
Narrate why each artist or track expands beyond the user's existing taste.

**Enhance mode**: Analyze and extend an existing Spotify playlist.
1. Call get_playlist_tracks(playlist_id) to read the full contents.
2. Identify the mood, genre, era, and audio character from the track list.
3. Use search_catalog and/or fetch_artist_top_tracks to find similar tracks not already in the playlist.
4. Call rank_and_select on the candidates.
5. Call create_spotify_playlist with a name like "[original name] — Enhanced" and the new tracks only (do not duplicate the originals).
Narrate the vibe you identified and explain why each new track fits the playlist's character.

**Timeline mode**: Trace the evolution of a genre across years. Your output is a playlist and a musical argument.
1. Reason through the genre's history in the given year range. Identify approximately `count` inflection points — moments where the sound, production technique, culture, or technology shifted. Cluster them where change was rapid, space them where things held steady. Think in terms of causation: what broke from the past, what it inaugurated.
2. Issue search_catalog_by_year calls in batches of 10 per turn. Each turn: pick the next 10 inflection points, call search_catalog_by_year for each (as parallel tool calls within that turn), then continue to the next batch. Be precise — you are resolving known tracks to Spotify IDs. Set year_from/year_to to a 2–3 year window around the release.
3. For any inflection point that returned 0 results, issue fallback search_catalog calls (no year filter) — batch those as well, up to 10 per turn.
4. Once all inflection points are searched, collect all resolved track IDs in strict chronological order.
5. Call create_spotify_playlist with tracks in order, named "[Genre] [from_year]–[to_year]".
Narrate each inflection point: the year, what changed, why this track captures it, what it broke from or inaugurated. Do not use rank_and_select — curation is editorial and chronological here.

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


_MAX_TURNS = 25
_WRAP_UP_TURN = 21  # inject wrap-up nudge at this turn so the agent finishes cleanly

_WRAP_UP_MSG = (
    "You have gathered enough material. Do NOT call any more search or info tools. "
    "Call rank_and_select with all candidate tracks collected so far, then create_spotify_playlist. "
    "Finish the playlist now."
)

_WRAP_UP_MSG_TIMELINE = (
    "You have gathered enough material. Do NOT call any more search or info tools. "
    "Collect every track_id you have resolved so far and call create_spotify_playlist "
    "with them in strict chronological order. Do NOT call rank_and_select. Finish the playlist now."
)


def _fmt_call(name: str, inp: dict) -> str:
    if name == "search_catalog_by_year":   return f'Searching {inp["year_from"]}–{inp.get("year_to", inp["year_from"])}: "{inp["genre_query"]}"'
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
    if name == "get_playlist_tracks":
        return f'Reading playlist {inp["playlist_id"]}'
    if name == "forgotten_favorites":
        return f'Forgotten favorites (stale {inp.get("stale_days", 90)}d, min {inp.get("min_plays", 3)} plays)'
    if name == "get_taste_profile":
        return "Loading taste profile"
    if name == "get_setlist_tracks":
        return f'Setlist: {inp["artist_name"]} ({inp.get("setlist_pages", 3)} pages)'
    if name == "get_history_top_artists":
        return f'Top artists from history (n={inp.get("n", 10)})'
    if name == "get_spotify_recommendations":
        return f'Spotify recommendations (limit={inp.get("limit", 50)})'
    return f'{name}({json.dumps(inp)[:80]})'


def _fmt_result(name: str, result: dict) -> str:
    if "error" in result:               return f'Error: {result["error"]}'
    if name == "search_catalog_by_year":
        n = result.get("count", 0);     return f'{n} track{"s" if n != 1 else ""} found'
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
    if name == "get_playlist_tracks":
        n = result.get("count", 0)
        return f'{n} track{"s" if n != 1 else ""} loaded'
    if name == "forgotten_favorites":
        n = result.get("count", 0); total = result.get("total_found", n)
        return f'{n} tracks (of {total} forgotten)'
    if name == "get_taste_profile":
        fp = result.get("fingerprint", {})
        era = result.get("era_distribution", {})
        top = max(era, key=era.get) if era else "unknown"
        return f'energy={fp.get("energy", 0):.2f} valence={fp.get("valence", 0):.2f} · top era: {top}'
    if name == "get_setlist_tracks":
        n = result.get("count", 0)
        return f'{n} live track{"s" if n != 1 else ""} resolved'
    if name == "get_history_top_artists":
        n = result.get("count", 0)
        return f'{n} artists'
    if name == "get_spotify_recommendations":
        n = result.get("count", 0)
        return f'{n} recommendation{"s" if n != 1 else ""}'
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


def _run_liked_export(args) -> None:
    from spotaify.clients.spotify_client import SpotifyClient
    print("Fetching your liked songs...", flush=True)
    client = SpotifyClient(user_auth=True)
    tracks = client.liked_tracks()
    if not tracks:
        print("[error] No liked songs found in your Spotify library.", flush=True)
        return
    n = len(tracks)
    track_ids = [t["track_id"] for t in tracks]

    if args.liked_update_playlist:
        playlist_name = args.liked_update_playlist_name or args.liked_update_playlist
        print(f"Found {n} liked song{'s' if n != 1 else ''}. Checking '{playlist_name}' for duplicates...", flush=True)
        url, added = client.add_tracks_deduped(args.liked_update_playlist, track_ids)
        already = n - added
        if added == 0:
            print(f"All {n} liked songs are already in the playlist — nothing added.", flush=True)
        elif already > 0:
            print(f"Added {added} new song{'s' if added != 1 else ''} ({already} already present).", flush=True)
        else:
            print(f"Added {added} song{'s' if added != 1 else ''} to the playlist.", flush=True)
    else:
        print(f"Found {n} liked song{'s' if n != 1 else ''}. Creating playlist...", flush=True)
        name = args.liked_name or "My Liked Songs"
        description = (args.liked_description or f"All {n} liked songs exported from Spotify.")[:300]
        url = client.create_playlist(name, track_ids, public=args.liked_public, description=description)

    print(f"\nDone — {url}", flush=True)


def run_agent(prompt: str) -> None:
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not found in .env file.")

    is_timeline = prompt.startswith("Timeline mode")
    wrap_up_msg = _WRAP_UP_MSG_TIMELINE if is_timeline else _WRAP_UP_MSG
    # Timeline searches in batches of 10 across multiple turns — keep enough history
    # so track IDs from early batches are still visible at create_spotify_playlist time.
    keep_turns = 12 if is_timeline else 3

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    messages = [{"role": "user", "content": prompt}]
    print(f"\nPrompt: {prompt}\n{'─' * 60}", flush=True)

    turn = 0
    while turn < _MAX_TURNS:
        response = _call_with_retry(
            client,
            model="claude-sonnet-4-6",
            max_tokens=8192,
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

        if response.stop_reason == "max_tokens":
            # Response cut off mid-generation. If there are truncated tool_use blocks we
            # MUST provide tool_result blocks before sending anything else (API rule).
            # Execute whatever searches were generated, then fold the wrap-up into the
            # same user message so the agent can finish without more searches.
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            if tool_use_blocks:
                print(f"\n[Turn {turn}] Response truncated at {len(tool_use_blocks)} tool calls — executing then wrapping up.", flush=True)
                tool_results = []
                collected_ids: list[str] = []
                for block in tool_use_blocks:
                    print(f"\n-> {_fmt_call(block.name, block.input)}", flush=True)
                    result = execute_tool(block.name, block.input)
                    print(f"  <- {_fmt_result(block.name, result)}", flush=True)
                    # Collect track_ids from search results so we can pass them explicitly
                    # in the wrap-up — the model struggles to extract IDs from 50+ tool_results.
                    if block.name in ("search_catalog", "search_catalog_by_year"):
                        collected_ids.extend(
                            t["track_id"] for t in result.get("tracks", []) if t.get("track_id")
                        )
                    payload = json.dumps(_strip_embeddings(_cap_result(result)))
                    if len(payload) > _MAX_RESULT_CHARS:
                        payload = json.dumps({"error": "result truncated", "preview": payload[:500]})
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": payload,
                    })
                # For timeline mode, embed the collected IDs directly in the wrap-up so the
                # model doesn't have to re-parse dozens of tool_result blocks to find them.
                if is_timeline and collected_ids:
                    effective_wrap_up = (
                        wrap_up_msg
                        + f"\n\nTrack IDs resolved from the searches above ({len(collected_ids)} total)."
                        f" Use ALL of them, sorted in strict chronological order by their search"
                        f" window year: {json.dumps(collected_ids)}"
                    )
                else:
                    effective_wrap_up = wrap_up_msg
                # Append wrap-up as a text block in the same user message (valid per API spec)
                tool_results.append({"type": "text", "text": effective_wrap_up})
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
            else:
                # Only text was truncated — safe to just push wrap-up
                print(f"\n[Turn {turn}] Response truncated (max_tokens, text only) — injecting wrap-up.", flush=True)
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": wrap_up_msg})
            messages = _prune_messages(messages, keep_turns=keep_turns)
            turn += 1
            continue

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
            messages = _prune_messages(messages, keep_turns=keep_turns)
            turn += 1
            if turn == _WRAP_UP_TURN:
                print(f"\n[Turn {turn}] Injecting wrap-up nudge.", flush=True)
                messages.append({"role": "user", "content": wrap_up_msg})
        else:
            print(f"\n[Agent stopped: stop_reason={response.stop_reason}]", flush=True)
            break

    if turn >= _MAX_TURNS:
        print(f"\nReached {_MAX_TURNS}-turn limit — stopping.", flush=True)


def build_prompt(args: argparse.Namespace) -> str:
    parts = []

    if args.timeline:
        # Cap at 50 — searches run in batches of 10 per turn so no single API call
        # generates enough output tokens to hit rate limits.
        args.count = min(args.count, 50)
        parts.append(f"Timeline mode: trace the evolution of {args.timeline_genre} from {args.timeline_from} to {args.timeline_to}.")
        parts.append(f"Identify the key inflection points across those {args.timeline_to - args.timeline_from} years and find one defining track per moment.")

    elif args.enhance_playlist:
        name_clause = f" (named '{args.enhance_playlist_name}')" if args.enhance_playlist_name else ""
        parts.append(f"Enhance mode: analyze and extend the Spotify playlist with ID '{args.enhance_playlist}'{name_clause}.")
        parts.append("Call get_playlist_tracks to read its contents, identify the vibe and genre, then find matching tracks not already in the playlist and create a new enhanced playlist.")

    elif args.setlist:
        parts.append(f"Setlist mode: build a playlist from '{args.setlist_artist}'s live repertoire.")
        parts.append(f"Scan {args.setlist_pages} pages of recent setlists.")

    elif args.expand:
        if args.expand_year:
            parts.append(f"Expand mode: discover new music I haven't heard before, seeded from my top artists of {args.expand_year}.")
            parts.append(f"Call get_history_top_artists with year={args.expand_year} to get the seed artists, then map their connections and pull Spotify recommendations.")
        else:
            parts.append("Expand mode: discover new music I haven't heard before, using my all-time top artists as a springboard. Map connected artists and pull Spotify recommendations.")

    elif args.rediscovery:
        parts.append(f"Rediscovery mode: build a playlist from tracks I used to love but haven't played in at least {args.stale_days} days.")
        parts.append(f"Use minimum {args.min_plays} past plays as the loved threshold.")

    elif args.dna:
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
    parser.add_argument("--rediscovery", action="store_true", help="Build rediscovery playlist from listening history")
    parser.add_argument("--expand", action="store_true", help="Discover new music beyond current listening history")
    parser.add_argument("--expand-year", type=int, default=None, dest="expand_year", help="Seed expand mode from top artists of this year")
    parser.add_argument("--setlist", action="store_true", help="Build playlist from artist's live setlists")
    parser.add_argument("--setlist-artist", default="", dest="setlist_artist", help="Artist name for setlist mode")
    parser.add_argument("--setlist-pages", type=int, default=3, dest="setlist_pages", help="Number of setlist pages to scan (1-5)")
    parser.add_argument("--stale-days", type=int, default=90, dest="stale_days", help="Days since last play threshold")
    parser.add_argument("--min-plays", type=int, default=3, dest="min_plays", help="Minimum past play count")
    parser.add_argument("--count", type=int, default=20, help="Target playlist length")
    parser.add_argument("--enhance-playlist", default="", dest="enhance_playlist", metavar="PLAYLIST_ID", help="Analyze and extend an existing Spotify playlist")
    parser.add_argument("--enhance-playlist-name", default="", dest="enhance_playlist_name", help="Display name of the playlist being enhanced")
    parser.add_argument("--liked-export", action="store_true", dest="liked_export", help="Export all liked songs to a new playlist")
    parser.add_argument("--liked-name", default="My Liked Songs", dest="liked_name", help="Name for the exported liked-songs playlist")
    parser.add_argument("--liked-public", action="store_true", dest="liked_public", help="Make the exported playlist public")
    parser.add_argument("--liked-description", default="", dest="liked_description", help="Description for the exported playlist")
    parser.add_argument("--liked-update-playlist", default="", dest="liked_update_playlist", metavar="PLAYLIST_ID", help="Replace tracks in an existing playlist with liked songs")
    parser.add_argument("--liked-update-playlist-name", default="", dest="liked_update_playlist_name", help="Display name of the playlist being updated")
    parser.add_argument("--timeline", action="store_true", help="Build a genre evolution timeline playlist")
    parser.add_argument("--timeline-genre", default="", dest="timeline_genre", help="Genre to trace")
    parser.add_argument("--timeline-from", type=int, default=1980, dest="timeline_from", help="Start year")
    parser.add_argument("--timeline-to", type=int, default=2020, dest="timeline_to", help="End year")
    args = parser.parse_args()

    if args.liked_export:
        try:
            _run_liked_export(args)
        except Exception as e:
            print(f"\n[error] {e}")
        return

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