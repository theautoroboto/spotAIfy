# whosampled_client.py
import random
import re
import time
from urllib.parse import quote
from curl_cffi import requests as cf_requests
from bs4 import BeautifulSoup
from spotaify.config import WHOSAMPLED_USERNAME, WHOSAMPLED_PASSWORD

_BASE = "https://www.whosampled.com"
_DELAY = 2.0
_TRACK_HREF = re.compile(r"^/[^/]+/[^/]+/$")
_ARTIST_HREF = re.compile(r"^/[^/]+/$")

_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Sec-CH-UA": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
    "Sec-CH-UA-Mobile": "?0",
    "Sec-CH-UA-Platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
    "DNT": "1",
}

_session: cf_requests.Session | None = None
_session_ready: bool = False


def _get_session() -> cf_requests.Session:
    global _session
    if _session is None:
        _session = cf_requests.Session(impersonate="chrome124")
        _session.headers.update(_HEADERS)
    return _session


def _login(session: cf_requests.Session) -> bool:
    """POST credentials to WhoSampled. Returns True on success."""
    if not WHOSAMPLED_USERNAME or not WHOSAMPLED_PASSWORD:
        return False
    try:
        login_page = session.get(
            _BASE + "/user/login/", timeout=20, headers={"Referer": _BASE + "/"}
        )
        soup = BeautifulSoup(login_page.text, "html.parser")
        csrf = ""
        token_input = soup.find("input", {"name": "csrfmiddlewaretoken"})
        if token_input:
            csrf = token_input.get("value", "")
        resp = session.post(
            _BASE + "/user/login/",
            data={
                "csrfmiddlewaretoken": csrf,
                "username": WHOSAMPLED_USERNAME,
                "password": WHOSAMPLED_PASSWORD,
                "next": "/",
            },
            headers={
                "Referer": _BASE + "/user/login/",
                "Origin": _BASE,
                "Sec-Fetch-Site": "same-origin",
            },
            timeout=20,
            allow_redirects=True,
        )
        return "/user/login" not in resp.url
    except Exception:
        return False


def _ensure_session() -> None:
    """Warm the session and authenticate if credentials are configured."""
    global _session_ready
    if _session_ready:
        return
    session = _get_session()
    try:
        session.get(_BASE + "/", timeout=20, allow_redirects=True)
        time.sleep(1.5)
    except Exception:
        pass
    if WHOSAMPLED_USERNAME and WHOSAMPLED_PASSWORD:
        _login(session)
        time.sleep(1.0)
    _session_ready = True


def _get(url: str, referer: str | None = None, _retry: bool = True) -> cf_requests.Response | None:
    global _session, _session_ready
    _ensure_session()
    time.sleep(_DELAY + random.uniform(0, 1.5))
    headers = {"Referer": referer or _BASE + "/", "Sec-Fetch-Site": "same-origin"}
    try:
        resp = _get_session().get(url, timeout=20, allow_redirects=True, headers=headers)
        if resp.status_code == 403 and _retry:
            _session = None
            _session_ready = False
            time.sleep(3 + random.uniform(0, 2))
            return _get(url, referer, _retry=False)
        return resp
    except Exception:
        return None


def _is_login_wall(resp) -> bool:
    return resp is not None and "/user/login" in getattr(resp, "url", "")


def _slugify(text: str) -> str:
    text = re.sub(r"&", "and", text.strip())
    text = re.sub(r"[^\w\s-]", "", text)
    parts = re.split(r"[\s-]+", text.strip())
    return "-".join(p.capitalize() for p in parts if p)


def _parse_entries(section) -> list[dict]:
    entries = []
    for entry in section.find_all(True, class_=re.compile(r"listEntry|trackEntry|entry", re.I)):
        track_a = artist_a = None
        for a in entry.find_all("a", href=_TRACK_HREF):
            track_a = a
            break
        for a in entry.find_all("a", href=_ARTIST_HREF):
            artist_a = a
            break
        if track_a:
            entries.append({
                "title": track_a.get_text(strip=True),
                "artist": artist_a.get_text(strip=True) if artist_a else "",
                "whosampled_url": _BASE + track_a["href"],
            })
    return entries


def _find_section(soup: BeautifulSoup, keyword: str):
    for tag in soup.find_all(string=re.compile(keyword, re.I)):
        parent = tag.find_parent("section") or tag.find_parent(
            class_=re.compile(r"section|block", re.I)
        )
        if parent:
            return parent
    return None


def _page_to_samples(soup: BeautifulSoup) -> tuple[list, list]:
    samples_sec = _find_section(soup, "contains samples")
    sampled_by_sec = _find_section(soup, "sampled by")
    return (
        _parse_entries(samples_sec) if samples_sec else [],
        _parse_entries(sampled_by_sec) if sampled_by_sec else [],
    )


def _try_search_fallback(title: str, artist: str, original_url: str) -> dict:
    """Fall back to WhoSampled search when the direct URL 404s or hits a login wall."""
    search_url = f"{_BASE}/search/?q={quote(f'{title} {artist}')}"
    resp = _get(search_url, referer=_BASE + "/")
    if resp is None:
        return {"error": "WhoSampled unreachable (connection failed)", "samples": [], "sampled_by": [],
                "url": original_url, "sample_count": 0, "sampled_by_count": 0}
    if _is_login_wall(resp):
        return {"error": "WhoSampled requires login", "samples": [], "sampled_by": [],
                "url": original_url, "sample_count": 0, "sampled_by_count": 0}
    if resp.status_code != 200:
        return {"error": f"WhoSampled search blocked (HTTP {resp.status_code})", "samples": [], "sampled_by": [],
                "url": original_url, "sample_count": 0, "sampled_by_count": 0}
    soup = BeautifulSoup(resp.text, "html.parser")
    first = soup.find("a", href=_TRACK_HREF)
    if not first:
        return {"error": "Track not found on WhoSampled", "samples": [], "sampled_by": [],
                "url": search_url, "sample_count": 0, "sampled_by_count": 0}
    track_url = _BASE + first["href"]
    resp = _get(track_url, referer=search_url)
    if _is_login_wall(resp):
        return {"error": "WhoSampled requires login for this track", "samples": [], "sampled_by": [],
                "url": track_url, "sample_count": 0, "sampled_by_count": 0}
    return None, resp, track_url  # signal: caller should proceed with this resp


def fetch_samples(title: str, artist: str) -> dict:
    url = f"{_BASE}/{_slugify(artist)}/{_slugify(title)}/"
    resp = _get(url, referer=_BASE + "/")

    if resp is not None and resp.status_code == 429:
        retry = resp.headers.get("Retry-After", "?")
        raise RuntimeError(f"WhoSampled rate limit — retry after {retry}s")

    if resp is None or resp.status_code in (403, 404) or _is_login_wall(resp):
        fallback = _try_search_fallback(title, artist, url)
        if isinstance(fallback, dict):
            return fallback
        _, resp, url = fallback

    if resp is None or resp.status_code != 200:
        code = getattr(resp, "status_code", "?")
        return {"error": f"WhoSampled HTTP {code}", "samples": [], "sampled_by": [],
                "url": url, "sample_count": 0, "sampled_by_count": 0}

    soup = BeautifulSoup(resp.text, "html.parser")
    samples, sampled_by = _page_to_samples(soup)

    return {
        "url": url,
        "samples": samples[:8],
        "sampled_by": sampled_by[:5],
        "sample_count": len(samples),
        "sampled_by_count": len(sampled_by),
    }
