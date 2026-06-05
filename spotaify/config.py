import os
from dotenv import load_dotenv

load_dotenv()

_KEYRING_SERVICE = "spotaify"

# Secrets that benefit from keyring storage — tries keyring first, falls back to .env.
# Plain-text .env still works if keyring isn't set up.
def _get_secret(key: str) -> str:
    try:
        import keyring
        value = keyring.get_password(_KEYRING_SERVICE, key)
        if value:
            return value
    except Exception:
        pass
    return os.getenv(key, "")

SPOTIFY_CLIENT_ID     = os.getenv("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = _get_secret("SPOTIFY_CLIENT_SECRET")
SPOTIFY_REDIRECT_URI  = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:9090")
ANTHROPIC_API_KEY     = _get_secret("ANTHROPIC_API_KEY")
GENIUS_ACCESS_TOKEN   = _get_secret("GENIUS_ACCESS_TOKEN")
WHOSAMPLED_USERNAME   = os.getenv("WHOSAMPLED_USERNAME", "")
WHOSAMPLED_PASSWORD   = _get_secret("WHOSAMPLED_PASSWORD")
HISTORY_DIR           = os.getenv("HISTORY_DIR", "data/history")
GRAPH_DIR             = os.getenv("GRAPH_DIR", "data/graph")
MUSICBRAINZ_CONTACT   = os.getenv("MUSICBRAINZ_CONTACT", "")

# Propagate HF_TOKEN so huggingface_hub authenticates silently.
HF_TOKEN = _get_secret("HF_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
