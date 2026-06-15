"""
Store sensitive credentials in the OS keychain.

  Windows  → Credential Manager (encrypted with your Windows login)
  macOS    → Keychain
  Linux    → Secret Service (GNOME Keyring / KWallet)

Secrets stored here are never written to disk as plain text.
Plain-text .env still works as a fallback if you prefer it.

Usage:
    uv run python setup_secrets.py
"""
import getpass
import sys

try:
    import keyring
except ImportError:
    print("ERROR: keyring is not installed. Run: uv sync")
    sys.exit(1)

_SERVICE = "spotaify"

_SECRETS = [
    ("ANTHROPIC_API_KEY",     "Anthropic API key",                        True),
    ("SPOTIFY_CLIENT_SECRET", "Spotify client secret",                    True),
    ("GENIUS_ACCESS_TOKEN",   "Genius access token",                      True),
    ("HF_TOKEN",              "Hugging Face token",                       False),
    ("WHOSAMPLED_PASSWORD",   "WhoSampled password",                      False),
]


def _prompt(label: str, required: bool, already_set: bool) -> str:
    suffix = ""
    if already_set:
        suffix = " [already set — press Enter to keep]"
    elif not required:
        suffix = " [optional — press Enter to skip]"
    return getpass.getpass(f"{label}{suffix}: ")


def main() -> None:
    print("spotAIfy secret setup")
    print("Credentials are stored in your OS keychain — not in any file.\n")

    changed = 0
    for key, label, required in _SECRETS:
        existing = keyring.get_password(_SERVICE, key)
        value = _prompt(label, required, bool(existing))
        if value:
            keyring.set_password(_SERVICE, key, value)
            print(f"  ✓ {key} saved to keychain\n")
            changed += 1
        elif existing:
            print(f"  — {key} unchanged\n")
        else:
            print(f"  — {key} skipped\n")

    if changed:
        print(f"{changed} secret(s) updated. Run this script again any time to rotate credentials.")
    else:
        print("No changes made.")


if __name__ == "__main__":
    main()
