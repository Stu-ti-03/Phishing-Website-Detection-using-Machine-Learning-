"""
utils/preprocess.py
-------------------
URL pre-processing helpers.
"""

import re
from urllib.parse import urlparse


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

_SCHEME_RE = re.compile(r"^https?://", re.IGNORECASE)
_SPACE_RE  = re.compile(r"\s+")

# IMPROVED LOGIC: Reject URLs with these clearly invalid character patterns.
# Real URLs never contain spaces, backticks, or raw angle brackets.
_INVALID_CHARS_RE = re.compile(r"[`<>\"\'\\\s]")


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def clean_url(raw_url: str) -> str:
    """
    Normalise a raw URL string:
      1. Strip surrounding whitespace.
      2. Lowercase.
      3. Remove internal spaces.
      4. Remove http:// / https://.
      5. Remove leading www.

    # VIVA EXPLANATION:
    # Cleaning ensures that "HTTP://WWW.Google.Com/Mail" and
    # "google.com/mail" both produce the same feature values,
    # so the model isn't confused by cosmetic differences.
    """
    url = raw_url.strip().lower()
    url = _SPACE_RE.sub("", url)          # remove internal spaces
    url = _SCHEME_RE.sub("", url)         # strip scheme
    url = re.sub(r"^www\.", "", url)      # strip leading www.
    return url


def is_valid_url(url: str) -> bool:
    """
    IMPROVED LOGIC: Enhanced validity check with:
      1. Minimum length guard.
      2. Requires at least one dot (hostname has a TLD).
      3. Rejects clearly invalid characters (< > " ` backslash).
      4. Hostname must not be empty after urlparse.
      5. Hostname must contain only valid label characters
         (letters, digits, hyphens, dots).

    # VIVA EXPLANATION:
    # Stronger validation reduces garbage-in-garbage-out errors.
    # If we let junk strings reach the model, it may give meaningless
    # probability scores, misleading the user.
    # Each check adds a layer of sanity before touching the ML model.
    """
    # Guard 1: too short to be a real URL
    if not url or len(url) < 4:
        return False

    # Guard 2: must contain at least one dot (e.g. "example.com")
    if "." not in url:
        return False

    # NEW FEATURE: Guard 3 – reject URLs with obviously invalid characters
    if _INVALID_CHARS_RE.search(url):
        return False

    # Guard 4: attempt urlparse to extract hostname
    try:
        parsed = urlparse("http://" + url)
        hostname = parsed.netloc
        if not hostname:
            return False
    except Exception:
        return False

    # NEW FEATURE: Guard 5 – hostname must only contain valid characters
    # Valid hostname chars: letters, digits, hyphen, dot
    hostname_clean = hostname.split(":")[0]   # strip port if present
    if not re.match(r"^[a-z0-9.\-]+$", hostname_clean):
        return False

    return True
