"""
utils/features.py
-----------------
URL feature extraction for the ML model.

Feature vector (order matters – must EXACTLY match training in train_model.py):

    Index  Name                   Description
    ─────  ─────────────────────  ──────────────────────────────────────────────
      0    url_length             Total character count
      1    num_dots               Count of '.' in URL
      2    num_hyphens            Count of '-' in URL
      3    num_digits             Count of digit characters
      4    has_suspicious_keyword 1 if suspicious word found, else 0
      5    subdomain_count        Number of subdomain levels
      6    has_https              1 if original URL uses HTTPS, else 0  ← NEW
      7    url_entropy            Shannon entropy of URL characters      ← NEW
      8    special_char_count     Count of %, ?, =, _ characters        ← NEW

# VIVA EXPLANATION:
# Adding has_https, url_entropy, and special_char_count improves detection because:
#   - Phishing sites often avoid HTTPS (it costs money and leaves logs).
#   - Auto-generated phishing URLs have high randomness (entropy), unlike
#     readable human-made domain names.
#   - Special characters like %, ?, = are used to inject fake parameters
#     or confuse URL parsers.
"""

import re
import math
from urllib.parse import urlparse
from typing import List

# ──────────────────────────────────────────────
# Suspicious keyword list
# ──────────────────────────────────────────────

SUSPICIOUS_KEYWORDS = {
    "login", "verify", "verification", "bank", "secure", "account",
    "update", "confirm", "password", "signin", "paypal", "ebay",
    "amazon", "apple", "microsoft", "support", "alert", "suspended",
    "unlock", "validate", "credential", "wallet", "free", "gift",
    "prize", "lucky", "click", "limited", "offer",
}


# ──────────────────────────────────────────────
# Original feature extractors (preserved)
# ──────────────────────────────────────────────

def url_length(url: str) -> int:
    """Total character length of the cleaned URL."""
    return len(url)


def num_dots(url: str) -> int:
    """Count of '.' characters – more dots often mean deeper subdomains."""
    return url.count(".")


def num_hyphens(url: str) -> int:
    """Count of '-' characters – phishing URLs often abuse hyphens."""
    return url.count("-")


def num_digits(url: str) -> int:
    """Count of digit characters in the full URL string."""
    return sum(c.isdigit() for c in url)


def has_suspicious_keyword(url: str) -> int:
    """
    Returns 1 if the URL contains any token from the suspicious keyword list,
    0 otherwise.
    """
    tokens = re.split(r"[.\-/_?=&]", url.lower())
    for token in tokens:
        if token in SUSPICIOUS_KEYWORDS:
            return 1
    return 0


def subdomain_count(url: str) -> int:
    """
    Counts the number of subdomain levels.
    e.g. "sub1.sub2.example.com" → 2 subdomains
    """
    try:
        parsed = urlparse("http://" + url)
        host = parsed.hostname or ""
        parts = host.split(".")
        count = max(len(parts) - 2, 0)
        return count
    except Exception:
        return 0


# ──────────────────────────────────────────────
# NEW FEATURE: has_https
# ──────────────────────────────────────────────

def has_https(raw_url: str) -> int:
    """
    Returns 1 if the ORIGINAL (uncleaned) URL starts with 'https://', 0 otherwise.

    # VIVA EXPLANATION:
    # The clean_url() function strips the scheme for feature processing,
    # so we check HTTPS from the raw URL before cleaning.
    # Legitimate sites almost always use HTTPS. Phishing pages often use HTTP
    # to avoid the cost and traceability of SSL certificates.

    NOTE: This function must be called with the RAW url (before clean_url()),
    or pass the raw_url separately — see extract_features() for usage.
    """
    return 1 if raw_url.lower().startswith("https://") else 0


# ──────────────────────────────────────────────
# NEW FEATURE: url_entropy
# ──────────────────────────────────────────────

def url_entropy(url: str) -> float:
    """
    Compute the Shannon entropy of the URL string.

    Shannon entropy measures information randomness:
        H = -Σ p(c) * log2(p(c))
    where p(c) is the frequency of character c in the string.

    # VIVA EXPLANATION:
    # Human-readable URLs like "google.com/mail" have low entropy (repetitive chars).
    # Auto-generated phishing URLs like "a7Xk29mQ.tk/p=1&r=0x3" have HIGH entropy
    # because they use many different characters randomly.
    # High entropy is a strong signal of machine-generated (phishing) URLs.
    """
    if not url:
        return 0.0
    freq = {}
    for char in url:
        freq[char] = freq.get(char, 0) + 1
    total = len(url)
    entropy = -sum((count / total) * math.log2(count / total)
                   for count in freq.values())
    return round(entropy, 4)


# ──────────────────────────────────────────────
# NEW FEATURE: special_char_count
# ──────────────────────────────────────────────

def special_char_count(url: str) -> int:
    """
    Count the number of special characters commonly abused in phishing URLs:
        % → URL encoding tricks (e.g. %2F instead of /)
        ? → Query string injection
        = → Parameter assignment used to mimic login pages
        _ → Used in fake subdomain names (e.g. secure_login.evil.com)

    # VIVA EXPLANATION:
    # Phishers use URL encoding (%XX) to hide malicious paths from scanners.
    # Multiple = and ? are also used to craft fake-looking login URLs that
    # appear legitimate at first glance.
    """
    special_chars = set("%?=_")
    return sum(1 for c in url if c in special_chars)


# ──────────────────────────────────────────────
# Composite feature vector  (UPDATED)
# ──────────────────────────────────────────────

def extract_features(url: str, raw_url: str = None) -> List[float]:
    """
    Return the ordered feature vector for a URL.

    Args:
        url     : Pre-cleaned URL (output of clean_url()).
        raw_url : Original URL string (needed for has_https check).
                  If not provided, has_https will always return 0.

    # VIVA EXPLANATION:
    # Feature order MUST match the column order used during training.
    # If order changes, the model reads wrong features → garbage predictions.
    # That's why FEATURE_NAMES in train_model.py mirrors this list exactly.
    """
    return [
        url_length(url),                                  # 0
        num_dots(url),                                    # 1
        num_hyphens(url),                                 # 2
        num_digits(url),                                  # 3
        has_suspicious_keyword(url),                      # 4
        subdomain_count(url),                             # 5
        has_https(raw_url if raw_url else ""),            # 6  NEW FEATURE
        url_entropy(url),                                 # 7  NEW FEATURE
        special_char_count(url),                          # 8  NEW FEATURE
    ]
