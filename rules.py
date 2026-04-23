"""
utils/rules.py
--------------
Hybrid rule-based engine that works ALONGSIDE the ML model.

IMPROVED LOGIC:
    Rules no longer FORCE a "Phishing" result.
    Instead, each triggered rule BOOSTS the phishing probability by a fixed amount.
    The final probability is capped at 1.0.

    Example: ML gives 0.45 (uncertain), rule fires → 0.45 + 0.20 = 0.65 → Phishing.

# VIVA EXPLANATION:
# The old approach (force phish_prob = 0.80) caused FALSE POSITIVES:
#   - A URL like "bank-of-america.com" triggered the hyphens rule and was
#     immediately called phishing, even though it's a legitimate domain.
# The new BOOST approach respects the ML model's confidence. A rule only
# nudges the probability up — if the ML is very confident it's legitimate
# (prob = 0.10), one rule bump (→ 0.30) still won't cross the phishing threshold.
# This means BOTH the rule AND the model must agree to flag something as phishing,
# dramatically reducing false positives while keeping true positive rate high.
"""

import re
from urllib.parse import urlparse
from typing import List, Optional, Tuple

# ──────────────────────────────────────────────
# IMPROVED LOGIC: Boost amount per rule trigger
# ──────────────────────────────────────────────

RULE_BOOST_AMOUNT = 0.20   # each triggered rule adds this to phishing probability

# ──────────────────────────────────────────────
# Individual rules
# ──────────────────────────────────────────────

def rule_ip_address(url: str) -> Tuple[bool, Optional[str]]:
    """URLs with a bare IP address as host are highly suspicious."""
    try:
        host = urlparse("http://" + url).hostname or ""
        ip_pattern = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
        if ip_pattern.match(host):
            return True, "IP address used as hostname"
    except Exception:
        pass
    return False, None


def rule_too_many_subdomains(url: str, threshold: int = 4) -> Tuple[bool, Optional[str]]:
    """Excessive subdomain depth is a classic phishing tell."""
    try:
        host = urlparse("http://" + url).hostname or ""
        parts = host.split(".")
        if len(parts) > threshold + 1:
            return True, f"Too many subdomains ({len(parts) - 2})"
    except Exception:
        pass
    return False, None


def rule_suspicious_tld(url: str) -> Tuple[bool, Optional[str]]:
    """Some TLDs are disproportionately abused by phishers."""
    HIGH_RISK_TLDS = {".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top", ".loan", ".click"}
    for tld in HIGH_RISK_TLDS:
        if url.endswith(tld) or (tld + "/") in url or (tld + "?") in url:
            return True, f"High-risk TLD detected ({tld})"
    return False, None


def rule_double_slash_redirect(url: str) -> Tuple[bool, Optional[str]]:
    """Presence of '//' after the path start suggests redirect tricks."""
    if "//" in url:
        return True, "Double-slash redirect pattern detected"
    return False, None


def rule_at_symbol(url: str) -> Tuple[bool, Optional[str]]:
    """The '@' symbol causes browsers to ignore everything before it."""
    if "@" in url:
        return True, "'@' symbol detected (host masking)"
    return False, None


def rule_long_url(url: str, threshold: int = 100) -> Tuple[bool, Optional[str]]:
    """Abnormally long URLs are often generated phishing pages."""
    if len(url) > threshold:
        return True, f"Abnormally long URL ({len(url)} chars)"
    return False, None


def rule_multiple_hyphens(url: str, threshold: int = 4) -> Tuple[bool, Optional[str]]:
    """Excessive hyphens in the domain portion suggest brand impersonation."""
    try:
        host = urlparse("http://" + url).hostname or ""
        if host.count("-") >= threshold:
            return True, f"Multiple hyphens in domain ({host.count('-')})"
    except Exception:
        pass
    return False, None


# ──────────────────────────────────────────────
# Rule runner  (IMPROVED LOGIC)
# ──────────────────────────────────────────────

ALL_RULES = [
    rule_ip_address,
    rule_too_many_subdomains,
    rule_suspicious_tld,
    rule_double_slash_redirect,
    rule_at_symbol,
    rule_long_url,
    rule_multiple_hyphens,
]


def run_rules(url: str) -> Tuple[float, List[str]]:
    """
    IMPROVED LOGIC: Run ALL rules (not just the first hit) and accumulate boosts.

    Returns:
        boost       : Total probability boost from all triggered rules (float).
        rule_names  : List of triggered rule names (empty list if none fired).

    # VIVA EXPLANATION:
    # Old version stopped at the first triggered rule — losing information.
    # New version collects ALL triggered rules. If 2 rules fire, the boost is
    # 0.20 + 0.20 = 0.40, reflecting that the URL is doubly suspicious.
    # The app.py caps the final probability at 1.0 to keep it a valid probability.
    """
    boost = 0.0
    triggered_names = []

    for rule_fn in ALL_RULES:
        triggered, name = rule_fn(url)
        if triggered:
            boost += RULE_BOOST_AMOUNT
            triggered_names.append(name)

    return boost, triggered_names
