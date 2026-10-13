"""Make phishing domain names unclickable before they reach a published artefact.

Reports name the registrable domains that hold most of a set's phishing. Most are
legitimate hosting platforms abused by phishers (``web.app``, ``weebly.com``) and
are named as they are: the point is that the platform is abused. Anything else is
assumed to be a dedicated phishing domain and is defanged (``evil[.]shop``), the
convention threat-intelligence reports use so a name cannot be followed by
accident. Unknown means defanged: the allowlist is the exception, not the rule.
"""

from __future__ import annotations

# Legitimate platforms that appear among the largest phishing domains of these sets.
HOSTING_PLATFORMS = frozenset({
    "firebaseapp.com",
    "netlify.app",
    "pages.dev",
    "r2.dev",
    "repl.co",
    "web.app",
    "weebly.com",
    "weeblysite.com",
    "wixsite.com",
    "wixstudio.com",
    "godaddysites.com",
    "000webhostapp.com",
    "framer.app",
})


def defang(domain: str) -> str:
    """``evil.shop`` -> ``evil[.]shop``; platforms and already-defanged names are left alone."""
    if domain in HOSTING_PLATFORMS or "[.]" in domain:
        return domain
    return domain.replace(".", "[.]")
