"""Masking helpers for privacy mode.

Privacy mode hides account identities (emails, org names) from rendered
output — for screen shares and screenshots — while keeping accounts
distinguishable by slot number, alias, and the first character of each
masked part.
"""

from __future__ import annotations

import re

MASK = "•••"

# Deliberately permissive: this redacts anything email-shaped from free
# text (captured subprocess output), so false positives beat leaks.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def mask_email(email: str) -> str:
    """``noahwross@gmail.com`` → ``n•••@g•••`` (first char of each part)."""
    if not email:
        return email
    local, sep, domain = email.partition("@")
    if not sep:
        return f"{email[:1]}{MASK}"
    return f"{local[:1]}{MASK}@{domain[:1]}{MASK}"


def mask_org(name: str) -> str:
    """Org names are masked to their first character; the ``personal`` tag
    (no org) is not identifying and passes through unchanged."""
    if not name or name == "personal":
        return name
    return f"{name[:1]}{MASK}"


def mask_text(text: str) -> str:
    """Redact every email-shaped token in free text (modal/action output)."""
    return _EMAIL_RE.sub(lambda m: mask_email(m.group(0)), text)
