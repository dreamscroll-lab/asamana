"""Pure string primitives shared across layers. Only stateless string helpers belong here."""

from __future__ import annotations

import re
import unicodedata
from hashlib import sha1

_SLUG_STRIP_RE = re.compile(r"[^a-zA-Z0-9]+")


def slugify(value: str, *, fallback_prefix: str | None = "item") -> str:
    """Turn an arbitrary string into an ascii slug.

    When nothing survives the ascii fold (e.g. CJK input), falls back to
    ``f"{fallback_prefix}-{<sha1 prefix>}"`` so the id stays stable and unique to the value; with
    no ``fallback_prefix`` the fallback is the bare hash, for a caller that adds its own prefix.
    """

    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    slug = _SLUG_STRIP_RE.sub("-", normalized.lower()).strip("-")
    if slug:
        return slug
    digest = sha1(value.encode("utf-8")).hexdigest()[:10]
    return f"{fallback_prefix}-{digest}" if fallback_prefix else digest
