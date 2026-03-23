"""Text helpers for keyword extraction and title matching."""

from __future__ import annotations

import hashlib
import re
from typing import Set

_STOP_WORDS = frozenset(
    "a an the of for and in on to with by from is are was were be been being "
    "have has had do does did will would could should may might must shall can "
    "that this these those it its we our us they their them using based via "
    "approach method methods results paper study".split()
)

# Returns a set of meaningful lowercase words (>=3 chars, no stop words)
def important_words(text: str) -> Set[str]:
    if not text:
        return set()
    return {
        w for w in re.findall(r"\b[a-z]{3,}\b", text.lower()) if w not in _STOP_WORDS
    }

# Short hash of a title's important words - used for deduplication
def title_hash(title: str) -> str:
    return hashlib.md5(
        " ".join(sorted(important_words(title))[:8]).encode()
    ).hexdigest()[:16]

# Fuzzy check whether two paper titles refer to the same work
def titles_match(a: str, b: str) -> bool:
    wa, wb = important_words(a), important_words(b)
    if not wa or not wb:
        return False
    return len(wa & wb) / max(1, min(len(wa), len(wb))) > 0.5
