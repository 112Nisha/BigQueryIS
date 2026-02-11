"""PDF text and metadata extraction via Apache Tika."""

from __future__ import annotations

import os
import re
from typing import List

from tika import parser as tika_parser


def extract_pdf(filepath: str) -> dict:
    """Extract title, abstract, references, and arXiv ID from a PDF."""
    try:
        parsed = tika_parser.from_file(filepath)
        text = parsed.get("content", "") or ""
        meta = parsed.get("metadata", {}) or {}
    except Exception as e:
        print(f"  ⚠ PDF extraction error: {e}")
        return {
            "text": "",
            "title": None,
            "abstract": None,
            "references": [],
            "arxiv_id": None,
        }

    return {
        "text": text,
        "title": _extract_title(text, meta),
        "abstract": _extract_abstract(text),
        "references": _extract_references(text),
        "arxiv_id": _extract_arxiv_id(text, filepath),
    }


def _extract_title(text: str, meta: dict) -> str | None:
    # First try metadata sources - these are usually more reliable
    for k in ("dc:title", "title", "Title", "pdf:docinfo:title"):
        if meta.get(k) and len(str(meta[k]).strip()) > 5:
            title = str(meta[k]).strip()
            # Skip generic or unhelpful metadata titles
            if title.lower() not in ("untitled", "document", "microsoft word"):
                return title
    
    # Build list of candidate titles from the first ~60 lines
    candidates = []
    lines = [ln.strip() for ln in text.split("\n")[:60] if ln.strip()]
    
    for i, line in enumerate(lines):
        # Skip lines that are clearly not titles
        if re.match(r"^(arXiv:|http|www\.|Page\s*\d|^\d+$|^\s*\d+\s*$)", line, re.I):
            continue
        # Skip author-like lines (emails, affiliations)
        if re.search(r"@|university|department|institute|\d{4,}", line, re.I):
            continue
        # Skip headers/footers
        if re.match(r"^(abstract|introduction|keywords|submitted|accepted|published)", line, re.I):
            continue
        # Skip section headers like "1 Introduction" or "I. Background"
        if re.match(r"^(\d+|[IVXLC]+)\.?\s+[A-Z]", line):
            continue
        # Skip very short or very long lines
        if not (15 < len(line) < 300):
            continue
        # Skip lines ending with colon (likely section headers)
        if line.endswith(":"):
            continue
        
        # Score candidates - earlier lines and title-cased lines are preferred
        score = 100 - i  # Earlier is better
        # Boost if line is mostly title-cased (capitals at word starts)
        words = line.split()
        title_cased = sum(1 for w in words if w and w[0].isupper())
        if len(words) > 2 and title_cased / len(words) > 0.6:
            score += 20
        # Boost if it's in the first 10 lines
        if i < 10:
            score += 30
        # Penalize lines that look like subtitles or descriptions
        if line.startswith(("A ", "An ", "The ", "We ", "This ", "In ")):
            score -= 10
        
        candidates.append((score, line))
    
    if candidates:
        candidates.sort(key=lambda x: -x[0])
        return candidates[0][1]
    
    return None


def _extract_abstract(text: str) -> str | None:
    for pat in (
        r"(?:ABSTRACT|Abstract)\s*[:\.\-]?\s*(.*?)(?=\n\s*(?:I\.?\s+)?(?:INTRODUCTION|Introduction|1\.\s+Introduction|Keywords|KEYWORDS))",
        r"(?:ABSTRACT|Abstract)\s*[:\.\-]?\s*(.*?)(?=\n\n\n)",
    ):
        m = re.search(pat, text, re.DOTALL | re.I)
        if m:
            a = " ".join(m.group(1).split())
            if len(a) > 100:
                return a[:2000]
    return None


def _extract_references(text: str) -> List[dict]:
    m = re.search(
        r"(?:^|\n)\s*(?:REFERENCES|References|BIBLIOGRAPHY)\s*\n", text, re.I
    )
    if not m:
        return []
    rtext = text[m.end() :]
    refs = re.findall(
        r"\[(\d+)\]\s*([^\[\]]{30,600}?)(?=\[\d+\]|\n\n|$)", rtext
    )
    if not refs:
        refs = re.findall(r"^(\d+)\.\s+(.{30,500})", rtext, re.M)
    out: list[dict] = []
    for num, content in refs[:50]:
        title = None
        q = re.search(r'"([^"]{15,200})"', content)
        if q:
            title = q.group(1)
        else:
            parts = content.split(". ")
            if len(parts) >= 2 and 10 < len(parts[1].strip()) < 200:
                title = parts[1].strip()
        out.append(
            {"number": num, "text": " ".join(content.split()), "title": title}
        )
    return out


def _extract_arxiv_id(text: str, filepath: str) -> str | None:
    m = re.search(r"(\d{4}\.\d{4,5})", os.path.basename(filepath))
    if m:
        return m.group(1)
    m = re.search(r"arXiv:(\d{4}\.\d{4,5})", text)
    return m.group(1) if m else None
