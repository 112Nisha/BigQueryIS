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
    for k in ("dc:title", "title", "Title", "pdf:docinfo:title"):
        if meta.get(k) and len(str(meta[k]).strip()) > 5:
            return str(meta[k]).strip()
    for line in (ln.strip() for ln in text.split("\n")[:50] if ln.strip()):
        if re.match(r"^(arXiv:|http|www\.|Page\s*\d|^\d+$)", line, re.I):
            continue
        if 15 < len(line) < 300 and not line.endswith(":"):
            return line
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
