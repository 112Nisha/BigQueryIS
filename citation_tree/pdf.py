"""PDF text and metadata extraction via Apache Tika."""

from __future__ import annotations

import os
import re
import time
from typing import List

import requests
import tika
from tika import parser as tika_parser
tika.initVM()


# Downloads a paper's PDF to pdfs_dir if it is not already cached.
# Uses arxiv_id as the filename (e.g. 1706.03762.pdf).
# Returns the local path on success, or None if the paper has no arxiv_id or the download fails.
def download_pdf(paper, pdfs_dir: str, rate_limit: float = 1.2) -> str | None:
    arxiv_id = getattr(paper, "arxiv_id", None)
    pdf_url = getattr(paper, "pdf_url", None)

    if not arxiv_id:
        return None

    local_path = os.path.join(pdfs_dir, f"{arxiv_id}.pdf")

    # Already cached — no download needed
    if os.path.exists(local_path):
        return local_path

    url = pdf_url or f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    try:
        time.sleep(rate_limit)
        headers = {"User-Agent": "Mozilla/5.0 (compatible; citation-tree-bot/1.0)"}
        resp = requests.get(url, headers=headers, timeout=30, stream=True)
        resp.raise_for_status()
        os.makedirs(pdfs_dir, exist_ok=True)
        with open(local_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=8192):
                fh.write(chunk)
        return local_path
    except Exception as exc:
        print(f"  PDF download failed ({arxiv_id}): {exc}")
        return None

# Extracts title, abstract, references, and arXiv ID from a PDF
def extract_pdf(filepath: str) -> dict:
    try:
        parsed = tika_parser.from_file(filepath)
        text = parsed.get("content", "") or ""
        meta = parsed.get("metadata", {}) or {}
    except Exception as e:
        print(f"  PDF extraction error: {e}")
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

# Extracts title from PDF text and metadata
def _extract_title(text: str, meta: dict) -> str | None:
    for k in ("dc:title", "title", "Title", "pdf:docinfo:title"):
        if meta.get(k) and len(str(meta[k]).strip()) > 5:
            title = str(meta[k]).strip()
            if title.lower() not in ("untitled", "document", "microsoft word"):
                return title
    
    candidates = []
    lines = [ln.strip() for ln in text.split("\n")[:60] if ln.strip()]
    
    for i, line in enumerate(lines):
        if re.match(r"^(arXiv:|http|www\.|Page\s*\d|^\d+$|^\s*\d+\s*$)", line, re.I):
            continue

        if re.match(r"^\d+(\.\d+)*\s+", line):
            continue

        if re.search(r"@|university|department|institute|\d{4,}", line, re.I):
            continue
  
        if re.match(r"^(abstract|introduction|keywords|submitted|accepted|published)", line, re.I):
            continue

        if re.match(r"^(\d+|[IVXLC]+)\.?\s+[A-Z]", line):
            continue

        if not (15 < len(line) < 300):
            continue

        if line.endswith(":"):
            continue
        
        # Scoring heuristic to find the most likely title candidate among the lines, based on position, formatting cues, and content
        score = 100 - i 
        words = line.split()
        title_cased = sum(1 for w in words if w and w[0].isupper())
        if len(words) > 2 and title_cased / len(words) > 0.6:
            score += 20
        if i < 10:
            score += 30
        if line.startswith(("A ", "An ", "The ", "We ", "This ", "In ")):
            score -= 10
        
        candidates.append((score, line))
    
    if candidates:
        candidates.sort(key=lambda x: -x[0])
        return candidates[0][1]
    
    return None


# Extracts abstract from PDF text using regex patterns to find the relevant section 
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

# Extracts references from PDF text by finding the relevant section
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

# Extracts arXiv ID from PDF text or filename
def _extract_arxiv_id(text: str, filepath: str) -> str | None:
    m = re.search(r"(\d{4}\.\d{4,5})", os.path.basename(filepath))
    if m:
        return m.group(1)
    m = re.search(r"arXiv:(\d{4}\.\d{4,5})", text)
    return m.group(1) if m else None
