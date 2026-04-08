"""PDF text and metadata extraction via Apache Tika."""

from __future__ import annotations

import os
import re
import time
from threading import Lock
from typing import List
import os
import re
import fitz  
import io
from PIL import Image
import pytesseract


import requests
import tika
from citation_tree.cache import GlobalRequestGate
from citation_tree.config import GLOBAL_ARXIV_MIN_INTERVAL
from citation_tree.ml import trim_to_last_sentence
from tika import parser as tika_parser

from citation_tree.text_utils import titles_match
tika.initVM()

# types of arxiv id formats
_MODERN = r"(?P<id>\d{4}\.\d{4,5}(?:v\d+)?)"
_LEGACY = r"(?P<id>[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)"

# Tika server startup/parsing can race across threads; serialize parser calls.
_TIKA_LOCK = Lock()

# Used for paper ids - checks if the id is of the form of an arxiv id
_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?", re.I)

# Cleans the arxiv id by extracting the base and version and normalizing it to a standard format 
def normalize_arxiv_id(arxiv_id: str | None) -> str | None:
    if not arxiv_id:
        return None
    m = _ARXIV_ID_RE.search(arxiv_id)
    if not m:
        return None
    base = m.group(1)
    version = m.group(2) or ""
    return f"{base}{version}"


# gets the latest version of an arxiv id by querying the arxiv API with the base and getting the latest version 
# def resolve_latest_arxiv_id(arxiv_id: str | None) -> str | None:
#     normalized = normalize_arxiv_id(arxiv_id)
#     if not normalized:
#         return None

#     base = normalized.split("v", 1)[0]
#     try:
#         resp = GlobalRequestGate.request(
#             requests,
#             "GET",
#             "http://export.arxiv.org/api/query",
#             group="arxiv",
#             min_interval=GLOBAL_ARXIV_MIN_INTERVAL,
#             params={"id_list": base},
#             timeout=20,
#         )
#         if resp.status_code == 200:
#             # arXiv entry ids look like: http://arxiv.org/abs/1706.03762v7
#             m = re.search(r"<id>\s*https?://arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)\s*</id>", resp.text, re.I)
#             if m:
#                 return m.group(1)
#     except Exception:
#         pass
#     return normalized

# replacing the current version of the pdf with the latest version by downloading the latest version
# def ensure_latest_pdf_path(filepath: str, rate_limit: float = 1.2) -> str:
#     if not os.path.exists(filepath):
#         return filepath

#     arxiv_id = _extract_arxiv_id("", filepath)
#     latest_id = resolve_latest_arxiv_id(arxiv_id)
#     latest_id = arxiv_id
#     if not latest_id:
#         return filepath

#     latest_path = os.path.join(os.path.dirname(filepath), f"{latest_id}.pdf")
#     if os.path.exists(latest_path):
#         return latest_path

#     try:
#         time.sleep(rate_limit)
#         resp = GlobalRequestGate.request(
#             requests,
#             "GET",
#             f"https://arxiv.org/pdf/{latest_id}.pdf",
#             group="pdf",
#             min_interval=0.0,
#             headers={"User-Agent": "Mozilla/5.0 (compatible; citation-tree-bot/1.0)"},
#             timeout=30,
#             stream=True,
#             allow_redirects=True,
#         )
#         resp.raise_for_status()
#         with open(latest_path, "wb") as fh:
#             for chunk in resp.iter_content(chunk_size=8192):
#                 if chunk:
#                     fh.write(chunk)
#         return latest_path
#     except Exception:
#         return filepath


# Extracts title, abstract, references, and arXiv ID from a PDF
def extract_pdf(filepath: str) -> dict:
    try:
        with _TIKA_LOCK:
            parsed = tika_parser.from_file(filepath)
        text = parsed.get("content", "") or ""
        meta = parsed.get("metadata", {}) or {}
        temp_abstract = _extract_abstract(text) or ""
        
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
        "abstract": trim_to_last_sentence(temp_abstract),  
        "references": _extract_references(text),
        "arxiv_id": _extract_arxiv_id(text, filepath),
    }

# Extracts title from PDF text and metadata
def _extract_title(text: str, meta: dict) -> str | None:
    for k in ("dc:title", "title", "Title", "pdf:docinfo:title"):
        # making sure the title isn't of the form arXiv:1009.0288v1, isn't too short or too generic
        if meta.get(k) and len(str(meta[k]).strip()) > 5:
            title = str(meta[k]).strip()
            if (
            len(title) > 5
            and title.lower() not in ("untitled", "document", "microsoft word")
            and not re.match(r"^arxiv:\d{4}\.\d{4,5}", title, re.I)
            and not re.match(r"^\d{4}\.\d{4,5}", title)  
            and "[" not in title  
        ):
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



# searches for an arxiv id in the text
def _search_arxiv(text: str) -> str | None:
    if not text:
        return None

    text = text.replace("\u00a0", " ")

    patterns = [
        rf"arXiv\s*[:：]\s*{_MODERN}",
        rf"arXiv\s*[:：]\s*{_LEGACY}",
        rf"\b{_MODERN}\b",
        rf"\b{_LEGACY}\b",
    ]

    for p in patterns:
        m = re.search(p, text, re.I)
        if m:
            return m.group("id")
    return None

# searches for an arxiv id in the image but does so for different rotated pages since the arxiv id for some pdfs are on the left margin
def _ocr_image_for_arxiv(img: Image.Image) -> str | None:
    for angle in (0, 90, 270, 180):
        test_img = img.rotate(angle, expand=True) if angle else img
        text = pytesseract.image_to_string(test_img, config="--psm 6")
        arxiv_id = _search_arxiv(text)
        if arxiv_id:
            return arxiv_id
    return None


# finding the arxiv id in different ways - checking pdf name, checking text extracted by tika, etc.
def _extract_arxiv_id(text: str, filepath: str, max_pages: int = 3, use_ocr: bool = True) -> str | None:

    filename = os.path.basename(filepath)

    # checks the file name
    for pattern in (_MODERN, _LEGACY):
        m = re.search(pattern, filename, re.I)
        if m:
            return m.group("id")

    # checks the text extracted by tika  
    front_text = text[:12000]
    arxiv_id = _search_arxiv(front_text)
    if arxiv_id:
        return arxiv_id

    # checks the file but extracting the first max_pages, then extracting the blocks of layout regions (like margin areas)
    try:
        doc = fitz.open(filepath)
        collected_text = []

        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            t = page.get_text("text")
            if t:
                collected_text.append(t)
            blocks = page.get_text("blocks")
            for b in blocks:
                if len(b) >= 5 and b[4]:
                    collected_text.append(b[4])

        combined = "\n".join(collected_text)
        arxiv_id = _search_arxiv(combined)
        if arxiv_id:
            return arxiv_id

    except Exception as e:
        print(f"PyMuPDF text extraction failed for {filepath}: {e}")

    # checks the image via ocr
    if not use_ocr:
        return None

    try:
        doc = fitz.open(filepath)

        for i, page in enumerate(doc):
            if i >= 2: 
                break

            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.open(io.BytesIO(pix.tobytes("png")))

            w, h = img.size
            candidates = [img]
            left_crop = img.crop((0, 0, int(w * 0.18), h))
            right_crop = img.crop((int(w * 0.82), 0, w, h))
            candidates = [left_crop, right_crop, img]

            for candidate in candidates:
                arxiv_id = _ocr_image_for_arxiv(candidate)
                if arxiv_id:
                    return arxiv_id

    except Exception as e:
        print(f"OCR arXiv extraction failed for {filepath}: {e}")

    return None