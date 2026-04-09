"""PDF text and metadata extraction via Apache Tika."""

from __future__ import annotations

import os
import re
import time
from threading import Lock
from typing import List

import requests
import tika
from citation_tree.cache import GlobalRequestGate
from citation_tree.config import GLOBAL_ARXIV_MIN_INTERVAL
from citation_tree.ml import extract_abstract_with_llm, trim_to_last_sentence
from tika import parser as tika_parser
tika.initVM()

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
def resolve_latest_arxiv_id(arxiv_id: str | None) -> str | None:
    normalized = normalize_arxiv_id(arxiv_id)
    if not normalized:
        return None

    base = normalized.split("v", 1)[0]
    try:
        resp = GlobalRequestGate.request(
            requests,
            "GET",
            "http://export.arxiv.org/api/query",
            group="arxiv",
            min_interval=GLOBAL_ARXIV_MIN_INTERVAL,
            params={"id_list": base},
            timeout=20,
        )
        if resp.status_code == 200:
            # arXiv entry ids look like: http://arxiv.org/abs/1706.03762v7
            m = re.search(r"<id>\s*https?://arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)\s*</id>", resp.text, re.I)
            if m:
                return m.group(1)
    except Exception:
        pass
    return normalized

# replacing the current version of the pdf with the latest version by downloading the latest version
def ensure_latest_pdf_path(filepath: str, rate_limit: float = 1.2) -> str:
    if not os.path.exists(filepath):
        return filepath

    arxiv_id = _extract_arxiv_id("", filepath)
    latest_id = resolve_latest_arxiv_id(arxiv_id)
    if not latest_id:
        return filepath

    latest_path = os.path.join(os.path.dirname(filepath), f"{latest_id}.pdf")
    if os.path.exists(latest_path):
        return latest_path

    try:
        time.sleep(rate_limit)
        resp = GlobalRequestGate.request(
            requests,
            "GET",
            f"https://arxiv.org/pdf/{latest_id}.pdf",
            group="pdf",
            min_interval=0.0,
            headers={"User-Agent": "Mozilla/5.0 (compatible; citation-tree-bot/1.0)"},
            timeout=30,
            stream=True,
            allow_redirects=True,
        )
        resp.raise_for_status()
        with open(latest_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    fh.write(chunk)
        return latest_path
    except Exception:
        return filepath

# Downloads a paper's PDF to pdfs_dir if it is not already cached.
# Uses arxiv_id as the filename (e.g. 1706.03762.pdf).
# Returns the local path on success, or None if the paper has no arxiv_id or the download fails.
def download_pdf(paper, pdfs_dir: str, rate_limit: float = 1.2) -> str | None:
    latest_arxiv_id = resolve_latest_arxiv_id(getattr(paper, "arxiv_id", None))
    if latest_arxiv_id:
        paper.arxiv_id = latest_arxiv_id

    candidates: list[tuple[str, str | None]] = []
    seen_urls: set[str] = set()

    def _add_candidate(url: str | None, arxiv_hint: str | None = None) -> None:
        if not url:
            return
        norm = url.strip()
        if not norm or norm in seen_urls:
            return
        seen_urls.add(norm)
        candidates.append((norm, arxiv_hint))

    paper_pdf_url = getattr(paper, "pdf_url", None)
    if paper_pdf_url:
        hint = None
        m = re.search(r"arxiv\.org/pdf/(\d{4}\.\d{4,5}(?:v\d+)?)", paper_pdf_url, re.I)
        if m:
            hint = m.group(1)
        _add_candidate(paper_pdf_url, hint)

    if latest_arxiv_id:
        _add_candidate(f"https://arxiv.org/pdf/{latest_arxiv_id}.pdf", latest_arxiv_id)

    # Also try arXiv title search as a fallback in case upstream PDF links are stale/broken.
    if getattr(paper, "title", None):
        try:
            query = paper.title.replace(" ", "+")
            search_url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=3"

            r = GlobalRequestGate.request(
                requests,
                "GET",
                search_url,
                group="arxiv",
                min_interval=GLOBAL_ARXIV_MIN_INTERVAL,
                timeout=20,
            )

            if r.status_code == 200:
                entries = re.findall(r"<id>http://arxiv.org/abs/(.*?)</id>", r.text)
                for arxiv_id in entries:
                    if arxiv_id:
                        _add_candidate(f"https://arxiv.org/pdf/{arxiv_id}.pdf", arxiv_id)

        except Exception as e:
            print(f"  arXiv search failed: {e}")

    if not candidates:
        print(f"  No PDF available for: {paper.title[:60]}")
        return None

    safe_id = (paper.id or "unknown").replace(":", "_")
    last_exc = None

    for url, arxiv_hint in candidates:
        filename = f"{arxiv_hint}.pdf" if arxiv_hint else f"{safe_id}.pdf"
        local_path = os.path.join(pdfs_dir, filename)

        if os.path.exists(local_path):
            if arxiv_hint:
                paper.arxiv_id = arxiv_hint
            return local_path

        try:
            time.sleep(rate_limit)

            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; citation-tree-bot/1.0)"
            }

            resp = GlobalRequestGate.request(
                requests,
                "GET",
                url,
                group="pdf",
                min_interval=0.0,
                headers=headers,
                timeout=30,
                stream=True,
                allow_redirects=True,
            )

            resp.raise_for_status()

            os.makedirs(pdfs_dir, exist_ok=True)

            with open(local_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        fh.write(chunk)

            if arxiv_hint:
                paper.arxiv_id = arxiv_hint
            return local_path

        except Exception as exc:
            last_exc = exc

    if last_exc is not None:
        print(f"  PDF download failed ({paper.id}): {last_exc}")
    return None

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


# Extracts abstract from PDF text using only the LLM on the first chunks.
def _extract_abstract(text: str) -> str | None:
    return extract_abstract_with_llm(text, max_chunks=3) or None

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
    m = re.search(r"(\d{4}\.\d{4,5}(?:v\d+)?)", os.path.basename(filepath), re.I)
    if m:
        return m.group(1)
    m = re.search(r"arXiv:(\d{4}\.\d{4,5}(?:v\d+)?)", text, re.I)
    return m.group(1) if m else None
