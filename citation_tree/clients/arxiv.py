"""arXiv API client."""

from __future__ import annotations

import re
from typing import List
from xml.etree import ElementTree as ET

from citation_tree.cache import Cache
from citation_tree.clients.base import BaseClient
from citation_tree.config import ARXIV_API, GLOBAL_ARXIV_MIN_INTERVAL
from citation_tree.models import Paper


class ArxivClient(BaseClient):
    def __init__(self, cache: Cache):
        super().__init__(cache, rate=GLOBAL_ARXIV_MIN_INTERVAL)
        self.rate_group = "arxiv"

    def search(self, query: str, max_results: int = 10) -> List[Paper]:
        def fetch():
            q = " ".join(re.sub(r"[^\w\s]", " ", query).split()[:15])
            r = self._get(
                ARXIV_API,
                params={
                    "search_query": f"all:{q}",
                    "start": 0,
                    "max_results": max_results,
                    "sortBy": "relevance",
                },
                timeout=30,
            )
            return self._parse(r.text) if r.status_code == 200 else []

        return self._request(
            f"ax:s:{query}:{max_results}", fetch, "arXiv search"
        )

    def get_by_id(self, arxiv_id: str) -> Paper | None:
        def fetch():
            cid = re.sub(r"v\d+$", "", arxiv_id)
            r = self._get(
                ARXIV_API, params={"id_list": cid}, timeout=30
            )
            if r.status_code == 200:
                ps = self._parse(r.text)
                return ps[0] if ps else None
            return None

        return self._request(
            f"ax:i:{arxiv_id}", fetch, "arXiv lookup", multi=False
        )

    def _parse(self, xml: str) -> List[Paper]:
        papers: List[Paper] = []
        try:
            root = ET.fromstring(xml)
            ns = {
                "a": "http://www.w3.org/2005/Atom",
                "x": "http://arxiv.org/schemas/atom",
            }
            for entry in root.findall("a:entry", ns):
                t = entry.find("a:title", ns)
                if t is None or not t.text:
                    continue
                title = " ".join(t.text.split())
                aid = None
                ie = entry.find("a:id", ns)
                if ie is not None and ie.text:
                    m = re.search(r"(\d{4}\.\d{4,5})", ie.text)
                    if m:
                        aid = m.group(1)
                if not aid:
                    continue
                authors = [
                    n.text
                    for a in entry.findall("a:author", ns)
                    if (n := a.find("a:name", ns)) is not None and n.text
                ]
                year = None
                pub = entry.find("a:published", ns)
                if pub is not None and pub.text:
                    ym = re.search(r"(\d{4})", pub.text)
                    if ym:
                        year = int(ym.group(1))
                abstract = None
                s = entry.find("a:summary", ns)
                if s is not None and s.text:
                    abstract = " ".join(s.text.split())
                cats: list[str] = []
                for c in entry.findall("x:primary_category", ns):
                    if ct := c.get("term"):
                        cats.append(ct)
                for c in entry.findall("a:category", ns):
                    if (ct := c.get("term")) and ct not in cats:
                        cats.append(ct)
                papers.append(
                    Paper(
                        id=f"arxiv:{aid}",
                        title=title,
                        authors=authors,
                        year=year,
                        abstract=abstract,
                        arxiv_id=aid,
                        url=f"https://arxiv.org/abs/{aid}",
                        pdf_url=f"https://arxiv.org/pdf/{aid}.pdf",
                        categories=cats,
                        source="arxiv",
                    )
                )
        except Exception as e:
            print(f"  ⚠ arXiv parse: {e}")
        return papers
