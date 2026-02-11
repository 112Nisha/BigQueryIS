"""Semantic Scholar API client."""

from __future__ import annotations

from typing import List

from citation_tree.cache import Cache
from citation_tree.clients.base import BaseClient
from citation_tree.config import SEMANTIC_SCHOLAR_API
from citation_tree.models import Paper


class S2Client(BaseClient):
    _F = (
        "paperId,title,authors,year,abstract,venue,citationCount,"
        "externalIds,fieldsOfStudy,url,openAccessPdf"
    )

    def __init__(self, cache: Cache):
        super().__init__(
            cache, rate=1.0, headers={"User-Agent": "CitationTree/2.0"}
        )

    def search(self, query: str, limit: int = 10) -> List[Paper]:
        def fetch():
            r = self.session.get(
                f"{SEMANTIC_SCHOLAR_API}/paper/search",
                params={"query": query, "limit": limit, "fields": self._F},
                timeout=30,
            )
            return (
                [p for i in r.json().get("data", []) if (p := self._parse(i))]
                if r.status_code == 200
                else []
            )

        return self._request(f"s2:s:{query}:{limit}", fetch, "S2 search")

    def get_by_arxiv(self, arxiv_id: str) -> Paper | None:
        def fetch():
            r = self.session.get(
                f"{SEMANTIC_SCHOLAR_API}/paper/arXiv:{arxiv_id}",
                params={"fields": self._F},
                timeout=30,
            )
            return self._parse(r.json()) if r.status_code == 200 else None

        return self._request(
            f"s2:a:{arxiv_id}", fetch, "S2 arXiv", multi=False
        )

    def _get_related(
        self,
        paper_id: str,
        endpoint: str,
        nested_key: str,
        rel: str,
        limit: int,
    ) -> List[Paper]:
        def fetch():
            r = self.session.get(
                f"{SEMANTIC_SCHOLAR_API}/paper/{paper_id}/{endpoint}",
                params={"limit": limit, "fields": self._F},
                timeout=30,
            )
            ps: list[Paper] = []
            if r.status_code == 200:
                for item in r.json().get("data", []):
                    inner = item.get(nested_key)
                    if inner and (p := self._parse(inner)):
                        p.relation_type = rel
                        ps.append(p)
            return ps

        tag = "refs" if endpoint == "references" else "cites"
        return self._request(
            f"s2:{tag}:{paper_id}:{limit}", fetch, f"S2 {tag}"
        )

    def get_references(self, pid: str, limit: int = 50):
        return self._get_related(
            pid, "references", "citedPaper", "reference", limit
        )

    def get_citations(self, pid: str, limit: int = 50):
        return self._get_related(
            pid, "citations", "citingPaper", "citation", limit
        )

    def _parse(self, d: dict) -> Paper | None:
        if not d or not d.get("paperId") or not d.get("title"):
            return None
        ext = d.get("externalIds") or {}
        authors = [
            a["name"] for a in (d.get("authors") or []) if a.get("name")
        ]
        pdf = None
        oa = d.get("openAccessPdf")
        if oa and isinstance(oa, dict):
            pdf = oa.get("url")
        axid = ext.get("ArXiv")
        if axid and not pdf:
            pdf = f"https://arxiv.org/pdf/{axid}.pdf"
        return Paper(
            id=f"s2:{d['paperId']}",
            title=d["title"],
            authors=authors,
            year=d.get("year"),
            abstract=d.get("abstract"),
            venue=d.get("venue"),
            citations_count=d.get("citationCount", 0) or 0,
            arxiv_id=axid,
            doi=ext.get("DOI"),
            url=d.get("url"),
            pdf_url=pdf,
            categories=d.get("fieldsOfStudy") or [],
            source="semantic_scholar",
        )
