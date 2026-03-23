"""OpenAlex API client."""

from __future__ import annotations

from typing import List

from citation_tree.cache import Cache
from citation_tree.clients.base import BaseClient
from citation_tree.config import GLOBAL_OA_MIN_INTERVAL, OPENALEX_API
from citation_tree.models import Paper


class OAClient(BaseClient):
    _SEL = (
        "id,title,authorships,publication_year,abstract_inverted_index,"
        "cited_by_count,doi,open_access,primary_location,concepts"
    )

    def __init__(self, cache: Cache):
        super().__init__(
            cache,
            rate=GLOBAL_OA_MIN_INTERVAL,
            headers={
                "User-Agent": "CitationTree/2.0 (mailto:research@example.com)"
            },
        )
        self.rate_group = "oa"

    # searches openalex for papers matching the query, returns a list of papers, in this code, the query is usually a paper title
    def search(self, query: str, limit: int = 10) -> List[Paper]:
        def fetch():
            r = self._get(
                f"{OPENALEX_API}/works",
                params={
                    "search": query,
                    "per_page": limit,
                    "select": self._SEL,
                },
                timeout=30,
            )
            return (
                [
                    p
                    for i in r.json().get("results", [])
                    if (p := self._parse(i))
                ]
                if r.status_code == 200
                else []
            )

        return self._request(
            f"oa:s:{query}:{limit}", fetch, "OA search"
        )
    
    # gets references of a paper by its openalex id
    def get_references(self, oa_id: str, limit: int = 50) -> List[Paper]:
        def fetch():
            r = self._get(
                f"{OPENALEX_API}/works/{oa_id}", timeout=30
            )
            if r.status_code != 200:
                return []
            refs = r.json().get("referenced_works", [])[:limit]
            if not refs:
                return []
            filt = "|".join(x.split("/")[-1] for x in refs[:50])
            r2 = self._get(
                f"{OPENALEX_API}/works",
                params={
                    "filter": f"openalex_id:{filt}",
                    "per_page": 50,
                    "select": self._SEL,
                },
                timeout=30,
            )
            ps: list[Paper] = []
            if r2.status_code == 200:
                for i in r2.json().get("results", []):
                    if p := self._parse(i):
                        p.relation_type = "reference"
                        ps.append(p)
            return ps

        return self._request(
            f"oa:r:{oa_id}:{limit}", fetch, "OA refs"
        )
    
    # gets citations of a paper by its openalex id
    # this is similar to get_references but with pagination to fetch citations in batches because a paper can have many more citations than references
    def get_citations(self, oa_id: str, limit: int = 50) -> List[Paper]:
        def fetch():
            ps: list[Paper] = []

            fetch_all = limit <= 0
            per_page = 200 if fetch_all else min(200, limit)
            remaining = None if fetch_all else max(0, limit)
            page = 1

            while fetch_all or (remaining and remaining > 0):
                batch = per_page if fetch_all else min(per_page, remaining)
                r = self._get(
                    f"{OPENALEX_API}/works",
                    params={
                        "filter": f"cites:{oa_id}",
                        "per_page": batch,
                        "page": page,
                        "sort": "cited_by_count:desc",
                        "select": self._SEL,
                    },
                    timeout=30,
                )
                if r.status_code != 200:
                    break

                results = r.json().get("results", []) or []
                if not results:
                    break

                for i in results:
                    if p := self._parse(i):
                        p.relation_type = "citation"
                        ps.append(p)

                if len(results) < batch:
                    break

                if not fetch_all and remaining is not None:
                    remaining -= len(results)
                page += 1

            return ps

        return self._request(
            f"oa:c:{oa_id}:{limit}", fetch, "OA cites"
        )
    
    # converts a dictionary response to a list of papers because openalex returns a JSON
    def _parse(self, d: dict) -> Paper | None:
        if not d or not d.get("title"):
            return None
        oid = (d.get("id") or "").split("/")[-1]
        if not oid:
            return None
        authors = [
            a["author"]["display_name"]
            for a in (d.get("authorships") or [])
            if a.get("author", {}).get("display_name")
        ]
        abstract = None
        idx = d.get("abstract_inverted_index")
        if idx and isinstance(idx, dict):
            try:
                words = [""] * (max(max(ps) for ps in idx.values()) + 1)
                for w, positions in idx.items():
                    for pos in positions:
                        words[pos] = w
                abstract = " ".join(words)
            except Exception:
                pass
        doi = (
            d["doi"].replace("https://doi.org/", "") if d.get("doi") else None
        )
        pdf = (d.get("open_access") or {}).get("oa_url")
        is_oa = (d.get("open_access") or {}).get("is_oa")
        loc = d.get("primary_location") or {}
        venue = (
            loc.get("source", {}).get("display_name")
            if loc.get("source")
            else None
        )
        cats = [
            c["display_name"]
            for c in (d.get("concepts") or [])[:5]
            if c.get("display_name")
        ]
        return Paper(
            id=f"oa:{oid}",
            title=d["title"],
            authors=authors,
            year=d.get("publication_year"),
            abstract=abstract,
            venue=venue,
            citations_count=d.get("cited_by_count", 0) or 0,
            doi=doi,
            is_open_access=bool(is_oa) or bool(pdf),
            pdf_url=pdf,
            categories=cats,
            source="openalex",
        )
