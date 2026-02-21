"""Base API client with shared caching, rate-limiting, and serialization."""

from __future__ import annotations

import requests

from citation_tree.cache import Cache, RateLimiter
from citation_tree.config import RATE_LIMIT
from citation_tree.models import Paper


class BaseClient:
    """Common infrastructure inherited by every API-specific client."""

    def __init__(
        self,
        cache: Cache,
        rate: float = RATE_LIMIT,
        headers: dict | None = None,
    ):
        self.cache = cache
        self.limiter = RateLimiter(rate)
        self.session = requests.Session()
        if headers:
            self.session.headers.update(headers)

    def _request(self, key: str, fetch, label: str, multi: bool = True):
        """Fetch with cache-first semantics.

        *multi=True*  → expects/returns a list of Papers.
        *multi=False* → expects/returns a single Paper or None.
        """
        cached = self.cache.get(key)
        if cached is not None:
            return (
                [self._to_paper(d) for d in cached if d]
                if multi
                else self._to_paper(cached)
            )
        self.limiter.wait()
        try:
            result = fetch()
            if multi:
                papers = result or []
                # Only cache non-empty results to avoid persisting transient failures
                if papers:
                    self.cache.set(key, [self._from_paper(p) for p in papers])
                return papers
            if result:
                self.cache.set(key, self._from_paper(result))
            return result
        except Exception as e:
            print(f"  ⚠ {label}: {e}")
        return [] if multi else None

    @staticmethod
    def _from_paper(p: Paper) -> dict:
        return {
            "id": p.id,
            "title": p.title,
            "authors": p.authors,
            "year": p.year,
            "abstract": p.abstract,
            "venue": p.venue,
            "citations_count": p.citations_count,
            "arxiv_id": p.arxiv_id,
            "doi": p.doi,
            "url": p.url,
            "pdf_url": p.pdf_url,
            "categories": p.categories,
            "source": p.source,
            "relation_type": p.relation_type,
        }

    @staticmethod
    def _to_paper(d: dict) -> Paper:
        p = Paper(
            id=d["id"],
            title=d["title"],
            authors=d.get("authors", []),
            year=d.get("year"),
            abstract=d.get("abstract"),
            venue=d.get("venue"),
            citations_count=d.get("citations_count", 0),
            arxiv_id=d.get("arxiv_id"),
            doi=d.get("doi"),
            url=d.get("url"),
            pdf_url=d.get("pdf_url"),
            categories=d.get("categories", []),
            source=d.get("source", "unknown"),
        )
        p.relation_type = d.get("relation_type", "unknown")
        return p
