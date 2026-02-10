"""Data classes for papers and citation trees."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Paper:
    """A single academic paper with metadata and tree-position fields."""

    id: str
    title: str
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    abstract: Optional[str] = None
    venue: Optional[str] = None
    citations_count: int = 0
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    categories: List[str] = field(default_factory=list)
    source: str = "unknown"
    depth: int = 0
    parent_id: Optional[str] = None
    relevance_score: float = 0.0
    relation_type: str = "unknown"
    improvement: str = ""
    similarity_to_parent: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "abstract": (self.abstract or "")[:500],
            "venue": self.venue,
            "citations_count": self.citations_count,
            "arxiv_id": self.arxiv_id,
            "doi": self.doi,
            "url": self.url,
            "pdf_url": self.pdf_url,
            "categories": self.categories,
            "source": self.source,
            "depth": self.depth,
            "parent_id": self.parent_id,
            "relation_type": self.relation_type,
            "relevance_score": round(self.relevance_score, 3),
            "improvement": self.improvement,
            "similarity_to_parent": round(self.similarity_to_parent, 3),
        }


@dataclass
class CitationTree:
    """A rooted tree of papers connected by citation / reference edges."""

    root: Paper
    papers: Dict[str, Paper] = field(default_factory=dict)
    edges: List[Tuple[str, str, str]] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "root_id": self.root.id,
            "papers": {pid: p.to_dict() for pid, p in self.papers.items()},
            "edges": [
                {"source": s, "target": t, "relation": r}
                for s, t, r in self.edges
            ],
        }
