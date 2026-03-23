"""Data classes for papers and citation trees."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import re

# A single academic paper with metadata and fields for tree position and similarity score (relevance to parent)
@dataclass
class Paper:

    id: str
    title: str
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    abstract: Optional[str] = None
    venue: Optional[str] = None
    citations_count: int = 0
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
    is_open_access: Optional[bool] = None
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    categories: List[str] = field(default_factory=list)
    source: str = "unknown"
    depth: int = 0
    parent_id: Optional[str] = None
    relevance_score: float = 0.0
    relation_type: str = "unknown"
    full_text: Optional[str] = None
    improvement: str = ""
    similarity_to_parent: float = 0.0
    summary: str = ""
    
    # Removes latex math symbols from the text
    @staticmethod
    def _clean_latex(text: str) -> str:
        if not text:
            return text
        text = re.sub(r'\$\$[^$]*\$\$', '', text)
        text = re.sub(r'\$[^$]+\$', '', text)
        text = re.sub(r'\\\([^)]*\\\)', '', text)
        text = re.sub(r'\\\[[^\]]*\\\]', '', text)
        text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
        text = re.sub(r'\\[a-zA-Z]+', '', text)
        text = re.sub(r'[{}\\]', '', text)
        text = re.sub(r'[_^]+', '', text)
        text = re.sub(r'\s*-?\d+pt\s*', '', text)
        text = re.sub(r'\bdocument\b', '', text)
        text = re.sub(r'\bminimal\b', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    # Converts the Paper object to a dictionary, this is for JSON serialization
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self._clean_latex(self.title),
            "authors": self.authors,
            "year": self.year,
            "abstract": self._clean_latex(self.abstract or ""),
            "venue": self.venue,
            "citations_count": self.citations_count,
            "arxiv_id": self.arxiv_id,
            "doi": self.doi,
            "is_open_access": self.is_open_access,
            "url": self.url,
            "pdf_url": self.pdf_url,
            "categories": self.categories,
            "source": self.source,
            "depth": self.depth,
            "parent_id": self.parent_id,
            "relation_type": self.relation_type,
            "relevance_score": round(self.relevance_score, 3),
            "improvement": self._clean_latex(self.improvement),
            "similarity_to_parent": round(self.similarity_to_parent, 3),
            "text": self._clean_latex((self.full_text or "")),
            "summary": self._clean_latex(self.summary),
        }


# A rooted tree of papers connected by citation/reference edges.
# The tree has a dictionary mapping paper IDs to Paper objects, and a list of edges
# where each edge is a (source_id, target_id, relation_type) tuple ("reference" or "citation"), converted to dictionaries
@dataclass
class CitationTree:

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
