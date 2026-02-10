#!/usr/bin/env python3
"""
Citation Tree Builder v2

A robust tool that builds citation trees by:
1. Extracting text and references from PDFs using Tika
2. Searching multiple APIs (Semantic Scholar, arXiv, OpenAlex)
3. Finding related papers through citations, references, and semantic similarity
4. Downloading available papers and visualizing the tree

Usage:
    python citation_tree_v2.py <pdf_filename> [--depth N] [--max-papers N]
"""

import os
import sys
import re
import json
import time
import hashlib
import argparse
import requests
import urllib.parse
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
from xml.etree import ElementTree as ET

# Tika for PDF extraction
from tika import parser as tika_parser
import tika
tika.initVM()

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDFS_DIR = os.path.join(BASE_DIR, "pdfs")
TREE_PDFS_DIR = os.path.join(BASE_DIR, "tree_pdfs")
CACHE_DIR = os.path.join(BASE_DIR, ".citation_cache_v2")

os.makedirs(TREE_PDFS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# API endpoints
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"
ARXIV_API = "http://export.arxiv.org/api/query"
OPENALEX_API = "https://api.openalex.org"

# Rate limiting (seconds between requests)
RATE_LIMIT = 1.2


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Paper:
    """Represents a research paper."""
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
    source: str = "unknown"  # Which API found this paper
    local_path: Optional[str] = None
    depth: int = 0
    parent_id: Optional[str] = None
    relevance_score: float = 0.0
    relation_type: str = "unknown"  # reference, citation, related


@dataclass  
class CitationTree:
    """The citation tree structure."""
    root: Paper
    papers: Dict[str, Paper] = field(default_factory=dict)
    edges: List[Tuple[str, str, str]] = field(default_factory=list)  # (parent, child, relation_type)
    downloaded: Set[str] = field(default_factory=set)


# ============================================================================
# CACHING UTILITIES
# ============================================================================

class Cache:
    """Simple file-based cache."""
    
    def __init__(self, cache_dir: str = CACHE_DIR, ttl_days: int = 7):
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_days * 24 * 3600
    
    def _get_path(self, key: str) -> str:
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{hash_key}.json")
    
    def get(self, key: str) -> Optional[Any]:
        path = self._get_path(key)
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if time.time() - data.get('_ts', 0) < self.ttl_seconds:
                    return data.get('value')
            except:
                pass
        return None
    
    def set(self, key: str, value: Any):
        path = self._get_path(key)
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump({'_ts': time.time(), 'value': value}, f)
        except:
            pass


# ============================================================================
# API CLIENTS
# ============================================================================

class RateLimiter:
    """Simple rate limiter."""
    
    def __init__(self, min_interval: float = RATE_LIMIT):
        self.min_interval = min_interval
        self.last_request = 0
    
    def wait(self):
        elapsed = time.time() - self.last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request = time.time()


class ArxivClient:
    """Client for arXiv API."""
    
    def __init__(self, cache: Cache):
        self.cache = cache
        self.rate_limiter = RateLimiter(3.0)  # arXiv requires 3s between requests
        self.session = requests.Session()
    
    def search(self, query: str, max_results: int = 10) -> List[Paper]:
        """Search arXiv for papers matching query."""
        cache_key = f"arxiv:search:{query}:{max_results}"
        cached = self.cache.get(cache_key)
        if cached:
            return [self._dict_to_paper(p) for p in cached]
        
        self.rate_limiter.wait()
        papers = []
        
        try:
            # Clean query for arXiv search
            clean_query = re.sub(r'[^\w\s]', ' ', query)
            clean_query = ' '.join(clean_query.split()[:15])  # Limit to 15 words
            
            params = {
                'search_query': f'all:{clean_query}',
                'start': 0,
                'max_results': max_results,
                'sortBy': 'relevance'
            }
            
            response = self.session.get(ARXIV_API, params=params, timeout=30)
            
            if response.status_code == 200:
                papers = self._parse_arxiv_response(response.text)
                self.cache.set(cache_key, [self._paper_to_dict(p) for p in papers])
        except Exception as e:
            print(f"    ⚠ arXiv search error: {e}")
        
        return papers
    
    def get_by_id(self, arxiv_id: str) -> Optional[Paper]:
        """Get paper by arXiv ID."""
        cache_key = f"arxiv:id:{arxiv_id}"
        cached = self.cache.get(cache_key)
        if cached:
            return self._dict_to_paper(cached)
        
        self.rate_limiter.wait()
        
        try:
            # Clean arxiv ID (remove version if present for search)
            clean_id = re.sub(r'v\d+$', '', arxiv_id)
            
            params = {'id_list': clean_id}
            response = self.session.get(ARXIV_API, params=params, timeout=30)
            
            if response.status_code == 200:
                papers = self._parse_arxiv_response(response.text)
                if papers:
                    self.cache.set(cache_key, self._paper_to_dict(papers[0]))
                    return papers[0]
        except Exception as e:
            print(f"    ⚠ arXiv lookup error: {e}")
        
        return None
    
    def _parse_arxiv_response(self, xml_text: str) -> List[Paper]:
        """Parse arXiv API XML response."""
        papers = []
        
        try:
            root = ET.fromstring(xml_text)
            ns = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
            
            for entry in root.findall('atom:entry', ns):
                title_elem = entry.find('atom:title', ns)
                if title_elem is None or not title_elem.text:
                    continue
                
                title = ' '.join(title_elem.text.split())
                
                # Extract arXiv ID from the id URL
                id_elem = entry.find('atom:id', ns)
                arxiv_id = None
                if id_elem is not None and id_elem.text:
                    match = re.search(r'(\d{4}\.\d{4,5})(v\d+)?', id_elem.text)
                    if match:
                        arxiv_id = match.group(1)
                
                if not arxiv_id:
                    continue
                
                # Authors
                authors = []
                for author in entry.findall('atom:author', ns):
                    name_elem = author.find('atom:name', ns)
                    if name_elem is not None and name_elem.text:
                        authors.append(name_elem.text)
                
                # Year from published date
                year = None
                published = entry.find('atom:published', ns)
                if published is not None and published.text:
                    year_match = re.search(r'(\d{4})', published.text)
                    if year_match:
                        year = int(year_match.group(1))
                
                # Abstract
                abstract = None
                summary = entry.find('atom:summary', ns)
                if summary is not None and summary.text:
                    abstract = ' '.join(summary.text.split())
                
                # Categories
                categories = []
                for cat in entry.findall('arxiv:primary_category', ns):
                    term = cat.get('term')
                    if term:
                        categories.append(term)
                for cat in entry.findall('atom:category', ns):
                    term = cat.get('term')
                    if term and term not in categories:
                        categories.append(term)
                
                # PDF URL
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                
                paper = Paper(
                    id=f"arxiv:{arxiv_id}",
                    title=title,
                    authors=authors,
                    year=year,
                    abstract=abstract,
                    arxiv_id=arxiv_id,
                    url=f"https://arxiv.org/abs/{arxiv_id}",
                    pdf_url=pdf_url,
                    categories=categories,
                    source="arxiv"
                )
                papers.append(paper)
                
        except Exception as e:
            print(f"    ⚠ arXiv parse error: {e}")
        
        return papers
    
    def _paper_to_dict(self, paper: Paper) -> dict:
        return {
            'id': paper.id, 'title': paper.title, 'authors': paper.authors,
            'year': paper.year, 'abstract': paper.abstract, 'arxiv_id': paper.arxiv_id,
            'url': paper.url, 'pdf_url': paper.pdf_url, 'categories': paper.categories,
            'source': paper.source
        }
    
    def _dict_to_paper(self, d: dict) -> Paper:
        return Paper(
            id=d['id'], title=d['title'], authors=d.get('authors', []),
            year=d.get('year'), abstract=d.get('abstract'), arxiv_id=d.get('arxiv_id'),
            url=d.get('url'), pdf_url=d.get('pdf_url'), categories=d.get('categories', []),
            source=d.get('source', 'arxiv')
        )


class SemanticScholarClient:
    """Client for Semantic Scholar API."""
    
    def __init__(self, cache: Cache):
        self.cache = cache
        self.rate_limiter = RateLimiter(1.0)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "CitationTreeV2/1.0"})
    
    def search(self, query: str, limit: int = 10) -> List[Paper]:
        """Search for papers."""
        cache_key = f"s2:search:{query}:{limit}"
        cached = self.cache.get(cache_key)
        if cached:
            return [self._dict_to_paper(p) for p in cached if p]
        
        self.rate_limiter.wait()
        papers = []
        
        try:
            url = f"{SEMANTIC_SCHOLAR_API}/paper/search"
            params = {
                'query': query,
                'limit': limit,
                'fields': 'paperId,title,authors,year,abstract,venue,citationCount,externalIds,fieldsOfStudy,url,openAccessPdf'
            }
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                for item in data.get('data', []):
                    paper = self._parse_paper(item)
                    if paper:
                        papers.append(paper)
                self.cache.set(cache_key, [self._paper_to_dict(p) for p in papers])
        except Exception as e:
            print(f"    ⚠ S2 search error: {e}")
        
        return papers
    
    def get_by_arxiv_id(self, arxiv_id: str) -> Optional[Paper]:
        """Get paper by arXiv ID."""
        cache_key = f"s2:arxiv:{arxiv_id}"
        cached = self.cache.get(cache_key)
        if cached:
            return self._dict_to_paper(cached)
        
        self.rate_limiter.wait()
        
        try:
            url = f"{SEMANTIC_SCHOLAR_API}/paper/arXiv:{arxiv_id}"
            params = {
                'fields': 'paperId,title,authors,year,abstract,venue,citationCount,referenceCount,externalIds,fieldsOfStudy,url,openAccessPdf'
            }
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                paper = self._parse_paper(response.json())
                if paper:
                    self.cache.set(cache_key, self._paper_to_dict(paper))
                    return paper
        except Exception as e:
            print(f"    ⚠ S2 arXiv lookup error: {e}")
        
        return None
    
    def get_references(self, paper_id: str, limit: int = 50) -> List[Paper]:
        """Get papers referenced by this paper."""
        cache_key = f"s2:refs:{paper_id}:{limit}"
        cached = self.cache.get(cache_key)
        if cached:
            return [self._dict_to_paper(p) for p in cached if p]
        
        self.rate_limiter.wait()
        papers = []
        
        try:
            url = f"{SEMANTIC_SCHOLAR_API}/paper/{paper_id}/references"
            params = {
                'limit': limit,
                'fields': 'paperId,title,authors,year,abstract,venue,citationCount,externalIds,fieldsOfStudy,url,openAccessPdf'
            }
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                for item in data.get('data', []):
                    cited = item.get('citedPaper')
                    if cited:
                        paper = self._parse_paper(cited)
                        if paper:
                            paper.relation_type = 'reference'
                            papers.append(paper)
                self.cache.set(cache_key, [self._paper_to_dict(p) for p in papers])
        except Exception as e:
            print(f"    ⚠ S2 refs error: {e}")
        
        return papers
    
    def get_citations(self, paper_id: str, limit: int = 50) -> List[Paper]:
        """Get papers that cite this paper."""
        cache_key = f"s2:cites:{paper_id}:{limit}"
        cached = self.cache.get(cache_key)
        if cached:
            return [self._dict_to_paper(p) for p in cached if p]
        
        self.rate_limiter.wait()
        papers = []
        
        try:
            url = f"{SEMANTIC_SCHOLAR_API}/paper/{paper_id}/citations"
            params = {
                'limit': limit,
                'fields': 'paperId,title,authors,year,abstract,venue,citationCount,externalIds,fieldsOfStudy,url,openAccessPdf'
            }
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                for item in data.get('data', []):
                    citing = item.get('citingPaper')
                    if citing:
                        paper = self._parse_paper(citing)
                        if paper:
                            paper.relation_type = 'citation'
                            papers.append(paper)
                self.cache.set(cache_key, [self._paper_to_dict(p) for p in papers])
        except Exception as e:
            print(f"    ⚠ S2 cites error: {e}")
        
        return papers
    
    def _parse_paper(self, data: dict) -> Optional[Paper]:
        if not data or not data.get('paperId') or not data.get('title'):
            return None
        
        ext_ids = data.get('externalIds', {}) or {}
        authors = [a.get('name', '') for a in (data.get('authors') or []) if a.get('name')]
        
        pdf_url = None
        oa = data.get('openAccessPdf')
        if oa and isinstance(oa, dict):
            pdf_url = oa.get('url')
        
        arxiv_id = ext_ids.get('ArXiv')
        if arxiv_id and not pdf_url:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        
        return Paper(
            id=f"s2:{data['paperId']}",
            title=data['title'],
            authors=authors,
            year=data.get('year'),
            abstract=data.get('abstract'),
            venue=data.get('venue'),
            citations_count=data.get('citationCount', 0) or 0,
            arxiv_id=arxiv_id,
            doi=ext_ids.get('DOI'),
            url=data.get('url'),
            pdf_url=pdf_url,
            categories=data.get('fieldsOfStudy', []) or [],
            source='semantic_scholar'
        )
    
    def _paper_to_dict(self, paper: Paper) -> dict:
        return {
            'id': paper.id, 'title': paper.title, 'authors': paper.authors,
            'year': paper.year, 'abstract': paper.abstract, 'venue': paper.venue,
            'citations_count': paper.citations_count, 'arxiv_id': paper.arxiv_id,
            'doi': paper.doi, 'url': paper.url, 'pdf_url': paper.pdf_url,
            'categories': paper.categories, 'source': paper.source,
            'relation_type': paper.relation_type
        }
    
    def _dict_to_paper(self, d: dict) -> Paper:
        p = Paper(
            id=d['id'], title=d['title'], authors=d.get('authors', []),
            year=d.get('year'), abstract=d.get('abstract'), venue=d.get('venue'),
            citations_count=d.get('citations_count', 0), arxiv_id=d.get('arxiv_id'),
            doi=d.get('doi'), url=d.get('url'), pdf_url=d.get('pdf_url'),
            categories=d.get('categories', []), source=d.get('source', 'semantic_scholar')
        )
        p.relation_type = d.get('relation_type', 'unknown')
        return p


class OpenAlexClient:
    """Client for OpenAlex API (free, no key required)."""
    
    def __init__(self, cache: Cache):
        self.cache = cache
        self.rate_limiter = RateLimiter(0.2)  # OpenAlex allows 10 req/sec
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "CitationTreeV2/1.0 (mailto:research@example.com)"
        })
    
    def search(self, query: str, limit: int = 10) -> List[Paper]:
        """Search for papers."""
        cache_key = f"oa:search:{query}:{limit}"
        cached = self.cache.get(cache_key)
        if cached:
            return [self._dict_to_paper(p) for p in cached if p]
        
        self.rate_limiter.wait()
        papers = []
        
        try:
            url = f"{OPENALEX_API}/works"
            params = {
                'search': query,
                'per_page': limit,
                'select': 'id,title,authorships,publication_year,abstract_inverted_index,cited_by_count,doi,open_access,primary_location,concepts'
            }
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                for item in data.get('results', []):
                    paper = self._parse_paper(item)
                    if paper:
                        papers.append(paper)
                self.cache.set(cache_key, [self._paper_to_dict(p) for p in papers])
        except Exception as e:
            print(f"    ⚠ OpenAlex search error: {e}")
        
        return papers
    
    def get_references(self, openalex_id: str, limit: int = 50) -> List[Paper]:
        """Get papers referenced by this work."""
        cache_key = f"oa:refs:{openalex_id}:{limit}"
        cached = self.cache.get(cache_key)
        if cached:
            return [self._dict_to_paper(p) for p in cached if p]
        
        self.rate_limiter.wait()
        papers = []
        
        try:
            # Get the work's referenced_works
            url = f"{OPENALEX_API}/works/{openalex_id}"
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                ref_ids = data.get('referenced_works', [])[:limit]
                
                if ref_ids:
                    # Fetch details for referenced works
                    self.rate_limiter.wait()
                    filter_str = '|'.join([r.split('/')[-1] for r in ref_ids[:50]])
                    url = f"{OPENALEX_API}/works"
                    params = {
                        'filter': f'openalex_id:{filter_str}',
                        'per_page': 50,
                        'select': 'id,title,authorships,publication_year,cited_by_count,doi,open_access,primary_location,concepts'
                    }
                    response = self.session.get(url, params=params, timeout=30)
                    
                    if response.status_code == 200:
                        for item in response.json().get('results', []):
                            paper = self._parse_paper(item)
                            if paper:
                                paper.relation_type = 'reference'
                                papers.append(paper)
                
                self.cache.set(cache_key, [self._paper_to_dict(p) for p in papers])
        except Exception as e:
            print(f"    ⚠ OpenAlex refs error: {e}")
        
        return papers
    
    def get_citations(self, openalex_id: str, limit: int = 50) -> List[Paper]:
        """Get papers that cite this work."""
        cache_key = f"oa:cites:{openalex_id}:{limit}"
        cached = self.cache.get(cache_key)
        if cached:
            return [self._dict_to_paper(p) for p in cached if p]
        
        self.rate_limiter.wait()
        papers = []
        
        try:
            url = f"{OPENALEX_API}/works"
            params = {
                'filter': f'cites:{openalex_id}',
                'per_page': limit,
                'sort': 'cited_by_count:desc',
                'select': 'id,title,authorships,publication_year,cited_by_count,doi,open_access,primary_location,concepts'
            }
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                for item in response.json().get('results', []):
                    paper = self._parse_paper(item)
                    if paper:
                        paper.relation_type = 'citation'
                        papers.append(paper)
                self.cache.set(cache_key, [self._paper_to_dict(p) for p in papers])
        except Exception as e:
            print(f"    ⚠ OpenAlex cites error: {e}")
        
        return papers
    
    def _parse_paper(self, data: dict) -> Optional[Paper]:
        if not data or not data.get('title'):
            return None
        
        # Extract OpenAlex ID
        oa_id = data.get('id', '').split('/')[-1] if data.get('id') else None
        if not oa_id:
            return None
        
        # Authors
        authors = []
        for authorship in (data.get('authorships') or []):
            author = authorship.get('author', {})
            if author.get('display_name'):
                authors.append(author['display_name'])
        
        # Abstract (stored as inverted index in OpenAlex)
        abstract = None
        abs_idx = data.get('abstract_inverted_index')
        if abs_idx and isinstance(abs_idx, dict):
            try:
                words = [''] * (max(max(positions) for positions in abs_idx.values()) + 1)
                for word, positions in abs_idx.items():
                    for pos in positions:
                        words[pos] = word
                abstract = ' '.join(words)
            except:
                pass
        
        # DOI
        doi = data.get('doi', '').replace('https://doi.org/', '') if data.get('doi') else None
        
        # PDF URL from open access
        pdf_url = None
        oa = data.get('open_access', {})
        if oa.get('oa_url'):
            pdf_url = oa['oa_url']
        
        # Primary location
        location = data.get('primary_location', {}) or {}
        venue = None
        if location.get('source'):
            venue = location['source'].get('display_name')
        
        # Categories from concepts
        categories = [c.get('display_name') for c in (data.get('concepts') or [])[:5] if c.get('display_name')]
        
        return Paper(
            id=f"oa:{oa_id}",
            title=data['title'],
            authors=authors,
            year=data.get('publication_year'),
            abstract=abstract,
            venue=venue,
            citations_count=data.get('cited_by_count', 0) or 0,
            doi=doi,
            pdf_url=pdf_url,
            categories=categories,
            source='openalex'
        )
    
    def _paper_to_dict(self, paper: Paper) -> dict:
        return {
            'id': paper.id, 'title': paper.title, 'authors': paper.authors,
            'year': paper.year, 'abstract': paper.abstract, 'venue': paper.venue,
            'citations_count': paper.citations_count, 'doi': paper.doi,
            'pdf_url': paper.pdf_url, 'categories': paper.categories,
            'source': paper.source, 'relation_type': paper.relation_type
        }
    
    def _dict_to_paper(self, d: dict) -> Paper:
        p = Paper(
            id=d['id'], title=d['title'], authors=d.get('authors', []),
            year=d.get('year'), abstract=d.get('abstract'), venue=d.get('venue'),
            citations_count=d.get('citations_count', 0), doi=d.get('doi'),
            pdf_url=d.get('pdf_url'), categories=d.get('categories', []),
            source=d.get('source', 'openalex')
        )
        p.relation_type = d.get('relation_type', 'unknown')
        return p


# ============================================================================
# PDF EXTRACTION
# ============================================================================

class PDFExtractor:
    """Extract information from PDF files."""
    
    @staticmethod
    def extract(filepath: str) -> dict:
        """Extract text and metadata from PDF."""
        try:
            parsed = tika_parser.from_file(filepath)
            text = parsed.get('content', '') or ''
            metadata = parsed.get('metadata', {}) or {}
            
            return {
                'text': text,
                'metadata': metadata,
                'title': PDFExtractor._extract_title(text, metadata),
                'abstract': PDFExtractor._extract_abstract(text),
                'references': PDFExtractor._extract_references(text),
                'arxiv_id': PDFExtractor._extract_arxiv_id(text, filepath)
            }
        except Exception as e:
            print(f"  ⚠ PDF extraction error: {e}")
            return {'text': '', 'metadata': {}, 'title': None, 'abstract': None, 'references': [], 'arxiv_id': None}
    
    @staticmethod
    def _extract_title(text: str, metadata: dict) -> Optional[str]:
        # Try metadata
        for key in ['dc:title', 'title', 'Title', 'pdf:docinfo:title']:
            if metadata.get(key):
                title = str(metadata[key]).strip()
                if len(title) > 5:
                    return title
        
        # Extract from text
        lines = [l.strip() for l in text.split('\n')[:50] if l.strip()]
        for line in lines:
            if re.match(r'^(arXiv:|http|www\.|Page\s*\d|^\d+$)', line, re.I):
                continue
            if 15 < len(line) < 300 and not line.endswith(':'):
                return line
        
        return None
    
    @staticmethod
    def _extract_abstract(text: str) -> Optional[str]:
        patterns = [
            r'(?:ABSTRACT|Abstract)\s*[:\.\-]?\s*(.*?)(?=\n\s*(?:I\.?\s+)?(?:INTRODUCTION|Introduction|1\.\s+Introduction|Keywords|KEYWORDS))',
            r'(?:ABSTRACT|Abstract)\s*[:\.\-]?\s*(.*?)(?=\n\n\n)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                abstract = ' '.join(match.group(1).split())
                if len(abstract) > 100:
                    return abstract[:2000]
        
        return None
    
    @staticmethod
    def _extract_references(text: str) -> List[dict]:
        """Extract reference entries from paper."""
        refs = []
        
        # Find references section
        ref_match = re.search(r'(?:^|\n)\s*(?:REFERENCES|References|BIBLIOGRAPHY)\s*\n', text, re.IGNORECASE)
        if not ref_match:
            return refs
        
        ref_text = text[ref_match.end():]
        
        # Try numbered patterns: [1], [2], etc.
        numbered_refs = re.findall(r'\[(\d+)\]\s*([^\[\]]{30,600}?)(?=\[\d+\]|\n\n|$)', ref_text)
        if numbered_refs:
            for num, content in numbered_refs[:50]:
                refs.append({
                    'number': num,
                    'text': ' '.join(content.split()),
                    'title': PDFExtractor._guess_title_from_ref(content)
                })
        else:
            # Try period-numbered: 1. 2. etc.
            dot_refs = re.findall(r'^(\d+)\.\s+(.{30,500})', ref_text, re.MULTILINE)
            for num, content in dot_refs[:50]:
                refs.append({
                    'number': num,
                    'text': ' '.join(content.split()),
                    'title': PDFExtractor._guess_title_from_ref(content)
                })
        
        return refs
    
    @staticmethod
    def _guess_title_from_ref(ref_text: str) -> Optional[str]:
        """Try to extract paper title from reference."""
        # Quoted titles
        quoted = re.search(r'"([^"]{15,200})"', ref_text)
        if quoted:
            return quoted.group(1)
        
        # Title after authors (before journal/year)
        # Authors usually end with period, title follows
        parts = ref_text.split('. ')
        if len(parts) >= 2:
            # Skip first part (authors) and take second
            candidate = parts[1].strip()
            if 10 < len(candidate) < 200:
                return candidate
        
        return None
    
    @staticmethod
    def _extract_arxiv_id(text: str, filepath: str) -> Optional[str]:
        """Extract arXiv ID from text or filename."""
        # From filename
        match = re.search(r'(\d{4}\.\d{4,5})', os.path.basename(filepath))
        if match:
            return match.group(1)
        
        # From text
        match = re.search(r'arXiv:(\d{4}\.\d{4,5})', text)
        if match:
            return match.group(1)
        
        return None


# ============================================================================
# PAPER DOWNLOADER
# ============================================================================

class PaperDownloader:
    """Download papers from various sources."""
    
    def __init__(self, output_dir: str = TREE_PDFS_DIR):
        self.output_dir = output_dir
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "CitationTreeV2/1.0"})
    
    def download(self, paper: Paper) -> Optional[str]:
        """Download paper PDF."""
        filename = self._get_filename(paper)
        output_path = os.path.join(self.output_dir, filename)
        
        if os.path.exists(output_path):
            return output_path
        
        urls_to_try = []
        
        if paper.arxiv_id:
            urls_to_try.append(f"https://arxiv.org/pdf/{paper.arxiv_id}.pdf")
        if paper.pdf_url:
            urls_to_try.append(paper.pdf_url)
        
        for url in urls_to_try:
            try:
                response = self.session.get(url, timeout=60, stream=True)
                if response.status_code == 200:
                    content_type = response.headers.get('content-type', '')
                    if 'pdf' in content_type.lower() or url.endswith('.pdf'):
                        with open(output_path, 'wb') as f:
                            for chunk in response.iter_content(8192):
                                f.write(chunk)
                        return output_path
            except:
                continue
        
        return None
    
    def _get_filename(self, paper: Paper) -> str:
        if paper.arxiv_id:
            return f"{paper.arxiv_id.replace('/', '_')}.pdf"
        safe_title = re.sub(r'[^\w\s-]', '', paper.title[:40])
        safe_title = re.sub(r'\s+', '_', safe_title)
        return f"{paper.id.split(':')[-1][:15]}_{safe_title}.pdf"


# ============================================================================
# CITATION TREE BUILDER
# ============================================================================

class CitationTreeBuilder:
    """Build citation trees from papers."""
    
    def __init__(self, max_depth: int = 2, max_papers: int = 30, 
                 download: bool = True, min_relevance: float = 0.1):
        self.max_depth = max_depth
        self.max_papers = max_papers
        self.download = download
        self.min_relevance = min_relevance
        
        self.cache = Cache()
        self.arxiv = ArxivClient(self.cache)
        self.s2 = SemanticScholarClient(self.cache)
        self.openalex = OpenAlexClient(self.cache)
        self.downloader = PaperDownloader()
        
        self.visited: Set[str] = set()
        self.title_hashes: Set[str] = set()  # To avoid duplicates by title
    
    def build(self, pdf_path: str) -> CitationTree:
        """Build citation tree from a PDF file."""
        print("\n" + "=" * 70)
        print("🌳 CITATION TREE BUILDER v2")
        print("=" * 70)
        
        # Extract info from PDF
        print(f"\n📄 Extracting from: {os.path.basename(pdf_path)}")
        extracted = PDFExtractor.extract(pdf_path)
        
        title = extracted['title'] or "Unknown Paper"
        abstract = extracted['abstract']
        arxiv_id = extracted['arxiv_id']
        references = extracted['references']
        
        print(f"   Title: {title[:70]}...")
        if arxiv_id:
            print(f"   arXiv ID: {arxiv_id}")
        print(f"   References extracted: {len(references)}")
        
        # Find root paper in databases
        print("\n🔍 Searching databases for paper info...")
        root_paper = self._find_paper(title, arxiv_id)
        
        if not root_paper:
            print("   Creating paper from extracted info")
            root_paper = Paper(
                id=f"local:{hashlib.md5(title.encode()).hexdigest()[:12]}",
                title=title,
                abstract=abstract,
                arxiv_id=arxiv_id,
                source='local'
            )
        else:
            print(f"   ✓ Found in {root_paper.source}")
            print(f"   📊 Citations: {root_paper.citations_count}")
        
        root_paper.local_path = pdf_path
        root_paper.depth = 0
        
        # Initialize tree
        tree = CitationTree(root=root_paper)
        tree.papers[root_paper.id] = root_paper
        self.visited.add(root_paper.id)
        self.title_hashes.add(self._title_hash(root_paper.title))
        
        # Build tree
        print(f"\n🔄 Building tree (depth: {self.max_depth}, max: {self.max_papers} papers)...")
        self._expand_node(tree, root_paper, extracted['references'])
        
        # Download papers
        if self.download:
            self._download_papers(tree)
        
        return tree
    
    def _find_paper(self, title: str, arxiv_id: str = None) -> Optional[Paper]:
        """Find paper in databases."""
        # Try Semantic Scholar with arXiv ID
        if arxiv_id:
            paper = self.s2.get_by_arxiv_id(arxiv_id)
            if paper:
                return paper
        
        # Try arXiv directly
        if arxiv_id:
            paper = self.arxiv.get_by_id(arxiv_id)
            if paper:
                return paper
        
        # Search by title
        if title:
            papers = self.s2.search(title, limit=3)
            if papers:
                # Find best match
                for p in papers:
                    if self._titles_match(title, p.title):
                        return p
            
            # Try OpenAlex
            papers = self.openalex.search(title, limit=3)
            if papers:
                for p in papers:
                    if self._titles_match(title, p.title):
                        return p
        
        return None
    
    def _expand_node(self, tree: CitationTree, paper: Paper, local_refs: List[dict] = None):
        """Expand a node by finding related papers."""
        if paper.depth >= self.max_depth:
            return
        
        if len(tree.papers) >= self.max_papers:
            return
        
        indent = "  " * (paper.depth + 1)
        print(f"{indent}📚 [{paper.depth}] {paper.title[:50]}...")
        
        related_papers = []
        
        # Get references and citations from APIs
        s2_id = paper.id.replace('s2:', '') if paper.id.startswith('s2:') else None
        oa_id = paper.id.replace('oa:', '') if paper.id.startswith('oa:') else None
        
        if s2_id:
            refs = self.s2.get_references(s2_id, limit=20)
            cites = self.s2.get_citations(s2_id, limit=20)
            print(f"{indent}   S2: {len(refs)} refs, {len(cites)} cites")
            related_papers.extend(refs)
            related_papers.extend(cites)
        
        if oa_id:
            refs = self.openalex.get_references(oa_id, limit=20)
            cites = self.openalex.get_citations(oa_id, limit=20)
            print(f"{indent}   OA: {len(refs)} refs, {len(cites)} cites")
            related_papers.extend(refs)
            related_papers.extend(cites)
        
        # Search arXiv for related papers by title keywords
        if paper.title:
            keywords = self._extract_keywords(paper.title)
            if keywords:
                arxiv_results = self.arxiv.search(keywords, max_results=10)
                print(f"{indent}   arXiv search: {len(arxiv_results)} results")
                for p in arxiv_results:
                    p.relation_type = 'related'
                related_papers.extend(arxiv_results)
        
        # If still not enough, search by title from local references
        if len(related_papers) < 5 and local_refs:
            print(f"{indent}   Searching {len(local_refs)} extracted refs...")
            for ref in local_refs[:10]:
                if ref.get('title'):
                    found = self._find_paper(ref['title'])
                    if found:
                        found.relation_type = 'reference'
                        related_papers.append(found)
        
        # Score and filter
        scored = self._score_papers(paper, related_papers)
        
        # Remove visited and duplicates
        filtered = []
        for p, score in scored:
            title_hash = self._title_hash(p.title)
            if p.id not in self.visited and title_hash not in self.title_hashes and score >= self.min_relevance:
                filtered.append((p, score))
                self.visited.add(p.id)
                self.title_hashes.add(title_hash)
        
        # Sort by relevance and take top N
        filtered.sort(key=lambda x: -x[1])
        to_add = filtered[:min(8, self.max_papers - len(tree.papers))]
        
        print(f"{indent}   ✓ Adding {len(to_add)} papers")
        
        # Add to tree
        for child, score in to_add:
            child.depth = paper.depth + 1
            child.parent_id = paper.id
            child.relevance_score = score
            tree.papers[child.id] = child
            tree.edges.append((paper.id, child.id, child.relation_type))
        
        # Recursively expand children
        for child, _ in to_add:
            if len(tree.papers) < self.max_papers:
                self._expand_node(tree, child)
    
    def _score_papers(self, source: Paper, candidates: List[Paper]) -> List[Tuple[Paper, float]]:
        """Score candidates by relevance to source."""
        results = []
        
        source_words = self._get_important_words(source.title + ' ' + (source.abstract or ''))
        source_cats = set(source.categories)
        
        for paper in candidates:
            if not paper or not paper.title:
                continue
            
            score = 0.0
            
            # Title/abstract word overlap
            paper_words = self._get_important_words(paper.title + ' ' + (paper.abstract or ''))
            if paper_words and source_words:
                overlap = len(source_words & paper_words) / max(1, len(source_words | paper_words))
                score += overlap * 0.4
            
            # Category overlap
            if paper.categories and source_cats:
                cat_overlap = len(source_cats & set(paper.categories)) / max(1, len(source_cats))
                score += cat_overlap * 0.3
            
            # Citation bonus (normalized)
            if paper.citations_count:
                score += min(paper.citations_count / 500, 0.2)
            
            # Relation type bonus
            if paper.relation_type in ('reference', 'citation'):
                score += 0.1
            
            results.append((paper, score))
        
        return results
    
    def _get_important_words(self, text: str) -> Set[str]:
        """Extract important words from text."""
        if not text:
            return set()
        
        stop_words = {
            'a', 'an', 'the', 'of', 'for', 'and', 'in', 'on', 'to', 'with', 'by',
            'from', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
            'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
            'might', 'must', 'shall', 'can', 'that', 'this', 'these', 'those', 'it',
            'its', 'we', 'our', 'us', 'they', 'their', 'them', 'using', 'based',
            'via', 'approach', 'method', 'methods', 'results', 'paper', 'study'
        }
        
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        return set(w for w in words if w not in stop_words)
    
    def _extract_keywords(self, title: str) -> str:
        """Extract search keywords from title."""
        words = self._get_important_words(title)
        # Take most distinctive words
        return ' '.join(list(words)[:6])
    
    def _titles_match(self, t1: str, t2: str) -> bool:
        """Check if two titles are similar enough."""
        w1 = self._get_important_words(t1)
        w2 = self._get_important_words(t2)
        if not w1 or not w2:
            return False
        overlap = len(w1 & w2) / max(1, min(len(w1), len(w2)))
        return overlap > 0.5
    
    def _title_hash(self, title: str) -> str:
        """Create hash from title for deduplication."""
        words = sorted(self._get_important_words(title))
        return hashlib.md5(' '.join(words[:8]).encode()).hexdigest()[:16]
    
    def _download_papers(self, tree: CitationTree):
        """Download papers in the tree."""
        print("\n📥 Downloading papers...")
        
        to_download = [p for p in tree.papers.values() 
                       if not p.local_path and (p.pdf_url or p.arxiv_id)]
        
        downloaded = 0
        for paper in to_download:
            path = self.downloader.download(paper)
            if path:
                paper.local_path = path
                tree.downloaded.add(paper.id)
                downloaded += 1
            time.sleep(0.3)
        
        print(f"   ✓ Downloaded {downloaded}/{len(to_download)} papers")


# ============================================================================
# VISUALIZATION
# ============================================================================

class TreeVisualizer:
    """Visualize citation trees."""
    
    @staticmethod
    def print_tree(tree: CitationTree):
        """Print tree as ASCII art."""
        print("\n" + "=" * 70)
        print("📊 CITATION TREE")
        print("=" * 70 + "\n")
        
        # Build children map
        children = defaultdict(list)
        for parent_id, child_id, rel_type in tree.edges:
            children[parent_id].append((child_id, rel_type))
        
        # Sort by year
        for pid in children:
            children[pid].sort(key=lambda x: tree.papers[x[0]].year or 9999)
        
        def print_node(paper_id: str, prefix: str = "", is_last: bool = True, rel: str = ""):
            paper = tree.papers.get(paper_id)
            if not paper:
                return
            
            connector = "└── " if is_last else "├── "
            year = f"[{paper.year}]" if paper.year else "[????]"
            icon = "📄" if paper.local_path else "🔗"
            cites = f"({paper.citations_count})" if paper.citations_count else ""
            rel_icon = {"reference": "←", "citation": "→", "related": "~"}.get(rel, "")
            
            title = paper.title[:55] + "..." if len(paper.title) > 55 else paper.title
            
            print(f"{prefix}{connector}{icon} {rel_icon}{year} {title} {cites}")
            
            if paper.authors:
                auth_prefix = prefix + ("    " if is_last else "│   ")
                authors = ", ".join(paper.authors[:3])
                if len(paper.authors) > 3:
                    authors += f" +{len(paper.authors)-3}"
                print(f"{auth_prefix}   by {authors}")
            
            child_list = children.get(paper_id, [])
            for i, (cid, crel) in enumerate(child_list):
                new_prefix = prefix + ("    " if is_last else "│   ")
                print_node(cid, new_prefix, i == len(child_list) - 1, crel)
        
        # Print root
        root = tree.root
        year = f"[{root.year}]" if root.year else "[????]"
        print(f"🌱 ROOT: {year} {root.title}")
        if root.authors:
            print(f"   by {', '.join(root.authors[:5])}")
        print()
        
        # Print children
        child_list = children.get(root.id, [])
        if child_list:
            print("📜 Related papers:")
            print("-" * 60)
            for i, (cid, crel) in enumerate(child_list):
                print_node(cid, "", i == len(child_list) - 1, crel)
        else:
            print("   (No related papers found)")
        
        print()
    
    @staticmethod
    def print_stats(tree: CitationTree):
        """Print statistics."""
        print("=" * 70)
        print("📈 STATISTICS")
        print("=" * 70)
        
        total = len(tree.papers)
        downloaded = len(tree.downloaded)
        years = [p.year for p in tree.papers.values() if p.year]
        
        print(f"\n📊 Total papers: {total}")
        print(f"📥 Downloaded: {downloaded}")
        
        if years:
            print(f"📅 Year range: {min(years)} - {max(years)}")
        
        # By depth
        depth_counts = defaultdict(int)
        for p in tree.papers.values():
            depth_counts[p.depth] += 1
        
        print("\n📚 By depth:")
        for d in sorted(depth_counts.keys()):
            label = "Root" if d == 0 else f"Level {d}"
            print(f"   {label}: {depth_counts[d]}")
        
        # By source
        source_counts = defaultdict(int)
        for p in tree.papers.values():
            source_counts[p.source] += 1
        
        print("\n🔌 By source:")
        for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
            print(f"   {src}: {count}")
        
        # Top cited
        top_cited = sorted([p for p in tree.papers.values() if p.citations_count], 
                          key=lambda x: -x.citations_count)[:5]
        if top_cited:
            print("\n⭐ Most cited:")
            for p in top_cited:
                print(f"   [{p.year or '?'}] {p.title[:45]}... ({p.citations_count})")
        
        print()
    
    @staticmethod
    def export_json(tree: CitationTree, path: str):
        """Export tree to JSON."""
        data = {
            "root": {"id": tree.root.id, "title": tree.root.title},
            "papers": [
                {
                    "id": p.id, "title": p.title, "authors": p.authors,
                    "year": p.year, "abstract": p.abstract[:500] if p.abstract else None,
                    "citations_count": p.citations_count, "arxiv_id": p.arxiv_id,
                    "doi": p.doi, "url": p.url, "pdf_url": p.pdf_url,
                    "categories": p.categories, "source": p.source,
                    "local_path": p.local_path, "depth": p.depth,
                    "parent_id": p.parent_id, "relation_type": p.relation_type,
                    "relevance_score": p.relevance_score
                }
                for p in tree.papers.values()
            ],
            "edges": tree.edges
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Exported to: {path}")


# ============================================================================
# MAIN
# ============================================================================

class TempArgs:
    depth = 5             
    max_papers = 20    
    no_download = False  
    no_details = False    
    min_relevance = 0.50     
    direction = "both"
    export = False 

import json
from collections import defaultdict

def print_citation_tree(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    papers = data["papers"]
    root_id = data["root"]["id"]

    # Build lookup tables
    id_to_paper = {p["id"]: p for p in papers}
    children = defaultdict(list)

    for paper in papers:
        parent_id = paper.get("parent_id")
        if parent_id:
            children[parent_id].append(paper["id"])

    def dfs(paper_id, indent=0):
        paper = id_to_paper[paper_id]
        prefix = "│   " * indent + ("├── " if indent > 0 else "")
        print(f"{prefix}{paper['title']} ({paper['year']})")

        for i, child_id in enumerate(children.get(paper_id, [])):
            dfs(child_id, indent + 1)

    print("\n📚 Citation Tree\n")
    dfs(root_id)


def main():
    parser = argparse.ArgumentParser(
        description="Build citation tree from research paper PDF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python citation_tree_v2.py 0704.0001.pdf
  python citation_tree_v2.py paper.pdf --depth 3 --max-papers 50
  python citation_tree_v2.py paper.pdf --no-download
        """
    )
    
    parser.add_argument("pdf", help="PDF file (name in pdfs/ or full path)")
    parser.add_argument("--depth", type=int, default=2, help="Max tree depth (default: 2)")
    parser.add_argument("--max-papers", type=int, default=30, help="Max papers (default: 30)")
    parser.add_argument("--no-download", action="store_true", help="Skip downloading PDFs")
    parser.add_argument("--export", type=str, help="Export JSON path")
    parser.add_argument("--min-relevance", type=float, default=0.1, help="Min relevance (default: 0.1)")
    
    pdf_path = "/home/dell/BigQueryIS/pdfs/0704.0005.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)
    
    # Build tree
    args = TempArgs()
    builder = CitationTreeBuilder(
        max_depth=args.depth,
        max_papers=args.max_papers,
        download=not args.no_download,
        min_relevance=args.min_relevance
    )
    
    tree = builder.build(pdf_path)
    
    # Visualize
    TreeVisualizer.print_tree(tree)
    TreeVisualizer.print_stats(tree)
    
    # Export
    export_path = args.export or os.path.join(BASE_DIR, "citation_tree_output.json")
    TreeVisualizer.export_json(tree, export_path)
    
    print("\n✅ Done!")
    print(f"📁 Downloaded papers: {TREE_PDFS_DIR}")
    print_citation_tree("citation_tree_output.json")


if __name__ == "__main__":
    main()
