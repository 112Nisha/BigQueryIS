"""Shared configuration — paths, API URLs, tuning parameters."""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDFS_DIR = os.path.join(BASE_DIR, "pdfs")
CACHE_DIR = os.path.join(BASE_DIR, ".cache")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# API endpoints
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"
ARXIV_API = "http://export.arxiv.org/api/query"
OPENALEX_API = "https://api.openalex.org"

# Default input — change this to any file inside pdfs/
INPUT_PDF = os.path.join(PDFS_DIR, "0704.0001.pdf")

# Tree parameters
MAX_DEPTH = 2
MAX_PAPERS = 20
MIN_RELEVANCE = 0.15

# Rate-limiting (seconds between requests)
RATE_LIMIT = 1.2
