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
INPUT_PDF = os.path.join(PDFS_DIR, "1706.03762v7.pdf")

# Tree parameters
MAX_DEPTH = 2
MAX_PAPERS = 40
MIN_RELEVANCE = 0.15
# Root -> top_k children -> each child -> top_k children (when MAX_DEPTH=2)
MAX_CHILDREN_PER_NODE = 4

# Per-node retrieval breadth (higher values improve coverage but cost more API calls)
API_REFERENCE_LIMIT = 50
API_CITATION_LIMIT = 50

# Rate-limiting (seconds between requests)
RATE_LIMIT = 1.2

# LLM controls (Groq/OpenAI-compatible usage in ml.py)
LLM_EXPLANATIONS_ENABLED = True
MAX_LLM_CALLS_PER_RUN = 40
MAX_TEXT_CHARS_FOR_SUMMARY = 10000
SUMMARY_CHUNK_SIZE = 5000
MAX_SUMMARY_CHUNKS = 2

# Debugging
DEBUG_PRINT_ALL_CITERS = True

# Performance
MAX_FETCH_WORKERS = 2
MAX_POSTPROCESS_WORKERS = 4
GLOBAL_HTTP_MAX_CONCURRENCY = 8
GLOBAL_ARXIV_MIN_INTERVAL = 1.2
GLOBAL_S2_MIN_INTERVAL = 0.8
GLOBAL_OA_MIN_INTERVAL = 0.25

# Google Gemini API (free tier — used for improvement explanations in ml.py)
# Get a free key at https://aistudio.google.com/app/apikey

GEMINI_API_KEY = "AIzaSyCNcjSsos4UiJ2wQPkkY80EopIP4nOeM1Y"
OPENAI_API_KEY = "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"
GROQ_API_KEY = "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"