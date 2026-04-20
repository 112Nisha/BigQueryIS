"""Shared configuration — paths, API URLs, tuning parameters."""

import os


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip()


def _env_secret(name: str) -> str:
    # Hosted secret UIs often preserve surrounding quotes from pasted shell snippets.
    value = _env_str(name, "")
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1].strip()
    return value

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
INPUT_PDF = _env_str("INPUT_PDF", os.path.join(PDFS_DIR, "1211.3711v1.pdf"))

# Tree parameters
MAX_DEPTH = _env_int("MAX_DEPTH", 3)
MAX_PAPERS = _env_int("MAX_PAPERS", 20)
MIN_RELEVANCE = _env_float("MIN_RELEVANCE", 0.15)
# Root -> top_k children -> each child -> top_k children (when MAX_DEPTH=2)
MAX_CHILDREN_PER_NODE = _env_int("MAX_CHILDREN_PER_NODE", 3)

# Per-node retrieval breadth (higher values improve coverage but cost more API calls)
API_REFERENCE_LIMIT = _env_int("API_REFERENCE_LIMIT", 50)
API_CITATION_LIMIT = _env_int("API_CITATION_LIMIT", 50)

# Rate-limiting (seconds between requests)
RATE_LIMIT = _env_float("RATE_LIMIT", 1.2)

# LLM controls (Groq/OpenAI-compatible usage in ml.py)
LLM_EXPLANATIONS_ENABLED = _env_bool("LLM_EXPLANATIONS_ENABLED", True)
MAX_LLM_CALLS_PER_RUN = _env_int("MAX_LLM_CALLS_PER_RUN", 1000)
MAX_TEXT_CHARS_FOR_SUMMARY = _env_int("MAX_TEXT_CHARS_FOR_SUMMARY", 15000)
SUMMARY_CHUNK_SIZE = _env_int("SUMMARY_CHUNK_SIZE", 5000)
MAX_SUMMARY_CHUNKS = _env_int("MAX_SUMMARY_CHUNKS", 3)
LLM_PROVIDER = _env_str("LLM_PROVIDER", "auto").lower()
GROQ_MODEL = _env_str("GROQ_MODEL", "llama-3.1-8b-instant")
GEMINI_MODEL = _env_str("GEMINI_MODEL", "gemini-2.0-flash")

# Debugging
DEBUG_PRINT_ALL_CITERS = _env_bool("DEBUG_PRINT_ALL_CITERS", False)

# Deployment/runtime modes
LOW_MEMORY_MODE = _env_bool("LOW_MEMORY_MODE", False)
SIMILARITY_ENABLED = _env_bool("SIMILARITY_ENABLED", not LOW_MEMORY_MODE)
PARALLEL_TREE_BUILDS = _env_bool("PARALLEL_TREE_BUILDS", not LOW_MEMORY_MODE)
DELETE_PDFS_AFTER_USE = _env_bool("DELETE_PDFS_AFTER_USE", True)

# Performance
MAX_FETCH_WORKERS = _env_int("MAX_FETCH_WORKERS", 1 if LOW_MEMORY_MODE else 2)
MAX_POSTPROCESS_WORKERS = _env_int("MAX_POSTPROCESS_WORKERS", 1 if LOW_MEMORY_MODE else 4)
GLOBAL_HTTP_MAX_CONCURRENCY = _env_int("GLOBAL_HTTP_MAX_CONCURRENCY", 8)
GLOBAL_ARXIV_MIN_INTERVAL = _env_float("GLOBAL_ARXIV_MIN_INTERVAL", 1.2)
GLOBAL_S2_MIN_INTERVAL = _env_float("GLOBAL_S2_MIN_INTERVAL", 0.8)
GLOBAL_OA_MIN_INTERVAL = _env_float("GLOBAL_OA_MIN_INTERVAL", 0.25)

# Provider credentials and contact metadata (env-only for local/dev/prod)
GEMINI_API_KEY = _env_secret("GEMINI_API_KEY")
OPENAI_API_KEY = _env_secret("OPENAI_API_KEY")
GROQ_API_KEY = _env_secret("GROQ_API_KEY")
SEMANTIC_SCHOLAR_API_KEY = _env_secret("SEMANTIC_SCHOLAR_API_KEY")
OPENALEX_API_KEY = _env_secret("OPENALEX_API_KEY")
OPENALEX_MAILTO = _env_str("OPENALEX_MAILTO", "")
ARXIV_CONTACT_EMAIL = _env_str("ARXIV_CONTACT_EMAIL", "")