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
INPUT_PDF = os.getenv("INPUT_PDF", os.path.join(PDFS_DIR, "1211.3711v1.pdf"))

# Tree parameters
MAX_DEPTH = _env_int("MAX_DEPTH", 2)
MAX_PAPERS = _env_int("MAX_PAPERS", 60)
MIN_RELEVANCE = _env_float("MIN_RELEVANCE", 0.15)
# Root -> top_k children -> each child -> top_k children (when MAX_DEPTH=2)
MAX_CHILDREN_PER_NODE = _env_int("MAX_CHILDREN_PER_NODE", 2)

# Per-node retrieval breadth (higher values improve coverage but cost more API calls)
API_REFERENCE_LIMIT = _env_int("API_REFERENCE_LIMIT", 50)
API_CITATION_LIMIT = _env_int("API_CITATION_LIMIT", 50)

# Rate-limiting (seconds between requests)
RATE_LIMIT = _env_float("RATE_LIMIT", 1.2)

# LLM controls (Groq/OpenAI-compatible usage in ml.py)
LLM_EXPLANATIONS_ENABLED = _env_bool("LLM_EXPLANATIONS_ENABLED", True)
MAX_LLM_CALLS_PER_RUN = _env_int("MAX_LLM_CALLS_PER_RUN", 1000)
MAX_TEXT_CHARS_FOR_SUMMARY = _env_int("MAX_TEXT_CHARS_FOR_SUMMARY", 10000)
SUMMARY_CHUNK_SIZE = _env_int("SUMMARY_CHUNK_SIZE", 5000)
MAX_SUMMARY_CHUNKS = _env_int("MAX_SUMMARY_CHUNKS", 30)

# Debugging
DEBUG_PRINT_ALL_CITERS = _env_bool("DEBUG_PRINT_ALL_CITERS", False)

# Performance
MAX_FETCH_WORKERS = _env_int("MAX_FETCH_WORKERS", 2)
MAX_POSTPROCESS_WORKERS = _env_int("MAX_POSTPROCESS_WORKERS", 4)
GLOBAL_HTTP_MAX_CONCURRENCY = _env_int("GLOBAL_HTTP_MAX_CONCURRENCY", 8)
GLOBAL_ARXIV_MIN_INTERVAL = _env_float("GLOBAL_ARXIV_MIN_INTERVAL", 1.2)
GLOBAL_S2_MIN_INTERVAL = _env_float("GLOBAL_S2_MIN_INTERVAL", 0.8)
GLOBAL_OA_MIN_INTERVAL = _env_float("GLOBAL_OA_MIN_INTERVAL", 0.25)

# API keys must come from environment variables in local/dev/prod.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")