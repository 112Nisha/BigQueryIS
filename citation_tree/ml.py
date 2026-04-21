

from __future__ import annotations

from typing import TYPE_CHECKING
import hashlib
import importlib
import time as _time
import threading
from numpy import dot
from openai import OpenAI
import re
from numpy.linalg import norm
from citation_tree.cache import Cache, RateLimiter
from citation_tree.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GROQ_MODEL,
    GROQ_API_KEY,
    LLM_EXPLANATIONS_ENABLED,
    LLM_PROVIDER,
    MAX_LLM_CALLS_PER_RUN,
    MAX_SUMMARY_CHUNKS,
    MAX_TEXT_CHARS_FOR_SUMMARY,
    SIMILARITY_ENABLED,
    SUMMARY_CHUNK_SIZE,
)

if TYPE_CHECKING:
    from citation_tree.models import Paper

rate_limiter = RateLimiter(1.2)
_cache = Cache(ttl_days=30)
_LLM_CALL_LOCK = threading.Lock()

_CACHE_SCHEMA_VERSION = "v2"
_SUMMARY_FALLBACK_MARKER = "Only limited source text was available, so this summary is metadata-based."
_IMPROVEMENT_FALLBACK_MARKER = "Based on available metadata and extracted context"

_similarity_model = None
_thread_state = threading.local()


# returns the state of the current thread
def _state() -> threading.local:
    if not hasattr(_thread_state, "llm_client"):
        _thread_state.llm_client = None
        _thread_state.llm_provider = None
        _thread_state.disabled_providers = set()
        _thread_state.llm_calls_used = 0
        _thread_state.llm_budget_exhausted = False
    return _thread_state

# Returns provider priority based on configured mode and available keys.
def _provider_candidates() -> list[str]:
    mode = (LLM_PROVIDER or "auto").strip().lower()
    available: list[str] = []

    if GROQ_API_KEY:
        available.append("groq")
    if GEMINI_API_KEY:
        available.append("gemini")

    if mode == "groq":
        return ["groq"] if "groq" in available else []
    if mode == "gemini":
        return ["gemini"] if "gemini" in available else []

    return available


# returns true if LLM explanations are enabled and at least one provider key is available
def llm_explanations_enabled() -> bool:
    return LLM_EXPLANATIONS_ENABLED and bool(_provider_candidates())

# Returns the number of LLM calls remaining before hitting the configured budget for a run
def _budget_remaining() -> int:
    st = _state()
    return max(0, MAX_LLM_CALLS_PER_RUN - st.llm_calls_used)

def _is_fallback_summary(text: str) -> bool:
    return _SUMMARY_FALLBACK_MARKER in (text or "")


def _is_fallback_improvement(text: str) -> bool:
    return _IMPROVEMENT_FALLBACK_MARKER in (text or "")

# creates a cache key for a parent-child pair of papers
def _pair_key(parent_id: str, child_id: str, is_reference: bool) -> str:
    rel = "ref" if is_reference else "cite"
    return f"ml:{_CACHE_SCHEMA_VERSION}:exp:{rel}:{parent_id}:{child_id}"

# creates a cache key for summaries of papers
def _summary_key(paper_id: str, text: str) -> str:
    digest = hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()
    return f"ml:{_CACHE_SCHEMA_VERSION}:sum:{paper_id}:{digest}"


def _abstract_key(text: str) -> str:
    digest = hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()
    return f"ml:{_CACHE_SCHEMA_VERSION}:abs:{digest}"

def _model_for_provider(provider: str) -> str:
    return GROQ_MODEL if provider == "groq" else GEMINI_MODEL


def _build_client(provider: str) -> OpenAI | None:
    if provider == "groq" and GROQ_API_KEY:
        return OpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
        )
    if provider == "gemini" and GEMINI_API_KEY:
        return OpenAI(
            api_key=GEMINI_API_KEY,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    return None


def _switch_provider() -> tuple[str | None, OpenAI | None]:
    st = _state()
    if st.llm_provider:
        st.disabled_providers.add(st.llm_provider)

    st.llm_client = None
    st.llm_provider = None

    for provider in _provider_candidates():
        if provider in st.disabled_providers:
            continue

        client = _build_client(provider)
        if client is None:
            continue

        st.llm_provider = provider
        st.llm_client = client
        return provider, client

    return None, None


# returns the active client, if the client is not initialized, it initializes it
def _get_client() -> OpenAI | None:
    st = _state()
    if not llm_explanations_enabled():
        return None

    candidates = _provider_candidates()
    if not candidates:
        return None

    if (
        st.llm_client is not None
        and st.llm_provider in candidates
        and st.llm_provider not in st.disabled_providers
    ):
        return st.llm_client

    _, client = _switch_provider()
    return client

# returns the text used for summaries, prefers full paper text first
def _select_text_for_summary(paper: Paper) -> str:
    full_text = (paper.full_text or "").strip()
    if full_text:
        # Keep enough text to let chunk-based summarization use configured chunk limits.
        return full_text[:MAX_TEXT_CHARS_FOR_SUMMARY]

    abstract = (paper.abstract or "").strip()
    if abstract:
        return abstract[:MAX_TEXT_CHARS_FOR_SUMMARY]

    return (paper.title or "").strip()[:MAX_TEXT_CHARS_FOR_SUMMARY]

# returns a list of chunks of text
def _chunk_text(text: str, chunk_size: int = SUMMARY_CHUNK_SIZE):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# extracts key points from a chunk of text using the LLM
def _extract_key_points(client, text_chunk: str) -> str:
    prompt = (
        "You are an expert academic reviewer.\n\n"
        "From the following paper section, extract ONLY:\n"
        "- Core problem\n"
        "- Proposed method / model\n"
        "- Key technical contributions\n"
        "- Limitations or open problems\n"
        "- Main results or claims\n\n"
        "Extract the following information as short bullet points.\n"
        "Include enough detail to understand the technical contribution.\n"
        "Section:\n"
        f"{text_chunk}\n"
    )
    
    result = _call_llm(client, prompt, max_output_tokens=250)
    return result

# summarizes a paper by chunking the text, extracting key points from each chunk, 
# and then merging those key points into a final summary using the LLM
def _summarize_paper(client, paper_text: str) -> str:
    chunks = _chunk_text(paper_text)
    if len(chunks) > MAX_SUMMARY_CHUNKS:
        chunks = chunks[:MAX_SUMMARY_CHUNKS]

    extracted_parts = []

    for i, chunk in enumerate(chunks):
        extracted = _extract_key_points(client, chunk)
        if extracted:
            extracted_parts.append(extracted)
            print(f"    Extracted key points from chunk {i+1}/{len(chunks)}")
            
    if not extracted_parts:
        return ""

    merged_extraction = "\n".join(extracted_parts)

    summary_prompt = (
        "You are an expert academic reviewer.\n\n"
        "Using ONLY the extracted information below, write a compact "
        "technical summary (150–200 words) covering:\n"
        "- Core problem\n"
        "- Method\n"
        "- Contributions\n"
        "- Limitations\n"
        "- Main results\n\n"
        "Do NOT add new information.\n\n"
        "Extracted information:\n"
        f"{merged_extraction}\n"
    )

    return _call_llm(client, summary_prompt, max_output_tokens=500)

# calls the LLM with the prompt
def _call_llm(client, prompt, max_output_tokens=600):
    st = _state()

    if st.llm_budget_exhausted or _budget_remaining() <= 0:
        st.llm_budget_exhausted = True
        return ""

    MAX_PROMPT_CHARS = 12000
    if len(prompt) > MAX_PROMPT_CHARS:
        prompt = prompt[:MAX_PROMPT_CHARS]

    for attempt in range(3):
        active_client = client or _get_client()
        if active_client is None:
            st.llm_budget_exhausted = True
            return ""

        provider = st.llm_provider or "groq"

        try:
            with _LLM_CALL_LOCK:
                rate_limiter.wait()
                st.llm_calls_used += 1

                response = active_client.chat.completions.create(
                    model=_model_for_provider(provider),
                    messages=[
                        {"role": "system", "content": "You are an expert academic reviewer."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                    max_tokens=max_output_tokens,
                )

            text = (response.choices[0].message.content or "").strip()

            if not text:
                continue

            word_count = len(text.split())

            if word_count < 15:
                continue

            return text

        except Exception as exc:
            msg = " ".join(str(exc).splitlines())[:120]
            low = msg.lower()
            if "rate" in low and "limit" in low:
                client = None
                _switch_provider()
                continue
            if "quota" in low or "insufficient" in low:
                client = None
                _switch_provider()
                continue
            if "unauthorized" in low or "invalid" in low or "api key" in low:
                client = None
                _switch_provider()
                continue
            _time.sleep(3 * (attempt + 1))

    return ""

# Lazy-loads sentence-transformer (all-MiniLM-L6-v2)
def _get_similarity_model():
    global _similarity_model
    if not SIMILARITY_ENABLED:
        return None

    if _similarity_model is False:
        return None

    if _similarity_model is None:
        try:
            st_module = importlib.import_module("sentence_transformers")
            _similarity_model = st_module.SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _similarity_model = False
    return _similarity_model

# Returns true if the similarity model is available, false otherwise
def is_similarity_available() -> bool:
    return _get_similarity_model() is not None


# Computes the cosine similarity (0-1) between two texts using the sentence-transformers model - checks
# how similar two texts are
def compute_similarity(text_a: str, text_b: str) -> float:
    model = _get_similarity_model()
    if model is None or not text_a or not text_b:
        return 0.0
    embeddings = model.encode([text_a[:512], text_b[:512]])

    return float(dot(embeddings[0], embeddings[1])/ (norm(embeddings[0]) * norm(embeddings[1]) + 1e-9))

# Trims text to the last full sentence, to avoid incomplete LLM outputs after truncating the output
def trim_to_last_sentence(text: str) -> str:
    matches = list(re.finditer(r"[.!?]", text))
    if not matches:
        return text.strip()
    return text[: matches[-1].end()].strip()


def _fallback_summary_from_metadata(paper: Paper) -> str:
    title = (paper.title or "This paper").strip()
    source = (paper.source or "unknown source").replace("_", " ")
    year = str(paper.year) if paper.year else "unknown year"
    venue = (paper.venue or "unspecified venue").strip()
    return (
        f"The paper '{title}' is recorded as a {source} entry from {year} "
        f"with venue information '{venue}'. "
        "Only limited source text was available, so this summary is metadata-based."
    )


def _backfill_abstract_from_summary(paper: Paper, summary: str) -> None:
    if (paper.abstract or "").strip() or not summary:
        return

    candidate = " ".join(summary.split())
    candidate = trim_to_last_sentence(candidate[:1200])
    if len(candidate) >= 80:
        paper.abstract = candidate


def _fallback_improvement_explanation(parent: Paper, child: Paper, is_reference: bool) -> str:
    parent_title = (parent.title or "the parent paper").strip()
    child_title = (child.title or "the child paper").strip()

    if is_reference:
        return (
            f"'{parent_title}' appears to build on prior ideas from '{child_title}'. "
            "Based on available metadata and extracted context, the relationship is likely a "
            "methodological or conceptual extension where the parent paper applies, refines, "
            "or broadens techniques introduced in the child paper."
        )

    return (
        f"'{child_title}' cites '{parent_title}' and appears to build on it as prior work. "
        "From the available metadata and extracted context, the child paper likely extends, "
        "adapts, or empirically validates ideas introduced by the parent paper."
    )


def extract_abstract_with_llm(text: str, max_chunks: int = 3) -> str:
    if not llm_explanations_enabled():
        return ""

    normalized = (text or "").strip()
    if len(normalized) < 400:
        return ""

    cache_key = _abstract_key(normalized)
    cached = _cache.get(cache_key)
    if cached:
        return cached

    st = _state()
    if st.llm_budget_exhausted or _budget_remaining() <= 0:
        return ""

    client = _get_client()
    if client is None:
        return ""

    chunk_limit = max(1, min(3, max_chunks))
    chunks = _chunk_text(normalized, chunk_size=3500)[:chunk_limit]
    if not chunks:
        return ""

    joined_chunks = "\n\n".join(
        f"[CHUNK {i + 1}]\n{chunk}" for i, chunk in enumerate(chunks)
    )

    prompt = (
        "You are extracting the abstract from noisy PDF text.\n\n"
        "Given the first chunks of a paper, return ONLY the abstract text.\n"
        "Rules:\n"
        "- If an Abstract section appears, extract that content only.\n"
        "- Stop before Introduction, Keywords, or numbered section headers.\n"
        "- Do not include labels like 'Abstract'.\n"
        "- Do not use markdown, bullets, quotes, or commentary.\n"
        "- If no abstract is identifiable, return an empty string.\n\n"
        "Paper chunks:\n"
        f"{joined_chunks}\n"
    )

    result = _call_llm(client, prompt, max_output_tokens=320)
    if not result:
        return ""

    cleaned = " ".join(result.split())
    cleaned = re.sub(r"^\s*abstract\s*[:\-.\s]+", "", cleaned, flags=re.I)
    low = cleaned.lower()

    if re.search(r"\b(no abstract|not identifiable|cannot determine|not found)\b", low):
        return ""

    if len(cleaned.split()) < 20:
        return ""

    cleaned = trim_to_last_sentence(cleaned)[:2000]
    if len(cleaned) < 100:
        return ""

    _cache.set(cache_key, cleaned)
    return cleaned


# Generates an explanation of the relationship between two papers, using the LLM API if available, 
# or returning an empty string if not.
def generate_improvement_explanation(parent: Paper, child: Paper, is_reference: bool) -> str:
    pair_cache_key = _pair_key(parent.id, child.id, is_reference)
    cached_exp = _cache.get(pair_cache_key)
    if cached_exp and (
        not _is_fallback_improvement(cached_exp)
        or not llm_explanations_enabled()
    ):
        return cached_exp

    explanation = ""
    parent_text = _select_text_for_summary(parent)
    child_text = _select_text_for_summary(child)

    if llm_explanations_enabled() and parent_text and child_text:
        st = _state()
        if not st.llm_budget_exhausted and _budget_remaining() > 0:
            client = _get_client()
            if client is not None:
                print(f"Parent/child text ready; LLM calls remaining: {_budget_remaining()}")

                if is_reference:
                    explanation = _generate_with_llm(client, parent, child, parent_text, child_text)
                else:
                    explanation = _generate_with_llm_citations(client, parent, child, parent_text, child_text)

    if not explanation:
        explanation = _fallback_improvement_explanation(parent, child, is_reference)

    if explanation and not _is_fallback_improvement(explanation):
        _cache.set(pair_cache_key, explanation)

    return explanation

# generates a summary for a paper
def _get_or_build_summary(client, paper: Paper, text: str) -> str:
    if paper.summary:
        _backfill_abstract_from_summary(paper, paper.summary)
        return paper.summary

    key = _summary_key(paper.id, text)
    cached = _cache.get(key)
    if cached and (
        not _is_fallback_summary(cached)
        or not llm_explanations_enabled()
    ):
        paper.summary = cached
        _backfill_abstract_from_summary(paper, cached)
        return cached

    summary = _summarize_paper(client, text)
    if not summary:
        summary = _fallback_summary_from_metadata(paper)

    paper.summary = summary
    if not _is_fallback_summary(summary):
        _cache.set(key, summary)
    _backfill_abstract_from_summary(paper, summary)
    return summary

# generates an explanation of how a parent paper improves upon a child paper using the LLM
def _generate_with_llm(client, parent: Paper, child: Paper, parent_text: str, child_text: str,) -> str:
    
    print("===============STARTING A NEW PAIR OF PAPERS=============")
    parent_summary = _get_or_build_summary(client, parent, parent_text)
    if not parent_summary:
        print("Failed to summarize parent paper.")
        return ""
    print("Parent summary done")

    child_summary = _get_or_build_summary(client, child, child_text)
    if not child_summary:
        print("Failed to summarize child paper.")
        return ""
    print("Child summary done")

    prompt = (
        "You are an expert academic reviewer.\n\n"
        "Below are structured summaries of two research papers:\n"
        "- A parent paper (the newer paper that cites the child paper)\n"
        "- A child paper (the referenced prior work)\n\n"
        "Using ONLY the information provided, explain:\n\n"
        "1. Which specific ideas, methods, or findings from the child paper "
        "influenced the parent paper.\n"
        "2. How the parent paper improves upon, extends, or diverges from the "
        "child paper’s approach (e.g., new methods, stronger assumptions, "
        "broader scope, improved results, or different conclusions).\n\n"
        "Be concrete and technical.\n"
        "Do not speculate beyond the summaries.\n"
        "Do not use bullet points.\n\n"
        "Parent paper summary:\n"
        f"{parent_summary}\n\n"
        "Child paper summary:\n"
        f"{child_summary}\n\n"
        "Write a concise explanation (120–180 words) focused on the most important "
        "connection between the two papers."
    )

    final_text = _call_llm(client, prompt, max_output_tokens=250)

    if final_text:
        return trim_to_last_sentence(final_text)    
    return ""

# generates an explanation of how a child paper builds upon a parent paper using the LLM
def _generate_with_llm_citations(client, parent: Paper, child: Paper, parent_text: str, child_text: str,) -> str:
    
    print("===============STARTING A NEW PAIR OF PAPERS=============")
    parent_summary = _get_or_build_summary(client, parent, parent_text)
    if not parent_summary:
        print("Failed to summarize parent paper.")
        return ""
    print("Parent summary done")

    child_summary = _get_or_build_summary(client, child, child_text)
    if not child_summary:
        print("Failed to summarize child paper.")
        return ""
    print("Child summary done")

    prompt = (
        "You are an expert academic reviewer.\n\n"
        "Below are structured summaries of two research papers:\n"
        "- A parent paper (the cited paper / prior work)\n"
        "- A child paper (the newer citing paper)\n\n"
        "Using ONLY the information provided, explain:\n\n"
        "1. Which specific ideas, methods, or findings from the parent paper "
        "influenced the child paper.\n"
        "2. How the child paper improves upon, extends, or diverges from the "
        "parent paper’s approach (e.g., new methods, stronger assumptions, "
        "broader scope, improved results, or different conclusions).\n\n"
        "Be concrete and technical.\n"
        "Do not speculate beyond the summaries.\n"
        "Do not use bullet points.\n\n"
        "Parent paper summary:\n"
        f"{parent_summary}\n\n"
        "Child paper summary:\n"
        f"{child_summary}\n\n"
        "Write a concise explanation (120–180 words) focused on the most important "
        "connection between the two papers."
    )

    final_text = _call_llm(client, prompt, max_output_tokens=250)

    if final_text:
        return trim_to_last_sentence(final_text)    
    return ""
