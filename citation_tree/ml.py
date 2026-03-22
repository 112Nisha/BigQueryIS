"""ML helpers — semantic similarity and improvement explanation generation.

Models are lazy-loaded so the rest of the pipeline works even when
sentence-transformers / transformers are not installed.

The generative explanation uses the Google Gemini API (gemini-1.5-flash,
free tier) when GEMINI_API_KEY is set; otherwise it falls back to a
keyword heuristic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import hashlib
import time as _time
import threading
from numpy import dot
from openai import OpenAI
import re
from numpy.linalg import norm
from citation_tree.cache import Cache, RateLimiter
from citation_tree.config import (
    GROQ_API_KEY,
    LLM_EXPLANATIONS_ENABLED,
    MAX_LLM_CALLS_PER_RUN,
    MAX_SUMMARY_CHUNKS,
    MAX_TEXT_CHARS_FOR_SUMMARY,
    SUMMARY_CHUNK_SIZE,
)

if TYPE_CHECKING:
    from citation_tree.models import Paper

rate_limiter = RateLimiter(1.2)
_cache = Cache(ttl_days=30)

_similarity_model = None
_thread_state = threading.local()


def _state() -> threading.local:
    if not hasattr(_thread_state, "llm_client"):
        _thread_state.llm_client = None
        _thread_state.llm_calls_used = 0
        _thread_state.llm_budget_exhausted = False
    return _thread_state


def llm_explanations_enabled() -> bool:
    return LLM_EXPLANATIONS_ENABLED and bool(GROQ_API_KEY)


def _budget_remaining() -> int:
    st = _state()
    return max(0, MAX_LLM_CALLS_PER_RUN - st.llm_calls_used)


def _pair_key(parent_id: str, child_id: str, is_reference: bool) -> str:
    rel = "ref" if is_reference else "cite"
    return f"ml:exp:{rel}:{parent_id}:{child_id}"


def _summary_key(paper_id: str, text: str) -> str:
    digest = hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()
    return f"ml:sum:{paper_id}:{digest}"


def _get_client() -> OpenAI | None:
    st = _state()
    if not llm_explanations_enabled():
        return None
    if st.llm_client is None:
        st.llm_client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
        )
    return st.llm_client


def _select_text_for_summary(paper: Paper) -> str:
    # Prefer abstract when available; it is much cheaper than full-text chunking.
    base = (paper.abstract or "").strip()
    if len(base) >= 400:
        return base[:MAX_TEXT_CHARS_FOR_SUMMARY]
    return (paper.full_text or paper.abstract or paper.title or "").strip()[:MAX_TEXT_CHARS_FOR_SUMMARY]


def _chunk_text(text: str, chunk_size: int = SUMMARY_CHUNK_SIZE):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


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

def _call_llm(client, prompt, max_output_tokens=600):
    st = _state()

    if st.llm_budget_exhausted or _budget_remaining() <= 0:
        st.llm_budget_exhausted = True
        return ""

    MAX_PROMPT_CHARS = 12000
    if len(prompt) > MAX_PROMPT_CHARS:
        prompt = prompt[:MAX_PROMPT_CHARS]

    for attempt in range(3):
        try:
            rate_limiter.wait()
            st.llm_calls_used += 1

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
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
                st.llm_budget_exhausted = True
                return ""
            if "quota" in low or "insufficient" in low:
                st.llm_budget_exhausted = True
                return ""
            _time.sleep(3 * (attempt + 1))

    return ""

# Lazy-loads sentence-transformers (all-MiniLM-L6-v2)
def _get_similarity_model():
    global _similarity_model
    if _similarity_model is None:
        try:
            from sentence_transformers import SentenceTransformer

            _similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            pass
    return _similarity_model


def is_similarity_available() -> bool:
    return _get_similarity_model() is not None


# Computes the cosine similarity (0-1) between two texts using the sentence-transformers model - returns 0 if the model is unavailable 
# or if either text is empty
def compute_similarity(text_a: str, text_b: str) -> float:
    model = _get_similarity_model()
    if model is None or not text_a or not text_b:
        return 0.0
    embeddings = model.encode([text_a[:512], text_b[:512]])

    return float(dot(embeddings[0], embeddings[1])/ (norm(embeddings[0]) * norm(embeddings[1]) + 1e-9))

# Trims text to the last full sentence, to avoid incomplete Gemini outputs after truncating the output
def trim_to_last_sentence(text: str) -> str:
    
    matches = list(re.finditer(r"[.!?]", text))
    if not matches:
        return text.strip()
    return text[: matches[-1].end()].strip()


# Generates an explanation of how the parent extends/improves on the child, using the Gemini API if available, 
# or returning an empty string if not.
def generate_improvement_explanation(parent: Paper, child: Paper, is_reference: bool) -> str:
    st = _state()
    if not llm_explanations_enabled():
        return ""

    pair_cache_key = _pair_key(parent.id, child.id, is_reference)
    cached_exp = _cache.get(pair_cache_key)
    if cached_exp:
        return cached_exp

    if st.llm_budget_exhausted or _budget_remaining() <= 0:
        return ""

    parent_text = _select_text_for_summary(parent)
    child_text = _select_text_for_summary(child)

    if not parent_text or not child_text:
        return ""

    client = _get_client()
    if client is None:
        return ""

    print(f"Parent/child text ready; LLM calls remaining: {_budget_remaining()}")

    if is_reference:
        explanation = _generate_with_gemini(client, parent, child, parent_text, child_text)
    else:
        explanation = _generate_with_gemini_citations(client, parent, child, parent_text, child_text)

    if explanation:
        _cache.set(pair_cache_key, explanation)
        return explanation

    return ""

# Generates an explanation using the Gemini API, with retries and model fallbacks in case of rate-limiting or quota exhaustion
def _get_or_build_summary(client, paper: Paper, text: str) -> str:
    if paper.summary:
        return paper.summary

    key = _summary_key(paper.id, text)
    cached = _cache.get(key)
    if cached:
        paper.summary = cached
        return cached

    summary = _summarize_paper(client, text)
    if summary:
        paper.summary = summary
        _cache.set(key, summary)
    return summary


def _generate_with_gemini(client, parent: Paper, child: Paper, parent_text: str, child_text: str,) -> str:
    
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


def _generate_with_gemini_citations(client, parent: Paper, child: Paper, parent_text: str, child_text: str,) -> str:
    
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
