"""ML helpers — semantic similarity and improvement explanation generation.

Models are lazy-loaded so the rest of the pipeline works even when
sentence-transformers / transformers are not installed.

The generative explanation uses the Google Gemini API (gemini-1.5-flash,
free tier) when GEMINI_API_KEY is set; otherwise it falls back to a
keyword heuristic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import time as _time
from numpy import dot
from openai import OpenAI
_openai_client = None
import re as _r
from numpy.linalg import norm
from citation_tree.cache import RateLimiter
from citation_tree.config import GROQ_API_KEY

if TYPE_CHECKING:
    from citation_tree.models import Paper

rate_limiter = RateLimiter(1.2)

_similarity_model = None

def _chunk_text(text: str, chunk_size: int = 15000):
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

    MAX_PROMPT_CHARS = 12000
    if len(prompt) > MAX_PROMPT_CHARS:
        prompt = prompt[:MAX_PROMPT_CHARS]

    for attempt in range(3):
        try:
            rate_limiter.wait()

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


# Computes the cosine similarity (0-1) between two texts using the sentence-transformers model - returns 0 if the model is unavailable 
# or if either text is empty
def compute_similarity(text_a: str, text_b: str) -> float:
    model = _get_similarity_model()
    if model is None or not text_a or not text_b:
        return 0.0
    embeddings = model.encode([text_a[:512], text_b[:512]])

    return float(dot(embeddings[0], embeddings[1])/ (norm(embeddings[0]) * norm(embeddings[1]) + 1e-9))

# Trims text to the last full sentence, to avoid incomplete Gemini outputs after truncating the output
def _trim_to_last_sentence(text: str) -> str:
    import re
    matches = list(re.finditer(r"[.!?]", text))
    if not matches:
        return text.strip()
    return text[: matches[-1].end()].strip()


# Generates an explanation of how the parent extends/improves on the child, using the Gemini API if available, 
# or returning an empty string if not.
def generate_improvement_explanation(parent: Paper, child: Paper) -> str:
    parent_text = (parent.full_text or parent.abstract or parent.title or "").strip()
    child_text = (child.full_text or child.abstract or child.title or "").strip()

    if not parent_text or not child_text:
        return ""
    else:
        print("Parent and child text for Gemini are ready")


    explanation = _generate_with_gemini(parent, child, parent_text, child_text)
    if explanation:
        return explanation

    return ""

# Generates an explanation using the Gemini API, with retries and model fallbacks in case of rate-limiting or quota exhaustion
def _generate_with_gemini(parent: Paper, child: Paper, parent_text: str, child_text: str,) -> str:
    # client = _get_gemini_model()
    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )
    if client is None:
        return ""
    
    print("===============STARTING A NEW PAIR OF PAPERS=============")
    parent_summary = _summarize_paper(client, parent_text)
    if not parent_summary:
        print("Failed to summarize parent paper.")
        return ""
    print("Parent summary done")
    parent.summary = parent_summary

    child_summary = _summarize_paper(client, child_text)
    if not child_summary:
        print("Failed to summarize child paper.")
        return ""
    print("Child summary done")
    child.summary = child_summary

    prompt = (
        "You are an expert academic reviewer.\n\n"
        "Below are structured summaries of two research papers:\n"
        "- A parent paper (the citing paper)\n"
        "- A child paper (the cited paper)\n\n"
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
        return _trim_to_last_sentence(final_text)    
    return ""
