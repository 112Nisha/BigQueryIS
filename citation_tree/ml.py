"""ML helpers — semantic similarity and improvement explanation generation.

Models are lazy-loaded so the rest of the pipeline works even when
sentence-transformers / transformers are not installed.

The generative explanation uses the Google Gemini API (gemini-1.5-flash,
free tier) when GEMINI_API_KEY is set; otherwise it falls back to a
keyword heuristic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from citation_tree.models import Paper

_similarity_model = None
_gemini_model = None


def _get_similarity_model():
    """Lazy-load sentence-transformers (all-MiniLM-L6-v2)."""
    global _similarity_model
    if _similarity_model is None:
        try:
            from sentence_transformers import SentenceTransformer

            _similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            pass
    return _similarity_model


def _get_gemini_model():
    """Lazy-initialise the Gemini generative model (free tier)."""
    global _gemini_model
    if _gemini_model is None:
        try:
            from citation_tree.config import GEMINI_API_KEY

            if not GEMINI_API_KEY:
                return None
            from google import genai

            _gemini_model = genai.Client(api_key=GEMINI_API_KEY)
        except Exception as exc:
            print(f"    ⚠ Failed to initialise Gemini client: {exc}")
    return _gemini_model


def compute_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity between two texts (0–1).  Returns 0 if model unavailable."""
    model = _get_similarity_model()
    if model is None or not text_a or not text_b:
        return 0.0
    embeddings = model.encode([text_a[:512], text_b[:512]])
    from numpy import dot
    from numpy.linalg import norm

    return float(
        dot(embeddings[0], embeddings[1])
        / (norm(embeddings[0]) * norm(embeddings[1]) + 1e-9)
    )


# ── improvement explanation ──────────────────────────────────────────


def generate_improvement_explanation(parent: Paper, child: Paper) -> str:
    """Compare *parent* (citing paper) with *child* (cited paper) and return
    a short natural-language explanation of how the parent improves on or
    extends the child's work and which idea in the child likely influenced
    the parent.

    Tries the Google Gemini API (gemini-1.5-flash, free tier) first;
    falls back to a keyword-heuristic explanation when the API is
    unavailable.
    """
    parent_text = (parent.full_text or parent.abstract or parent.title or "").strip()
    child_text = (child.full_text or child.abstract or child.title or "").strip()

    if not parent_text or not child_text:
        return ""

    # ── attempt Gemini-based explanation ──────────────────────────────
    explanation = _generate_with_gemini(parent, child, parent_text, child_text)
    if explanation:
        return explanation

    print("    ⚠ Gemini API unavailable — set GEMINI_API_KEY to generate improvement explanations.")
    return ""


def _generate_with_gemini(
    parent: Paper,
    child: Paper,
    parent_text: str,
    child_text: str,
) -> str:
    """Send both papers in full to Gemini (free tier) and ask for an explanation.

    Tries models in order of preference, falling back if one is rate-limited.
    """
    client = _get_gemini_model()
    if client is None:
        return ""

    prompt = (
        "You are an expert academic reviewer. Given two papers — a parent "
        "(the citing paper) and a child (the cited paper) — write an"
        " explanation that covers:\n"
        "1. What specific idea or contribution in the child paper likely "
        "influenced the parent paper.\n"
        "2. How the parent paper improves on, extends, or diverges from "
        "the child paper's work.\n"
        "Be specific and reference concrete methods, findings, or concepts "
        "from each paper. Do not use bullet points.\n\n"
        f"Parent paper (citing, {parent.year or 'unknown year'}):\n"
        f"Title: {parent.title}\n"
        f"Abstract: {parent_text}\n\n"
        f"Child paper (cited, {child.year or 'unknown year'}):\n"
        f"Title: {child.title}\n"
        f"Abstract: {child_text}"
    )

    try:
        import re as _re
        import time as _time

        from google.genai import types

        models_to_try = [
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
        ]

        for model_name in models_to_try:
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.3,
                            max_output_tokens=200,
                        ),
                    )
                    text = response.text.strip()
                    if len(text) > 10:
                        return text
                    break  # empty response, try next model
                except Exception as exc:
                    exc_str = str(exc)
                    if "429" in exc_str or "RESOURCE_EXHAUSTED" in exc_str:
                        if attempt < max_retries - 1:
                            match = _re.search(r"retry in ([\d.]+)s", exc_str, _re.IGNORECASE)
                            wait = float(match.group(1)) + 2 if match else 15
                            wait = min(wait, 60)
                            print(f"    ⏳ {model_name} rate limited — waiting {wait:.0f}s…")
                            _time.sleep(wait)
                            continue
                        # exhausted retries for this model, try next
                        print(f"    ⚠ {model_name} quota exhausted, trying next model…")
                        break
                    print(f"    ⚠ Gemini API error ({model_name}): {exc}")
                    return ""
    except Exception as exc:
        print(f"    ⚠ Gemini API error: {exc}")
    return ""
