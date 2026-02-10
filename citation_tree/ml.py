"""ML helpers — semantic similarity and improvement explanation generation.

Models are lazy-loaded so the rest of the pipeline works even when
sentence-transformers / transformers are not installed.
"""

from __future__ import annotations

from citation_tree.models import Paper
from citation_tree.text_utils import important_words

_similarity_model = None
_generator_pipeline = None


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


def _get_generator():
    """Lazy-load Flan-T5-base for text generation."""
    global _generator_pipeline
    if _generator_pipeline is None:
        try:
            from transformers import pipeline

            _generator_pipeline = pipeline(
                "text-generation",
                model="google/flan-t5-base",
                max_new_tokens=120,
            )
        except ImportError:
            pass
    return _generator_pipeline


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


def generate_improvement_explanation(parent: Paper, child: Paper) -> str:
    """Explain how *child* improves upon *parent*.

    Uses Flan-T5 when available, otherwise falls back to keyword-diff heuristic.
    """
    gen = _get_generator()
    parent_desc = (parent.abstract or parent.title)[:300]
    child_desc = (child.abstract or child.title)[:300]

    if gen is not None:
        prompt = (
            f"Paper A: {parent_desc}\n\n"
            f"Paper B: {child_desc}\n\n"
            "In two concise sentences explain what Paper B changed or "
            "improved compared to Paper A."
        )
        try:
            out = gen(prompt)
            return out[0]["generated_text"].strip()
        except Exception:
            pass

    # Heuristic fallback
    parent_kw = important_words(parent_desc)
    child_kw = important_words(child_desc)
    new_concepts = child_kw - parent_kw
    if new_concepts:
        sample = ", ".join(sorted(new_concepts)[:6])
        return f"Introduces concepts not present in the parent paper: {sample}."
    return "Extends the parent work in a closely related direction."
