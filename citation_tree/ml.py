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
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            model_name = "google/flan-t5-base"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            _generator_pipeline = (model, tokenizer)
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


def _clean_latex(text: str) -> str:
    """Remove LaTeX math symbols and clean up formatting."""
    import re
    if not text:
        return text
    # Remove display math first: $$...$$ 
    text = re.sub(r'\$\$[^$]*\$\$', '', text)
    # Remove inline math: $...$
    text = re.sub(r'\$[^$]+\$', '', text)
    # Remove \(...\) and \[...\]
    text = re.sub(r'\\\([^)]*\\\)', '', text)
    text = re.sub(r'\\\[[^\]]*\\\]', '', text)
    # Remove LaTeX commands with arguments: \command{...}
    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
    # Remove LaTeX commands with optional args: \command[...]{...}
    text = re.sub(r'\\[a-zA-Z]+\[[^\]]*\]\{[^}]*\}', '', text)
    # Remove standalone LaTeX commands: \command
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    # Remove stray braces, backslashes, and math symbols
    text = re.sub(r'[{}\\]', '', text)
    # Remove subscript/superscript markers
    text = re.sub(r'[_^]+', '', text)
    # Remove common math artifacts
    text = re.sub(r'\s*-?\d+pt\s*', '', text)  # Remove pt measurements like -69pt
    text = re.sub(r'\bdocument\b', '', text)  # Remove stray "document" from LaTeX
    text = re.sub(r'\bminimal\b', '', text)  # Remove stray "minimal"
    # Clean up extra whitespace and punctuation issues
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([,.])', r'\1', text)  # Remove space before punctuation
    text = re.sub(r'([,.])\1+', r'\1', text)  # Remove duplicate punctuation
    return text.strip()


def _is_poor_quality_response(result: str, parent_desc: str, child_desc: str) -> bool:
    """Check if the generated response is poor quality (repetitive, no difference, etc.)."""
    if not result or len(result) < 30:
        return True
    
    result_lower = result.lower()
    
    # Check for "no difference" patterns
    if "no difference" in result_lower or "same as" in result_lower:
        return True
    
    # Check if result is just echoing the abstract (similarity check)
    parent_lower = parent_desc.lower()[:200]
    child_lower = child_desc.lower()[:200]
    
    # If result contains most of the abstract text, it's just echoing
    if parent_lower[:100] in result_lower or child_lower[:100] in result_lower:
        return True
    
    # Check if result starts like an abstract (common patterns)
    abstract_starts = ['we present', 'we study', 'we propose', 'we introduce', 
                       'this paper', 'in this paper', 'a fully', 'the production',
                       'we consider', 'we report', 'we measure', 'a measurement']
    if any(result_lower.startswith(start) for start in abstract_starts):
        return True
    
    # Check for excessive repetition (same phrase repeated)
    sentences = result.split('.')
    if len(sentences) > 2:
        # Check if sentences are too similar
        unique_sentences = set(s.strip().lower() for s in sentences if s.strip())
        if len(unique_sentences) < len([s for s in sentences if s.strip()]) / 2:
            return True
    
    # Check for repeated phrases (more than 2 times)
    words = result_lower.split()
    if len(words) > 20:
        for i in range(len(words) - 5):
            phrase = ' '.join(words[i:i+5])
            if result_lower.count(phrase) > 2:
                return True
    
    return False


def _ensure_complete_sentences(text: str) -> str:
    """Ensure text ends with a complete sentence."""
    if not text:
        return text
    
    text = text.strip()
    
    # If already ends with proper punctuation, return as-is
    if text and text[-1] in '.!?':
        return text
    
    # Find the last complete sentence
    last_period = text.rfind('.')
    last_exclaim = text.rfind('!')
    last_question = text.rfind('?')
    
    last_end = max(last_period, last_exclaim, last_question)
    
    if last_end > len(text) * 0.5:  # Only truncate if we keep at least half
        return text[:last_end + 1]
    
    # If no good sentence ending, add a period
    return text + '.'


def generate_improvement_explanation(parent: Paper, child: Paper) -> str:
    """Explain how *parent* (the citing paper) builds upon *child* (the reference).

    In the citation tree, the parent is the paper being analyzed (newer work),
    and children are the references it cites (older foundational work).
    Uses Flan-T5 when available, otherwise falls back to keyword-diff heuristic.
    """
    gen = _get_generator()
    # Clean LaTeX from abstracts before processing
    parent_desc = _clean_latex((parent.abstract or parent.title)[:400])
    child_desc = _clean_latex((child.abstract or child.title)[:400])

    if gen is not None:
        model, tokenizer = gen
        prompt = (
            f"Compare two research papers. Paper 1 is a newer paper that cites Paper 2.\n\n"
            f"PAPER 1 (newer, citing paper): {parent_desc}\n\n"
            f"PAPER 2 (older, referenced paper): {child_desc}\n\n"
            f"Complete this sentence: 'The citing paper builds on this reference by'"
        )
        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            outputs = model.generate(
                **inputs, 
                max_new_tokens=100,
                do_sample=False,
                repetition_penalty=2.5,
                no_repeat_ngram_size=4,
            )
            result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            result = _clean_latex(result)
            
            # Check if response is good quality
            if not _is_poor_quality_response(result, parent_desc, child_desc):
                # Format nicely and ensure complete sentence
                if not result.lower().startswith('the citing'):
                    result = f"The parent paper builds on this reference by {result.lower()}"
                result = _ensure_complete_sentences(result)
                return result
        except Exception:
            pass

    # Heuristic fallback - generate a structured comparison
    parent_kw = important_words(parent_desc)
    child_kw = important_words(child_desc)
    new_in_parent = parent_kw - child_kw
    common_concepts = parent_kw & child_kw
    unique_to_ref = child_kw - parent_kw
    
    if common_concepts and new_in_parent:
        common_sample = ", ".join(sorted(common_concepts)[:3])
        new_sample = ", ".join(sorted(new_in_parent)[:4])
        return (
            f"This reference provides foundational work on {common_sample}. "
            f"The parent paper builds on it by introducing {new_sample}, "
            f"extending the research with new methods and analysis."
        )
    elif unique_to_ref:
        ref_sample = ", ".join(sorted(unique_to_ref)[:5])
        return (
            f"This reference contributes foundational concepts: {ref_sample}. "
            f"The parent paper builds upon these ideas in its research."
        )
    elif new_in_parent:
        new_sample = ", ".join(sorted(new_in_parent)[:5])
        return (
            f"The parent paper extends this reference by introducing: {new_sample}."
        )
    else:
        return (
            f"This reference provides foundational work that the parent paper "
            f"extends with refined methodology and deeper analysis."
        )
