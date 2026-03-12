"""ML helpers — semantic similarity and improvement explanation generation.

Models are lazy-loaded so the rest of the pipeline works even when
sentence-transformers / transformers are not installed.

The generative explanation uses the Google Gemini API (gemini-1.5-flash,
free tier) when GEMINI_API_KEY is set; otherwise it falls back to a
keyword heuristic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import re as _re
import time as _time
from google.genai import types
from numpy import dot
from numpy.linalg import norm
from citation_tree.config import GEMINI_API_KEY

if TYPE_CHECKING:
    from citation_tree.models import Paper

_similarity_model = None
_gemini_model = None

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
        "Write in short, information-dense sentences.\n"
        "Do NOT summarize. Do NOT add background.\n\n"
        "Section:\n"
        f"{text_chunk}\n"
    )

    return _call_gemini(client, prompt, max_output_tokens=250)

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

    return _call_gemini(client, summary_prompt, max_output_tokens=500)

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

# Lazy-initialises the Gemini generative model (free tier)
def _get_gemini_model():
    global _gemini_model
    if _gemini_model is None:
        try:
            from citation_tree.config import GEMINI_API_KEY

            if not GEMINI_API_KEY:
                return None
            from google import genai

            _gemini_model = genai.Client(api_key=GEMINI_API_KEY)
        except Exception as exc:
            print(f"    Failed to initialise Gemini client: {exc}")
    return _gemini_model


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

####### CHECK IS THIS IS PARALLELIZING BECAUSE IF U SEND IN PAPERS PARALLELY THEN
####### THE INPUT TOKEN LIMIT WILL BE EXCEEDED
# Generates an explanation of how the parent extends/improves on the child, using the Gemini API if available, 
# or returning an empty string if not.
def generate_improvement_explanation(parent: Paper, child: Paper) -> str:
    parent_text = (parent.full_text or parent.abstract or parent.title or "").strip()
    child_text = (child.full_text or child.abstract or child.title or "").strip()

    # MAX_INPUT_CHARS = 6000
    # parent_text = parent_text[:MAX_INPUT_CHARS]
    # child_text = child_text[:MAX_INPUT_CHARS]

    if not parent_text or not child_text:
        return ""
    else:
        print("Parent and child text for Gemini are ready")


    explanation = _generate_with_gemini(parent, child, parent_text, child_text)
    if explanation:
        return explanation

    # print("    Gemini API unavailable — set GEMINI_API_KEY to generate improvement explanations.")
    return ""

# Trying multiple models with retries in case of rate-limiting or quota exhaustion
def _call_gemini(client, prompt, max_output_tokens=600):
    models_to_try = [
        "gemini-2.5-pro",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-2.5-flash",
    ]

    for model_name in models_to_try:
        for attempt in range(2):
            try:
              
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.3,
                        max_output_tokens=max_output_tokens,
                    ),
                )
                # print("RAW RESPONSE:", response)
                text = (response.text or "").strip()
                if text:
                    return text
            except Exception as exc:
                print("Gemini error:", exc)
                return ""

    return ""

# Generates an explanation using the Gemini API, with retries and model fallbacks in case of rate-limiting or quota exhaustion
def _generate_with_gemini(parent: Paper, child: Paper, parent_text: str, child_text: str,) -> str:
    client = _get_gemini_model()
    if client is None:
        return ""
    
    # prompt_for_parent_summary = (
    #     "You are an expert academic reviewer.\n\n"
    #     "Summarize the following paper specifically for the purpose of later "
    #     "comparing it to another paper that cites it.\n\n"
    #     "Focus ONLY on:\n"
    #     "- The core problem the paper addresses\n"
    #     "- The main method / approach proposed\n"
    #     "- The key technical contributions\n"
    #     "- Any limitations or open problems mentioned\n"
    #     "- The main results or claims of improvement\n\n"
    #     "Do NOT include:\n"
    #     "- Background or literature review\n"
    #     "- Detailed experimental setup\n"
    #     "- References or citations\n"
    #     "- General motivation or broad discussion\n\n"
    #     "Write a compact, information-dense summary (150–200 words) that preserves "
    #     "technical details needed to explain how a later paper could extend, "
    #     "improve, or diverge from this work.\n\n"
    #     "Paper:\n"
    #     f"Title: {parent.title}\n"
    #     f"Year: {parent.year}\n"
    #     "Content:\n"
    #     f"{parent_text}\n\n"
    # )

    # parent_summary = _call_gemini(
    #     client,
    #     prompt_for_parent_summary,
    #     max_output_tokens=300,
    # )
    # if not parent_summary:
    #     print("Got no output from Gemini for parent summary.")
    #     return ""
    # else:
    #     print("Parent summary done")
    #     parent.summary = parent_summary
    

    # prompt_for_child_summary = (
    #     "You are an expert academic reviewer.\n\n"
    #     "Summarize the following paper specifically for the purpose of later "
    #     "comparing it to another paper that cites it.\n\n"
    #     "Focus ONLY on:\n"
    #     "- The core problem the paper addresses\n"
    #     "- The main method / approach proposed\n"
    #     "- The key technical contributions\n"
    #     "- Any limitations or open problems mentioned\n"
    #     "- The main results or claims of improvement\n\n"
    #     "Do NOT include:\n"
    #     "- Background or literature review\n"
    #     "- Detailed experimental setup\n"
    #     "- References or citations\n"
    #     "- General motivation or broad discussion\n\n"
    #     "Write a compact, information-dense summary (150–200 words) that preserves "
    #     "technical details needed to explain how a later paper could extend, "
    #     "improve, or diverge from this work.\n\n"
    #     "Paper:\n"
    #     f"Title: {child.title}\n"
    #     f"Year: {child.year}\n"
    #     "Content:\n"
    #     f"{child_text}\n\n"
    # )

    # child_summary = _call_gemini(
    #     client,
    #     prompt_for_child_summary,
    #     max_output_tokens=300,
    # )
    # if not child_summary:
    #     print("Got no output from Gemini for child summary.")
    #     return ""
    # else:
    #     print("Child summary done")
    #     child.summary = child_summary

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

    final_text = _call_gemini(
        client,
        prompt,
        max_output_tokens=250,
    )

    if final_text:
        return _trim_to_last_sentence(final_text)

    

    # prompt = (
    #     "You are an expert academic reviewer. Given two papers — a parent "
    #     "(the citing paper) and a child (the cited paper) — write an"
    #     " explanation that covers:\n"
    #     "1. What specific idea or contribution in the child paper likely "
    #     "influenced the parent paper.\n"
    #     "2. How the parent paper improves on, extends, or diverges from "
    #     "the child paper's work.\n"
    #     "Be specific and reference concrete methods, findings, or concepts "
    #     "from each paper. Do not use bullet points.\n\n"
    #     f"Parent paper (citing, {parent.year or 'unknown year'}):\n"
    #     f"Title: {parent.title}\n"
    #     f"Content: {parent_text}\n\n"
    #     f"Child paper (cited, {child.year or 'unknown year'}):\n"
    #     f"Title: {child.title}\n"
    #     f"Content: {child_text}\n"
    #     "Please make sure the explanation is concise (under 200 words) and focuses on the most important connection between the papers."
    # )

    
    return ""
