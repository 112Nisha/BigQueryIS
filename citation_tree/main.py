"""CLI entry point for the Citation Tree Builder."""

from __future__ import annotations

import json
import os
import sys
import webbrowser
from concurrent.futures import ThreadPoolExecutor

import tika

tika.initVM()

from citation_tree.builder import TreeBuilder
from citation_tree.config import INPUT_PDF, OUTPUT_DIR, PDFS_DIR
from citation_tree.pdf import resolve_pdf_source
from citation_tree.renderer import (
    render_html_reference_tree,
    render_html_citation_tree,
)


def build_trees(
    source: str | None = None,
    output_dir: str | None = None,
    open_browser: bool = True,
) -> dict:
    source_value = source or INPUT_PDF
    pdf, error = resolve_pdf_source(source_value, PDFS_DIR)

    if not pdf:
        raise ValueError(error)

    if source_value.startswith(("http://", "https://")):
        print(f"\n  Source URL resolved to local PDF: {pdf}")

    citation_builder = TreeBuilder()
    reference_builder = TreeBuilder()

    print("\n  Building reference and citation trees in parallel")
    with ThreadPoolExecutor(max_workers=2) as ex:
        ref_future = ex.submit(reference_builder.build_reference_tree, pdf)
        cite_future = ex.submit(citation_builder.build_citation_tree, pdf)
        ref_tree = None
        cite_tree = None

        try:
            ref_tree = ref_future.result()
        except Exception as exc:
            print(f"  Reference tree build failed: {exc}")

        try:
            cite_tree = cite_future.result()
        except Exception as exc:
            print(f"  Citation tree build failed: {exc}")

    target_output_dir = output_dir or OUTPUT_DIR
    os.makedirs(target_output_dir, exist_ok=True)

    ref_json_path = os.path.join(target_output_dir, "reference_tree.json")
    if ref_tree is not None:
        with open(ref_json_path, "w", encoding="utf-8") as f:
            json.dump(ref_tree.to_json(), f, indent=2, ensure_ascii=False)
        print(f"  Reference JSON saved to: {ref_json_path}")

    cite_json_path = os.path.join(target_output_dir, "citation_tree.json")
    if cite_tree is not None:
        with open(cite_json_path, "w", encoding="utf-8") as f:
            json.dump(cite_tree.to_json(), f, indent=2, ensure_ascii=False)
        print(f"  Citation JSON saved to: {cite_json_path}")

    ref_html_path = os.path.join(target_output_dir, "reference_tree.html")
    if ref_tree is not None:
        render_html_reference_tree(
            ref_tree,
            ref_html_path,
            has_reference=ref_tree is not None,
            has_citation=cite_tree is not None,
        )

    cite_html_path = os.path.join(target_output_dir, "citation_tree.html")
    if cite_tree is not None:
        render_html_citation_tree(
            cite_tree,
            cite_html_path,
            has_reference=ref_tree is not None,
            has_citation=cite_tree is not None,
        )

    page_to_open = ref_html_path if ref_tree is not None else cite_html_path
    if open_browser and (ref_tree is not None or cite_tree is not None):
        webbrowser.open(f"file://{page_to_open}")

    ref_stats = (
        f"reference=({len(ref_tree.papers)} papers, {len(ref_tree.edges)} edges)"
        if ref_tree is not None
        else "reference=(failed)"
    )
    cite_stats = (
        f"citation=({len(cite_tree.papers)} papers, {len(cite_tree.edges)} edges)"
        if cite_tree is not None
        else "citation=(failed)"
    )
    print(f"\n  Done: {ref_stats}, {cite_stats}")

    return {
        "source": source_value,
        "resolved_pdf": pdf,
        "has_reference": ref_tree is not None,
        "has_citation": cite_tree is not None,
        "reference_json": ref_json_path if ref_tree is not None else None,
        "citation_json": cite_json_path if cite_tree is not None else None,
        "reference_html": ref_html_path if ref_tree is not None else None,
        "citation_html": cite_html_path if cite_tree is not None else None,
        "open_page": page_to_open if ref_tree is not None or cite_tree is not None else None,
    }


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else INPUT_PDF
    try:
        result = build_trees(source=source, output_dir=OUTPUT_DIR, open_browser=True)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    if not result["has_reference"] and not result["has_citation"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
