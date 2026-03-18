"""CLI entry point for the Citation Tree Builder."""

from __future__ import annotations

import json
import os
import sys
import webbrowser

import tika

tika.initVM()

from citation_tree.builder import TreeBuilder
from citation_tree.config import INPUT_PDF, OUTPUT_DIR, PDFS_DIR
from citation_tree.renderer import render_html_reference_tree, render_html_citation_tree


def main():
    pdf = sys.argv[1] if len(sys.argv) > 1 else INPUT_PDF

    # Resolve relative names inside pdfs/
    if not os.path.isabs(pdf) and not os.path.exists(pdf):
        pdf = os.path.join(PDFS_DIR, pdf)

    if not os.path.exists(pdf):
        print(f"Error: file not found - {pdf}")
        sys.exit(1)

    builder = TreeBuilder()
    # tree = builder.build_reference_tree(pdf)

    # # Save JSON of the tree structure 
    # json_path = os.path.join(OUTPUT_DIR, "citation_tree.json")
    # with open(json_path, "w", encoding="utf-8") as f:
    #     json.dump(tree.to_json(), f, indent=2, ensure_ascii=False)
    # print(f"  JSON saved to: {json_path}")

    # # Render HTML page
    # html_path = os.path.join(OUTPUT_DIR, "citation_tree.html")
    # render_html(tree, html_path)

    # # Open in browser
    # webbrowser.open(f"file://{html_path}")

    # print(f"\n  Done: ({len(tree.papers)} papers, {len(tree.edges)} edges)")

    tree = builder.build_citation_tree(pdf)

    # Save JSON of the tree structure 
    json_path = os.path.join(OUTPUT_DIR, "citation_tree.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(tree.to_json(), f, indent=2, ensure_ascii=False)
    print(f"  JSON saved to: {json_path}")

    # Render HTML page
    html_path = os.path.join(OUTPUT_DIR, "citation_tree.html")
    render_html_citation_tree(tree, html_path)

    # Open in browser
    webbrowser.open(f"file://{html_path}")

    print(f"\n  Done: ({len(tree.papers)} papers, {len(tree.edges)} edges)")


if __name__ == "__main__":
    main()
