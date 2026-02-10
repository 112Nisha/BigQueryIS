# Citation Tree Builder

Build an interactive citation tree from a local PDF.  
Discovers related papers via **Semantic Scholar**, **arXiv**, and **OpenAlex**, uses ML to explain how each paper improved upon its parent, and renders a self-contained HTML visualization.

## Features

- **PDF extraction** — title, abstract, references, arXiv ID via Apache Tika
- **Multi-source discovery** — Semantic Scholar, arXiv, OpenAlex
- **ML-powered explanations** — Flan-T5 generates improvement summaries between parent and child papers
- **Semantic similarity** — sentence-transformers cosine similarity scores
- **Interactive HTML output** — dark-themed, searchable, with detail panel

## Setup

```bash
pip install requests tika

# optional — enables ML explanations & similarity scores
pip install sentence-transformers transformers numpy
```

> Tika requires Java 8+ on your system.

## Usage

```bash
# default: uses pdfs/0704.0001.pdf
python -m citation_tree

# override with a specific PDF
python -m citation_tree pdfs/some_paper.pdf
```

Output is written to `output/`:
- `citation_tree.json` — structured tree data
- `citation_tree.html` — interactive visualization (auto-opens in browser)

## Configuration

Edit `citation_tree/config.py`:

| Variable | Default | Description |
|---|---|---|
| `INPUT_PDF` | `pdfs/0704.0001.pdf` | Default input PDF |
| `MAX_DEPTH` | `2` | How deep the tree expands |
| `MAX_PAPERS` | `20` | Cap on total papers discovered |
| `MIN_RELEVANCE` | `0.15` | Score threshold for including a paper |

## Project Structure

```
citation_tree/            # main package
    __init__.py
    __main__.py           # python -m citation_tree entry point
    main.py               # CLI orchestration
    config.py             # paths, API URLs, tuning parameters
    models.py             # Paper & CitationTree dataclasses
    cache.py              # disk cache + rate limiter
    text_utils.py         # keyword extraction, title matching
    ml.py                 # similarity & improvement explanations
    pdf.py                # Tika-based PDF extraction
    builder.py            # tree construction orchestrator
    renderer.py           # HTML/JS visualization output
    clients/              # API client sub-package
        __init__.py
        base.py           # shared caching/rate-limit base class
        arxiv.py          # arXiv API
        semantic_scholar.py  # Semantic Scholar API
        openalex.py       # OpenAlex API
pdfs/                     # input PDFs (gitignored)
output/                   # generated JSON + HTML (gitignored)
.cache/                   # API response cache (gitignored)
```
