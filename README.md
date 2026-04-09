# Citation Tree Builder

Build an interactive citation tree from a local PDF or an online paper URL.  
Discovers related papers via **Semantic Scholar**, **arXiv**, and **OpenAlex**, uses ML to explain how each paper improved upon its parent, and renders a self-contained HTML visualization.

## Features

- **PDF extraction** — title, abstract, references, arXiv ID via Apache Tika
- **Online URL support** — accepts direct PDF links or paper landing-page URLs and auto-downloads the PDF
- **Multi-source discovery** — Semantic Scholar, arXiv, OpenAlex
- **ML-powered explanations** — Groq-hosted OpenAI-compatible models generate improvement summaries between parent and child papers
- **Semantic similarity** — sentence-transformers cosine similarity scores
- **Interactive HTML output** — dark-themed, searchable, with detail panel

## Setup

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install requests tika numpy openai sentence-transformers
```

> Tika requires **Java 8+** on your system.

### 2. Configure LLM explanations (optional)

The project currently uses a Groq-hosted OpenAI-compatible endpoint in `citation_tree/ml.py`.
To enable paper improvement summaries, set `GROQ_API_KEY` in `citation_tree/config.py`.

If you do not want LLM-generated explanations, set:

```python
LLM_EXPLANATIONS_ENABLED = False
```

Optional OCR/image extraction dependencies (only needed for scanned PDFs):

```bash
pip install pymupdf pillow pytesseract
```

## Usage

### Run from an online URL (recommended)

```bash
# direct PDF URL
python -m citation_tree "https://arxiv.org/pdf/1706.03762.pdf"

# paper page URL (the tool will try to discover a PDF link)
python -m citation_tree "https://arxiv.org/abs/1706.03762"
```

### Other supported inputs

```bash
# local path
python -m citation_tree pdfs/1211.3711v1.pdf

# bare filename (auto-resolves inside pdfs/)
python -m citation_tree 1211.3711v1.pdf
```

If you run `python -m citation_tree` with no argument, it falls back to `INPUT_PDF` in `citation_tree/config.py`.

Output is written to `output/`:
- `reference_tree.json` - structured reference tree
- `citation_tree.json` - structured citation tree
- `reference_tree.html` - interactive reference tree view
- `citation_tree.html` - interactive citation tree view

When generation succeeds, one of the HTML files is opened automatically in your browser.

### URL input notes

- Works best with public pages that expose a downloadable PDF.
- If a page is paywalled or requires login/cookies, provide a direct PDF URL or download the PDF locally and pass the file path.
- Downloaded remote PDFs are cached in `pdfs/` and reused on future runs.

### Typical workflow

1. Use a paper URL or put your target PDF in `pdfs/`.
2. Run `python -m citation_tree <url-or-filename-or-path>`.
3. Open `output/reference_tree.html` and `output/citation_tree.html`.
4. Tune limits in `citation_tree/config.py` and rerun if you need broader/deeper trees.

### Notes on first run

- Apache Tika may take extra time the first time it starts.
- Sentence-transformers may download model files on first use.
- Tree expansion can take time because it queries Semantic Scholar, OpenAlex, and arXiv.

## Configuration

Edit `citation_tree/config.py`:

| Variable | Default | Description |
|---|---|---|
| `INPUT_PDF` | `pdfs/1211.3711v1.pdf` | Default input PDF |
| `MAX_DEPTH` | `2` | How deep the tree expands |
| `MAX_PAPERS` | `60` | Cap on total papers discovered |
| `MIN_RELEVANCE` | `0.15` | Score threshold for including a paper |
| `MAX_CHILDREN_PER_NODE` | `2` | Max children kept per expanded node |
| `LLM_EXPLANATIONS_ENABLED` | `True` | Enable/disable LLM improvement summaries |
| `GROQ_API_KEY` | `""` | API key used by the OpenAI-compatible Groq client |

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
