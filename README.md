# Citation Tree Builder

Build an interactive citation tree from a local PDF or an online paper URL.  
Discovers related papers via **Semantic Scholar**, **arXiv**, and **OpenAlex**, uses ML to explain how each paper improved upon its parent, and renders a self-contained HTML visualization.

## Features

- **PDF extraction** — title, abstract, references, arXiv ID via Apache Tika
- **Online URL support** — accepts direct PDF links or paper landing-page URLs and auto-downloads the PDF
- **Multi-source discovery** — Semantic Scholar, arXiv, OpenAlex
- **ML-powered explanations** — Groq and/or Gemini generate improvement summaries between parent and child papers
- **Semantic similarity** — sentence-transformers cosine similarity scores
- **Interactive HTML output** — dark-themed, searchable, with detail panel
- **Simple web UI** — paste a paper link, wait on a loading page, then view generated HTML with a back-to-home button

## Setup

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> Tika requires **Java 8+** on your system.

### 2. Configure LLM explanations (optional)

To enable paper improvement summaries, set one or both of these environment variables:
- `GROQ_API_KEY`
- `GEMINI_API_KEY`

When both are set, `LLM_PROVIDER=auto` uses Groq first and falls back to Gemini if needed.

If you do not want LLM-generated explanations, set:

```bash
export LLM_EXPLANATIONS_ENABLED=false
```

Example shell setup:

```bash
cp .env.example .env
export GROQ_API_KEY="your_key_here"
export GEMINI_API_KEY="your_key_here"
export SEMANTIC_SCHOLAR_API_KEY="your_key_here"
export OPENALEX_API_KEY="your_key_here"
```

Everything else has defaults in code. Optional non-key settings you can add only if needed:

```bash
# Optional (not API keys)
export OPENALEX_MAILTO="you@example.com"
export ARXIV_CONTACT_EMAIL="you@example.com"
export LOW_MEMORY_MODE=true
export DELETE_PDFS_AFTER_USE=true
```

By default, downloaded PDFs are deleted after extraction (`DELETE_PDFS_AFTER_USE=true`).

Optional OCR/image extraction dependencies (only needed for scanned PDFs):

```bash
pip install pymupdf pillow pytesseract
```

## Usage

### Local development quick start

1. Clone the repo and enter the project folder.
2. Create and activate a virtual environment.
3. Install dependencies.
4. Export your API keys (or load them from a local `.env` file).
5. Start the web app.
6. Open the local URL.

```bash
git clone <your-repo-url>
cd BigQueryIS

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Option A: export variables directly
export GROQ_API_KEY="your_key_here"
export GEMINI_API_KEY="your_key_here"
export SEMANTIC_SCHOLAR_API_KEY="your_key_here"
export OPENALEX_API_KEY="your_key_here"

# Option B: load from local .env (never commit .env)
cp .env.example .env
set -a
source .env
set +a

python web_app.py
```

Open: `http://localhost:8000`

### Run the web app (simple hosted UI)

```bash
python web_app.py
```

Then open:

```text
http://localhost:8000
```

What the web app does:
- asks for a paper URL (or local PDF path)
- shows a loading/status page while trees are built
- opens a result page with the generated HTML embedded
- includes a persistent **Back to start** button

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

## Hosting

Recommended stack for simple hosting:
- **Flask** app (`web_app.py`) as the web server
- **Gunicorn** as the production process manager
- **Docker** runtime with Java installed for Tika

Production start command:

```bash
gunicorn -w 1 -b 0.0.0.0:${PORT:-8000} web_app:app
```

Notes for cloud deploy (Render/Railway/Fly.io/etc.):
- install command: `pip install -r requirements.txt`
- start command: `gunicorn -w 1 -b 0.0.0.0:$PORT web_app:app`
- ensure Java is available for Tika
- keep API keys in environment variables/secrets (not hardcoded)

### Deploy to Render (public URL)

This repo now includes `render.yaml` + `Dockerfile`, so Render can deploy it directly.

1. Push this repo to GitHub.
2. Create a Render account and click **New +** -> **Blueprint**.
3. Connect your GitHub repo and select this project.
4. Render will detect `render.yaml` and create the web service.
5. In Render service settings, set secret env var `GROQ_API_KEY`.
    - Optional additional API keys: `GEMINI_API_KEY`, `SEMANTIC_SCHOLAR_API_KEY`, `OPENALEX_API_KEY`.
6. Deploy. After the build finishes, Render gives you a public URL like:
    - `https://bigqueryis-web.onrender.com`
7. Share that URL. Users can open it, submit a paper link, and use the generated HTML views.

Memory note:
- `render.yaml` sets `LOW_MEMORY_MODE=true` by default for safer free-tier deploys.
- In low-memory mode, tree builds run sequentially and semantic similarity model loading is disabled.
- This reduces memory usage substantially, with a small quality tradeoff in semantic filtering.

### Make changes and reflect them on Render

Code changes flow:

1. Make your code change locally.
2. Test locally (`python web_app.py`).
3. Commit and push to GitHub.
4. If you deploy from `main` and `autoDeploy` is enabled, Render automatically starts a new deploy.
5. Open Render logs and wait until deploy status is `Live`.

Example:

```bash
git add .
git commit -m "Update citation tree behavior"
git push origin main
```

Config/secret changes flow:

1. Open your Render service dashboard.
2. Go to **Environment**.
3. Add or update variables (for example `GROQ_API_KEY`, `GEMINI_API_KEY`).
4. Save changes and trigger a redeploy when prompted.

`render.yaml` changes:

1. Commit and push the `render.yaml` update.
2. In Render, apply blueprint/infrastructure updates if prompted.
3. Redeploy and verify the service is `Live`.

Notes:
- Free Render services can sleep when idle and need a short cold-start wakeup.
- If you want to run without LLM summaries, set `LLM_EXPLANATIONS_ENABLED=false`.
- If you still hit memory limits, keep `LOW_MEMORY_MODE=true` and reduce `MAX_PAPERS` (for example to `30`).

### Notes on first run

- Apache Tika may take extra time the first time it starts.
- Sentence-transformers may download model files on first use.
- Tree expansion can take time because it queries Semantic Scholar, OpenAlex, and arXiv.

## Configuration

Set these via environment variables (defaults are defined in `citation_tree/config.py`):

| Variable | Default | Description |
|---|---|---|
| `INPUT_PDF` | `pdfs/1211.3711v1.pdf` | Default input PDF |
| `MAX_DEPTH` | `2` | How deep the tree expands |
| `MAX_PAPERS` | `60` | Cap on total papers discovered |
| `MIN_RELEVANCE` | `0.15` | Score threshold for including a paper |
| `MAX_CHILDREN_PER_NODE` | `2` | Max children kept per expanded node |
| `LLM_EXPLANATIONS_ENABLED` | `True` | Enable/disable LLM improvement summaries |
| `LLM_PROVIDER` | `auto` | `auto`, `groq`, or `gemini` |
| `DELETE_PDFS_AFTER_USE` | `True` | Delete downloaded PDFs after extraction to save disk space |
| `GROQ_API_KEY` | `""` | API key for Groq |
| `GEMINI_API_KEY` | `""` | API key for Gemini |
| `SEMANTIC_SCHOLAR_API_KEY` | `""` | Optional key for Semantic Scholar higher limits |
| `OPENALEX_API_KEY` | `""` | Optional OpenAlex API key |
| `OPENALEX_MAILTO` | `""` | Contact email for OpenAlex polite pool |
| `ARXIV_CONTACT_EMAIL` | `""` | Contact email used in arXiv User-Agent |

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
