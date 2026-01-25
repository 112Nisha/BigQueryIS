# BigQueryIS

## Set-up
```
pip install tika
pip install pyspark
pip install pyarrow
pip install fastparquet
pip install apache-airflow
pip install fitz
pip install yake
```

**Environment Variables**
```
export AIRFLOW_HOME=<path/to/dir/airflow> # note the airflow dir
```

## Order to run the files in:

1. getData
2. convertData
3. 

## Running Airflow
```
airflow db reset -y
airflow standalone
```
Now for the username and pwd, check the airflow directory on home (or wherever you installed airflow) and run
`/airflow/simple_auth_manager_passwords.json.generated`

## Dataset
[Yelp Dataset](https://business.yelp.com/data/resources/open-dataset/)

# Tika:

## 1. tika_extract_enhanced.py – Enhanced Tika-Only

This improves on the original with:

- **Table Detection**  
  Detects tables from both text patterns and XML output

- **Math Equation Extraction**  
  Identifies numbered equations, LaTeX patterns, and mathematical symbols  
  (∑, ∫, ∂, Greek letters, etc.)

- **LaTeX Preservation**  
  Converts LaTeX commands to Unicode  
  - `\alpha` → `α`  
  - `\frac{a}{b}` → `((a)/(b))`

- **Section Extraction**  
  Identifies document structure  
  (Abstract, Introduction, Methods, etc.)

- **Figure / Caption Detection**  
  Extracts captions such as:  
  - `Figure 1:`  
  - `Table 2:`

- **Reference Extraction**  
  Parses the References / Bibliography section

- **Improved Text Cleaning**  
  - Fixes hyphenation  
  - Removes arXiv artifacts  
  - Preserves paragraph structure

---

## 2. tika_extract_advanced.py – PyMuPDF + Tika Hybrid

This provides superior extraction using PyMuPDF:

- **Better Table Detection**  
  Uses PyMuPDF's built-in `find_tables()` plus position-based detection

- **Font-Based Math Detection**  
  Identifies math content by detecting mathematical fonts  
  (CMSY, CMMI, Symbol)

- **Structured Block Extraction**  
  Preserves document layout with bounding boxes

- **Markdown Table Output**  
  Auto-converts tables to Markdown format

- **Confidence Scoring**  
  Rates how likely extracted equations are real math


# Spark

## 1. YAKE Keyword Extraction (replaces TF-IDF word count)

- Extracts meaningful keyphrases (up to 3-grams)  
  Examples:  
  - "neural network architecture"  
  - "transfer learning"
- Lightweight, no GPU needed
- Much more informative than single word counts

## 2. TextRank-style Sentence Scoring

- Scores sentences by how well they represent the document's content
- Considers word frequency importance across the full text
- Penalizes too-short or too-long sentences
- Returns top 3 sentences per section instead of just 1


## 3. Better Section Detection

- Uses regex patterns to handle various heading formats  
  (1. Introduction, INTRODUCTION, etc.)
- Detects more section types:  
  - background  
  - discussion  
  - related work

## 4. Richer Semantic Roles

- **OVERVIEW (abstract)** – What the paper is about  
- **MOTIVATION (introduction)** – Why this work matters  
- **CONTEXT (background)** – What's been done before  
- **APPROACH (methods)** – How they did it  
- **FINDINGS (results)** – What they found  
- **TAKEAWAY (conclusion)** – Key conclusions


## 5. Three Output Files

| Output | Description |
|------|-------------|
| `output/paper_keywords/` | All keywords per section with scores |
| `output/paper_sentences/` | Top sentences per section |
| `output/paper_insights/` | Paper-level summary with best sentence per role + all keywords |
