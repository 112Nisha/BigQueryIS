# ================================
# Spark: Structured + Unstructured Fusion
# ================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, explode, split
from pyspark.sql.types import StringType
import pandas as pd

# --------------------------------
# 1. Start Spark
# --------------------------------
spark = SparkSession.builder \
    .appName("arxiv-structured-unstructured-fusion") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# --------------------------------
# 2. File paths
# --------------------------------
STRUCTURED_CSV = "arxiv_metadata.csv"
UNSTRUCTURED_JSONL = "data/tika_extracted.jsonl"

# --------------------------------
# 3. Load structured data (CSV)
# --------------------------------
df_structured = spark.read \
    .option("header", True) \
    .option("multiLine", True) \
    .option("quote", "\"") \
    .option("escape", "\"") \
    .option("mode", "PERMISSIVE") \
    .option("inferSchema", False) \
    .csv(STRUCTURED_CSV)

print("Structured schema:")
df_structured.printSchema()

# --------------------------------
# 4. Load unstructured data (Tika JSONL)
# --------------------------------
df_unstructured = spark.read.json(
    UNSTRUCTURED_JSONL
)

print("Unstructured schema:")
df_unstructured.printSchema()

# --------------------------------
# 5. ID normalization UDF (YOUR LOGIC)
# --------------------------------
def normalize_id(x):
    """
    Canonical arXiv ID normalization:
    - pad left side to 4 digits (LEFT padding)
    - pad right side to 4 digits (RIGHT padding)
    """
    if x is None:
        return None

    x = str(x)

    if "." not in x:
        return x

    left, right = x.split(".", 1)

    left = left.zfill(4)
    right = right.ljust(4, "0")

    return f"{left}.{right}"

normalize_id_udf = udf(normalize_id, StringType())

# --------------------------------
# 6. Normalize IDs on BOTH datasets
# --------------------------------
df_structured_norm = df_structured.withColumn(
    "norm_id",
    normalize_id_udf(col("id"))
)

df_unstructured_norm = df_unstructured.withColumn(
    "norm_id",
    normalize_id_udf(col("paper_id"))
)

# --------------------------------
# 7. Join structured + unstructured data
# --------------------------------
df_fused = df_structured_norm.join(
    df_unstructured_norm,
    on="norm_id",
    how="left"
)

# --------------------------------
# 8. Select unified schema
# --------------------------------
df_final = df_fused.select(
    col("id").alias("original_id"),
    col("title"),
    col("authors"),
    col("categories"),
    col("content_type"),
    col("created"),
    col("abstract"),
    col("full_text"),
    col("num_pages"),
    col("char_count"),
    col("word_count")
)

print("Final unified schema:")
df_final.printSchema()

# --------------------------------
# 9. Sanity checks (DO NOT SKIP)
# --------------------------------
total_rows = df_final.count()
rows_with_text = df_final.filter(col("full_text").isNotNull()).count()
rows_without_text = df_final.filter(col("full_text").isNull()).count()

print("===== SANITY CHECKS =====")
print(f"Total rows            : {total_rows}")
print(f"Rows WITH PDF text    : {rows_with_text}")
print(f"Rows WITHOUT PDF text : {rows_without_text}")

# --------------------------------
# 10. (Optional) Text chunking (Step 3)
# --------------------------------
df_chunks = df_final \
    .filter(col("full_text").isNotNull()) \
    .withColumn(
        "chunk_text",
        explode(split(col("full_text"), "\n\n"))
    ) \
    .select(
        col("original_id"),
        col("chunk_text")
    )

print("Chunked schema:")
df_chunks.printSchema()

# --------------------------------
# 11. Save outputs (optional)
# --------------------------------

# Save unified data (for BigQuery / Hive / Parquet)
df_final.write.mode("overwrite").parquet("output/unified_documents")

# Save chunked text
df_chunks.write.mode("overwrite").parquet("output/text_chunks")

print("✅ Spark fusion pipeline completed successfully")


# ================================
# INSIGHT EXTRACTION PIPELINE
# Using YAKE keywords + TextRank-style sentence scoring
# ================================

import re
import yake
from collections import Counter
from pyspark.sql.functions import lower, when, split, trim, length, collect_list, concat_ws
from pyspark.sql.types import ArrayType, StructType, StructField, FloatType
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number

# --------------------------------
# 12. Extract sections with improved regex
# --------------------------------
def extract_sections(text):
    """
    Extract paper sections using regex patterns.
    Handles various heading formats in academic papers.
    """
    if text is None:
        return []

    # Patterns for section headers (handles "1. Introduction", "INTRODUCTION", "1 Introduction", etc.)
    section_patterns = [
        (r'(?i)(?:^|\n)\s*(?:\d+\.?\s*)?abstract[:\s]*\n', "abstract"),
        (r'(?i)(?:^|\n)\s*(?:\d+\.?\s*)?introduction[:\s]*\n', "introduction"),
        (r'(?i)(?:^|\n)\s*(?:\d+\.?\s*)?(?:related\s+work|background|literature\s+review)[:\s]*\n', "background"),
        (r'(?i)(?:^|\n)\s*(?:\d+\.?\s*)?(?:method|methodology|approach|proposed\s+method)[:\s]*\n', "method"),
        (r'(?i)(?:^|\n)\s*(?:\d+\.?\s*)?(?:experiment|evaluation|results|empirical)[:\s]*\n', "results"),
        (r'(?i)(?:^|\n)\s*(?:\d+\.?\s*)?(?:discussion)[:\s]*\n', "discussion"),
        (r'(?i)(?:^|\n)\s*(?:\d+\.?\s*)?(?:conclusion|concluding\s+remarks|summary)[:\s]*\n', "conclusion"),
    ]

    positions = []
    for pattern, name in section_patterns:
        match = re.search(pattern, text)
        if match:
            positions.append((match.end(), name))

    if not positions:
        # Fallback: use simple keyword search
        text_l = text.lower()
        for name in ["abstract", "introduction", "method", "results", "conclusion"]:
            idx = text_l.find(name)
            if idx != -1:
                positions.append((idx, name))

    if not positions:
        return []

    positions.sort()
    sections = []

    for i, (start, name) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else min(start + 15000, len(text))
        section_text = text[start:end].strip()
        # Limit section length to avoid memory issues
        if len(section_text) > 50:
            sections.append({
                "section_name": name,
                "section_text": section_text[:15000]
            })

    return sections


section_schema = ArrayType(
    StructType([
        StructField("section_name", StringType(), True),
        StructField("section_text", StringType(), True)
    ])
)

extract_sections_udf = udf(extract_sections, section_schema)

# --------------------------------
# 13. YAKE Keyword Extraction
# --------------------------------
def extract_keywords_yake(text, top_n=10):
    """
    Extract keywords using YAKE (Yet Another Keyword Extractor).
    Returns top keywords with their scores.
    """
    if not text or len(text) < 50:
        return []
    
    try:
        # YAKE configuration for academic text
        kw_extractor = yake.KeywordExtractor(
            lan="en",
            n=3,  # max n-gram size
            dedupLim=0.7,  # deduplication threshold
            top=top_n,
            features=None
        )
        keywords = kw_extractor.extract_keywords(text[:10000])  # Limit input size
        # Return keywords (lower score = more important in YAKE)
        return [{"keyword": kw, "score": float(1 - score)} for kw, score in keywords]
    except Exception:
        return []


keyword_schema = ArrayType(
    StructType([
        StructField("keyword", StringType(), True),
        StructField("score", FloatType(), True)
    ])
)

extract_keywords_udf = udf(extract_keywords_yake, keyword_schema)

# --------------------------------
# 14. TextRank-style Sentence Scoring
# --------------------------------
def score_sentences_textrank(text, top_n=3):
    """
    Score sentences using a simplified TextRank approach.
    Ranks sentences by their similarity to the overall document.
    """
    if not text or len(text) < 100:
        return []
    
    try:
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
        
        if not sentences:
            return []
        
        # Tokenize and count word frequencies
        def tokenize(s):
            return re.findall(r'\b[a-z]{3,}\b', s.lower())
        
        # Document-level word frequencies
        all_words = []
        for s in sentences:
            all_words.extend(tokenize(s))
        word_freq = Counter(all_words)
        
        # Score each sentence by sum of word importance
        scored = []
        for sent in sentences:
            words = tokenize(sent)
            if not words:
                continue
            # Score = average word frequency (normalized by sentence length)
            score = sum(word_freq.get(w, 0) for w in words) / len(words)
            # Penalize very short or very long sentences
            if 50 < len(sent) < 500:
                score *= 1.2
            scored.append({"sentence": sent, "score": float(score)})
        
        # Sort by score descending
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_n]
    
    except Exception:
        return []


sentence_schema = ArrayType(
    StructType([
        StructField("sentence", StringType(), True),
        StructField("score", FloatType(), True)
    ])
)

score_sentences_udf = udf(score_sentences_textrank, sentence_schema)

# --------------------------------
# 15. Apply section extraction
# --------------------------------
df_sections = df_final \
    .filter(col("full_text").isNotNull()) \
    .withColumn("sections", extract_sections_udf(col("full_text"))) \
    .select(col("original_id"), col("title"), explode(col("sections")).alias("sec")) \
    .select(
        col("original_id"),
        col("title"),
        col("sec.section_name").alias("section_name"),
        col("sec.section_text").alias("section_text")
    )

print("Sections schema:")
df_sections.printSchema()

# --------------------------------
# 16. Assign semantic roles
# --------------------------------
df_roles = df_sections.withColumn(
    "semantic_role",
    when(col("section_name") == "abstract", "OVERVIEW")
    .when(col("section_name") == "introduction", "MOTIVATION")
    .when(col("section_name") == "background", "CONTEXT")
    .when(col("section_name").isin("method", "methodology"), "APPROACH")
    .when(col("section_name").isin("results", "discussion"), "FINDINGS")
    .when(col("section_name") == "conclusion", "TAKEAWAY")
)

# --------------------------------
# 17. Extract keywords per section
# --------------------------------
df_keywords = df_roles \
    .withColumn("keywords", extract_keywords_udf(col("section_text"))) \
    .select(
        col("original_id"),
        col("title"),
        col("section_name"),
        col("semantic_role"),
        explode(col("keywords")).alias("kw")
    ) \
    .select(
        col("original_id"),
        col("title"),
        col("section_name"),
        col("semantic_role"),
        col("kw.keyword").alias("keyword"),
        col("kw.score").alias("keyword_score")
    )

print("Keywords schema:")
df_keywords.printSchema()

# --------------------------------
# 18. Extract top sentences per section
# --------------------------------
df_top_sentences = df_roles \
    .withColumn("top_sentences", score_sentences_udf(col("section_text"))) \
    .select(
        col("original_id"),
        col("title"),
        col("section_name"),
        col("semantic_role"),
        explode(col("top_sentences")).alias("sent")
    ) \
    .select(
        col("original_id"),
        col("title"),
        col("section_name"),
        col("semantic_role"),
        col("sent.sentence").alias("key_sentence"),
        col("sent.score").alias("sentence_score")
    )

print("Top sentences schema:")
df_top_sentences.printSchema()

# --------------------------------
# 19. Create paper-level insights summary
# --------------------------------
# Aggregate keywords per paper
df_paper_keywords = df_keywords \
    .filter(col("keyword_score") > 0.3) \
    .groupBy("original_id", "title") \
    .agg(
        collect_list("keyword").alias("all_keywords")
    )

# Get best sentence per semantic role
window = Window.partitionBy("original_id", "semantic_role").orderBy(col("sentence_score").desc())

df_best_sentences = df_top_sentences \
    .filter(col("semantic_role").isNotNull()) \
    .withColumn("rank", row_number().over(window)) \
    .filter(col("rank") == 1) \
    .select("original_id", "semantic_role", "key_sentence")

# Pivot to get one row per paper with all insights
df_insights = df_best_sentences \
    .groupBy("original_id") \
    .pivot("semantic_role") \
    .agg(concat_ws(" ", collect_list("key_sentence")))

# Join keywords with insights
df_paper_insights = df_paper_keywords.join(df_insights, on="original_id", how="left")

print("Paper insights schema:")
df_paper_insights.printSchema()

# --------------------------------
# 20. Save outputs
# --------------------------------

# Save detailed keywords
df_keywords.write \
    .mode("overwrite") \
    .parquet("output/paper_keywords")

# Save top sentences
df_top_sentences.write \
    .mode("overwrite") \
    .parquet("output/paper_sentences")

# Save paper-level insights summary
df_paper_insights.write \
    .mode("overwrite") \
    .parquet("output/paper_insights")

print("✅ Insight extraction completed!")

# --------------------------------
# 21. Display sample results
# --------------------------------
import pandas as pd

print("\n" + "="*60)
print("SAMPLE PAPER INSIGHTS")
print("="*60)

df_pd = pd.read_parquet("output/paper_insights")
pd.set_option("display.max_colwidth", 100)
pd.set_option("display.width", None)
print(df_pd.head(5))

print("\n" + "="*60)
print("SAMPLE KEYWORDS")
print("="*60)

df_kw = pd.read_parquet("output/paper_keywords")
print(df_kw.head(20))
