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


from pyspark.sql.functions import lower, regexp_extract
from pyspark.sql.types import ArrayType, StructType, StructField
def extract_sections(text):
    if text is None:
        return []

    text_l = text.lower()

    markers = [
        ("abstract", "abstract"),
        ("introduction", "introduction"),
        ("method", "method"),
        ("methodology", "methodology"),
        ("experiment", "experiment"),
        ("results", "results"),
        ("conclusion", "conclusion")
    ]

    positions = []
    for name, marker in markers:
        idx = text_l.find(marker)
        if idx != -1:
            positions.append((idx, name))

    if not positions:
        return []

    positions.sort()
    sections = []

    for i, (start, name) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        sections.append({
            "section_name": name,
            "section_text": text[start:end].strip()
        })

    return sections


section_schema = ArrayType(
    StructType([
        StructField("section_name", StringType(), True),
        StructField("section_text", StringType(), True)
    ])
)

extract_sections_udf = udf(extract_sections, section_schema)


df_sections = df_final \
    .filter(col("full_text").isNotNull()) \
    .withColumn("sections", extract_sections_udf(col("full_text"))) \
    .select(col("original_id"), explode(col("sections")).alias("sec")) \
    .select(
        col("original_id"),
        col("sec.section_name"),
        col("sec.section_text")
    )

df_sections.printSchema()


from pyspark.sql.functions import when

df_roles = df_sections.withColumn(
    "semantic_role",
    when(col("section_name") == "abstract", "WHAT")
    .when(col("section_name") == "introduction", "WHY")
    .when(col("section_name").isin("method", "methodology", "experiment", "results"), "HOW")
    .when(col("section_name") == "conclusion", "WHY")
)
    
from pyspark.sql.functions import split, explode, trim, length

df_sentences = df_roles \
    .withColumn("sentence", explode(split(col("section_text"), r'(?<=[.!?])\s+'))) \
    .withColumn("sentence", trim(col("sentence"))) \
    .filter(length(col("sentence")) > 10)


from pyspark.ml.feature import Tokenizer, HashingTF, IDF

tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
words_df = tokenizer.transform(df_sentences)

hashingTF = HashingTF(
    inputCol="words",
    outputCol="rawFeatures",
    numFeatures=10000
)

tf_df = hashingTF.transform(words_df)

idf = IDF(inputCol="rawFeatures", outputCol="tfidf")
idf_model = idf.fit(tf_df)
tfidf_df = idf_model.transform(tf_df)


from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, size
df_scored = tfidf_df.withColumn(
    "score",
    size(col("words"))
)

window = Window.partitionBy("original_id", "semantic_role").orderBy(col("score").desc())

df_summary = df_scored \
    .filter(col("semantic_role").isNotNull()) \
    .withColumn("rank", row_number().over(window)) \
    .filter(col("rank") == 1) \
    .select("original_id", "semantic_role", "sentence")


df_summary.write \
    .mode("overwrite") \
    .parquet("output/paper_explanations")

import pandas as pd

df_pd = pd.read_parquet("output/paper_explanations")
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", None)

print(df_pd.head(10))
