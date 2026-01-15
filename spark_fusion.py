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


df = pd.read_parquet("output/unified_documents")
print(df.head())