from pyspark.sql import SparkSession
from pyspark.sql.functions import col, length, trim
from pyspark.sql.functions import regexp_replace, trim
import os

# ADD HIVE SUPPORT

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STRUCTURED_CSV = os.path.join(BASE_DIR, "data", "structured", "diabetes.csv")
EXTRACTED_JSON = os.path.join(BASE_DIR, "data", "extracted_text.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "output", "fused_multimodal_generic")
import shutil

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

spark = SparkSession.builder \
    .appName("Generic-Multimodal-Fusion") \
    .getOrCreate()

structured_df = spark.read.csv(
    STRUCTURED_CSV,
    header=True,
    inferSchema=True
)

structured_df = structured_df.toDF(
    *[c.strip() for c in structured_df.columns]
)

unstructured_df = spark.read.option("multiLine", True).json(
    EXTRACTED_JSON
)



if "text" in unstructured_df.columns:
    unstructured_df = unstructured_df.withColumn(
        "text",
        trim(
            regexp_replace(col("text"), r"\s+", " ")
        )
    )

structured_cols = structured_df.columns
unstructured_cols = unstructured_df.columns

for c in unstructured_cols:
    if c in structured_cols:
        unstructured_df = unstructured_df.withColumnRenamed(
            c, f"unstructured_{c}"
        )

fused_df = structured_df.join(
    unstructured_df,
    structured_df["document_name"] == unstructured_df["file_name"],
    how="left"
)

final_df = fused_df

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "output",
    "fused_multimodal_generic",
    "demo"
)

os.makedirs(os.path.dirname(OUTPUT_DIR), exist_ok=True)

final_df.write.mode("overwrite").parquet(OUTPUT_DIR)

print("Spark fusion complete - merged unstructured and structured data")

# final_df.show(truncate=100)