from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

df_unified = spark.read.parquet("output/unified_documents")
df_chunks  = spark.read.parquet("output/text_chunks")
df_kw      = spark.read.parquet("output/paper_keywords")
df_sent    = spark.read.parquet("output/paper_sentences")
df_ins     = spark.read.parquet("output/paper_insights")

# Preview
df_unified.show(5, truncate=80)
df_chunks.show(5, truncate=80)
df_kw.show(5, truncate=80)
df_sent.show(5, truncate=80)
df_ins.show(5, truncate=80)
