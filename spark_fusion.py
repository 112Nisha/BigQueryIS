# Spark: Structured + Unstructured Fusion

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, explode, split
from pyspark.sql.types import StringType
import pandas as pd
import re
import yake
from collections import Counter
from pyspark.sql.functions import when, split, collect_list, concat_ws
from pyspark.sql.types import ArrayType, StructType, StructField, FloatType
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number
import textwrap
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number
from pyspark.sql.functions import struct, sort_array

"""
Since we are not using machine-learning models to extract the what (overview and motivation),
why (context), and how (approach), we first split the extracted text into logical paper sections. 
We then identify representative sentences within each section using heuristic, 
TextRank-style scoring based on word importance, and present these sentences to the user as insights.
True ML would let us do actual summarization (via models like transformers), but we can make more detailed 'summaries'
using the top n sentences and not top 1 sentence like we are currently doing
"""

"""
We can proceed by using bigquery to store information like the rows from df_paper_insights, paper_keywords, etc.
We can then run queries like 
SELECT title, OVERVIEW
FROM paper_insights
WHERE OVERVIEW IS NOT NULL

This will allow us to extract the how, what, and why of the papers, as well as additonal information like keywords.
Additionally, we can add more information like metadata, titles, etc. into other bigquery tables so that querying can be done
for other relevant information.
"""

num_sentences = 3  # number of top n sentences to take after TextRank
num_papers = 5 # number of papers to display as output

# Starting Spark - initializing the Spark application and its runtime environment
spark = SparkSession.builder.appName("arxiv-structured-unstructured-fusion").getOrCreate()
spark.sparkContext.setLogLevel("WARN")


# Loading the structured and unstructured data
STRUCTURED_CSV = "arxiv_metadata.csv"
UNSTRUCTURED_JSONL = "data/tika_extracted.jsonl"
df_structured = spark.read.option("header", True).option("multiLine", True).option("quote", "\"").option("escape", "\"") \
    .option("mode", "PERMISSIVE").option("inferSchema", False).csv(STRUCTURED_CSV)

print("Structured schema:")
df_structured.printSchema()

df_unstructured = spark.read.json(UNSTRUCTURED_JSONL)

print("Unstructured schema:")
df_unstructured.printSchema()

# Normalizing the pdf IDs so that they can be mapped (they are the joining column bw the structured and unstructured data)
# pads left side to 4 digits and right side to 4 digits
def normalize_id(x):
    if x is None:
        return None
    x = str(x)

    if "." not in x:
        return x

    left, right = x.split(".", 1)
    left = left.zfill(4)
    right = right.ljust(4, "0")
    return f"{left}.{right}"

# Normalizing IDs for both dataframes and adding the new normalized ID column under the norm_id column
# A normalizing IDs user defined function is created for this
normalize_id_udf = udf(normalize_id, StringType())

df_structured_norm = df_structured.withColumn("norm_id",normalize_id_udf(col("id")))
df_unstructured_norm = df_unstructured.withColumn("norm_id",normalize_id_udf(col("paper_id")))

# Joining the structured and unstructured dataframes on the normalized ID column
df_fused = df_structured_norm.join(df_unstructured_norm,on="norm_id",how="left")

# Creating the final unified schema with selected columns from both dataframes
df_final = df_fused.select(col("id").alias("original_id"),col("title"),col("authors"),col("categories"),col("content_type"),
    col("created"),col("abstract"),col("full_text"),col("num_pages"),col("char_count"),col("word_count"))

print("Final unified schema:")
df_final.printSchema()

# Sanity check to see how many rows have text extracted vs how many do not
total_rows = df_final.count()
rows_with_text = df_final.filter(col("full_text").isNotNull()).count()
rows_without_text = df_final.filter(col("full_text").isNull()).count()

print("===== SANITY CHECK RESULTS =====")
print(f"Total rows            : {total_rows}")
print(f"Rows WITH PDF text    : {rows_with_text}")
print(f"Rows WITHOUT PDF text : {rows_without_text}")

# Chunking the text so that the df_chunks dataframe is of the form (original_id, chunk_text) where chunk_text is a section of the paper
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

# Saving the outputs 
df_final.write.mode("overwrite").parquet("output/unified_documents")
df_chunks.write.mode("overwrite").parquet("output/text_chunks")

print("Completed basic spark fusion pipeline but without insight extraction")


# Extracting insights using YAKE keywords (Yet Another Keyword Extractor) + TextRank-style sentence scoring
# Since we are not using ML, YAKE finds important words and short phrases in the text without needing a model
# TextRank-style scoring ranks sentences based on their similarity to the overall document, helping identify key sentences

# Extrancting paper sections using regex patterns
def extract_sections(text):

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
    
    # Find all section headers and their positions
    positions = []
    for pattern, name in section_patterns:
        match = re.search(pattern, text)
        if match:
            positions.append((match.end(), name))
    
    # If no sections are found, trying a simple substring search 
    if not positions:
        text_l = text.lower()
        for name in ["abstract", "introduction", "method", "results", "conclusion"]:
            idx = text_l.find(name)
            if idx != -1:
                positions.append((idx, name))

    if not positions:
        return []

    positions.sort()
    sections = []
    
    # Extract sections based on found positions, but only consider sections with sufficient length for relevance 
    for i, (start, name) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else min(start + 15000, len(text))
        section_text = text[start:end].strip()
        if len(section_text) > 50:
            sections.append({"section_name": name,"section_text": section_text[:15000]})

    return sections

# Creating a user defined function (UDF) for section extraction
section_schema = ArrayType(StructType([StructField("section_name", StringType(), True),StructField("section_text", StringType(), True)]))
extract_sections_udf = udf(extract_sections, section_schema)

# Extracting keywords using YAKE
def extract_keywords_yake(text, top_n=10):
    if not text or len(text) < 50:
        return []
    # Getting a YAKE score from the KeywordExtractor and returning a list of keywords with their scores
    # Subtracting from 1 to convert to a format of higher score -> more relevance
    try:
        kw_extractor = yake.KeywordExtractor(lan="en",n=3,dedupLim=0.7,top=top_n,features=None)
        keywords = kw_extractor.extract_keywords(text[:10000])  
        return [{"keyword": kw, "score": float(1 - score)} for kw, score in keywords]
    except Exception:
        return []

# Creating a UDF for YAKE keyword extraction
keyword_schema = ArrayType(StructType([StructField("keyword", StringType(), True),StructField("score", FloatType(), True)]))
extract_keywords_udf = udf(extract_keywords_yake, keyword_schema)

# Scoring sentences using a simplified TextRank approach and ranking them by their similarity to the overall document
def score_sentences_textrank(text, top_n=3):
    if not text or len(text) < 100:
        return []
    
    try:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
        
        if not sentences:
            return []
        
        def tokenize(s):
            return re.findall(r'\b[a-z]{3,}\b', s.lower())
        
        # getting document-level word frequencies
        all_words = []
        for s in sentences:
            all_words.extend(tokenize(s))
        word_freq = Counter(all_words)
        
        # scoring sentences based on word frequencies where score = average word frequency (normalized by sentence length)
        scored = []
        for sent in sentences:
            words = tokenize(sent)
            if not words:
                continue
            score = sum(word_freq.get(w, 0) for w in words) / len(words)
            if 50 < len(sent) < 500:
                score *= 1.2
            scored.append({"sentence": sent, "score": float(score)})
        
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_n]
    
    except Exception:
        return []

# Creating a UDF for TextRank-style sentence scoring
sentence_schema = ArrayType(StructType([StructField("sentence", StringType(), True),StructField("score", FloatType(), True)]))
score_sentences_udf = udf(score_sentences_textrank, sentence_schema)

# extracting logical sections from each papers full text and flattening them into a section-level df 
# Each row represents a single section associated with its original paper
df_sections = df_final.filter(col("full_text").isNotNull()).withColumn("sections", extract_sections_udf(col("full_text"))) \
    .select(col("original_id"), col("title"), explode(col("sections")).alias("sec")) \
    .select(col("original_id"),col("title"),col("sec.section_name").alias("section_name"),col("sec.section_text").alias("section_text"))

print("Sections schema:")
df_sections.printSchema()

# assigning semantic roles to each section for drawing insights
df_roles = df_sections.withColumn(
    "semantic_role",
    when(col("section_name") == "abstract", "OVERVIEW")
    .when(col("section_name") == "introduction", "MOTIVATION")
    .when(col("section_name") == "background", "CONTEXT")
    .when(col("section_name").isin("method", "methodology"), "APPROACH")
    .when(col("section_name").isin("results", "discussion"), "FINDINGS")
    .when(col("section_name") == "conclusion", "TAKEAWAY")
)


# Extracting keywords for each section
df_keywords = df_roles.withColumn("keywords", extract_keywords_udf(col("section_text"))) \
    .select(col("original_id"),col("title"),col("section_name"),col("semantic_role"),explode(col("keywords")).alias("kw")) \
    .select(col("original_id"),col("title"),col("section_name"),col("semantic_role"),col("kw.keyword").alias("keyword"),col("kw.score").alias("keyword_score"))

print("Keywords schema:")
df_keywords.printSchema()

# Extracting the top sentences of each section
df_top_sentences = df_roles.withColumn("top_sentences", score_sentences_udf(col("section_text"))) \
    .select(col("original_id"),col("title"),col("section_name"),col("semantic_role"),explode(col("top_sentences")).alias("sent")) \
    .select(col("original_id"),col("title"),col("section_name"),col("semantic_role"),col("sent.sentence").alias("key_sentence"),col("sent.score").alias("sentence_score"))

print("Top sentences schema:")
df_top_sentences.printSchema()


# Aggregating high-confidence keywords at the paper level and selecting highest-scoring sentence for each semantic role within a paper
df_paper_keywords = df_keywords.filter(col("keyword_score") > 0.3).groupBy("original_id", "title").agg(collect_list("keyword").alias("all_keywords"))

window = Window.partitionBy("original_id", "semantic_role").orderBy(col("sentence_score").desc())
# df_best_sentences = df_top_sentences.filter(col("semantic_role").isNotNull()).withColumn("rank", row_number().over(window)) \
#     .filter(col("rank") == 1).select("original_id", "semantic_role", "key_sentence")

window_section = (Window.partitionBy("original_id", "section_name").orderBy(col("sentence_score").desc()))
df_top_n_sentences = (df_top_sentences.withColumn("rank", row_number().over(window_section)).filter(col("rank") <= num_sentences))
df_best_sentences = (df_top_n_sentences.groupBy("original_id", "title", "section_name", "semantic_role").agg(concat_ws(
            " ",
            sort_array(
                collect_list(struct("rank", "key_sentence"))
            )["key_sentence"]
        ).alias("section_summary")
    )
)

# Combining the specific roles into one row for each paper and joining that with the corresponding keywords for that paper
# df_insights = df_best_sentences.groupBy("original_id").pivot("semantic_role").agg(concat_ws(" ", collect_list("key_sentence")))
df_insights = (df_best_sentences.groupBy("original_id").pivot("semantic_role").agg(concat_ws(" ", collect_list("section_summary"))))
df_paper_insights = df_paper_keywords.join(df_insights, on="original_id", how="left")

print("Paper insights schema:")
df_paper_insights.printSchema()

# Saving outputs of all the dataframes created above - detailed keywords, top sentences and paper-level insights
df_keywords.write.mode("overwrite").parquet("output/paper_keywords")
df_top_sentences.write.mode("overwrite").parquet("output/paper_sentences")
df_paper_insights.write.mode("overwrite").parquet("output/paper_insights")
print("Finished extracting insights")

# Cleaning text for readability - removing whitespace, cleaning up math artifacts, etc.
def clean_text_for_display(text, max_width=60):
    if text is None or pd.isna(text):
        return "—"
    text = str(text)
    text = ' '.join(text.split())
    text = text.replace('\\n', ' ')
    text = text.replace('  ', ' ')
    if len(text) > max_width:
        wrapped = textwrap.fill(text, width=max_width)
        return wrapped
    return text

# Formatting the keywords for readability
def format_keywords_list(keywords, max_display=8):
    if keywords is None or (isinstance(keywords, float) and pd.isna(keywords)):
        return "—"
    if isinstance(keywords, list):
        kw_list = keywords[:max_display]
        formatted = ", ".join(kw_list)
        if len(keywords) > max_display:
            formatted += f" (+{len(keywords) - max_display} more)"
        return formatted
    return str(keywords)

# Formatting the section header
def print_section_header(title):
    print("\n")
    print("┌" + "─" * 78 + "┐")
    print("│" + title.center(78) + "│")
    print("└" + "─" * 78 + "┘")

# Formatting the insights
def print_paper_card(row, index):
    print(f"\n{'━' * 80}")
    print(f"Paper #{index + 1}")
    print(f"{'━' * 80}")
    
    # Paper ID
    paper_id = row.get('original_id', 'N/A')
    print(f"ID: {paper_id}")
    
    # Title
    title = row.get('title', 'N/A')
    if title and not pd.isna(title):
        title_clean = clean_text_for_display(str(title), max_width=70)
        print(f"Title: {title_clean}")
    
    # Keywords
    keywords = row.get('all_keywords', None)
    keywords_str = format_keywords_list(keywords)
    print(f"\nKeywords:")
    print(f"      {keywords_str}")
    
    # Printing the insights
    roles = [('OVERVIEW','Overview'),('MOTIVATION','Motivation'),('CONTEXT','Context'),('APPROACH','Approach'),('FINDINGS','Findings'),('TAKEAWAY','Takeaway')]
    
    print(f"\nInsights:")
    for role_key, role_name in roles:
        if role_key in row and row[role_key] and not pd.isna(row[role_key]):
            insight_text = clean_text_for_display(str(row[role_key]), max_width=65)
            lines = insight_text.split('\n')
            print(f"      {role_name}:")
            for line in lines:
                print(f"         {line}")

# Printing a readable keywords table
def print_keywords_table(df_kw, max_rows=15):
    print(f"\n{'─' * 80}")
    print(f"  {'Paper ID':<12} {'Section':<14} {'Role':<12} {'Keyword':<25} {'Score':>8}")
    print(f"{'─' * 80}")
    
    for idx, row in df_kw.head(max_rows).iterrows():
        paper_id = str(row.get('original_id', ''))[:10]
        section = str(row.get('section_name', ''))[:12]
        role = str(row.get('semantic_role', ''))[:10] if row.get('semantic_role') else '—'
        keyword = str(row.get('keyword', ''))[:23]
        score = row.get('keyword_score', 0)
        
        # making a keyword score that easy to understand visually
        score_val = float(score) if score else 0
        score_bar = '█' * int(score_val * 5) + '░' * (5 - int(score_val * 5))
        
        print(f"  {paper_id:<12} {section:<14} {role:<12} {keyword:<25} {score_bar} {score_val:.2f}")
    
    print(f"{'─' * 80}")
    if len(df_kw) > max_rows:
        print(f"  ... and {len(df_kw) - max_rows} more keywords")

# Displaying data
print_section_header("PAPER INSIGHTS SUMMARY:\n")
df_pd = pd.read_parquet("output/paper_insights")
print(f"\n  Total papers with insights: {len(df_pd)}")

# Display top num_papers papers as cards
for idx, row in df_pd.head(num_papers).iterrows():
    print_paper_card(row.to_dict(), idx)

# Printing the extracted keywords
# print_section_header("EXTRACTED KEYWORDS")
# df_kw = pd.read_parquet("output/paper_keywords")
# print(f"\n  Total keyword extractions: {len(df_kw)}")
# print_keywords_table(df_kw, max_rows=15)
