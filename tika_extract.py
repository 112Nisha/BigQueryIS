from tika import parser
import tika
import os
import json
import re

tika.initVM()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# This should point to the folder where your downloaded PDFs are
UNSTRUCTURED_DIR = os.path.join(BASE_DIR, "pdfs")

# Use JSONL (one record per line, Spark-friendly)
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "tika_extracted.jsonl")
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

def clean_text(text):
    if not text:
        return ""
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    for filename in os.listdir(UNSTRUCTURED_DIR):
        if not filename.lower().endswith(".pdf"):
            continue

        filepath = os.path.join(UNSTRUCTURED_DIR, filename)

        # paper_id matches your metadata CSV
        paper_id = filename.replace(".pdf", "")

        print(f"Extracting {filename}")

        try:
            parsed = parser.from_file(filepath)

            text = clean_text(parsed.get("content", ""))
            metadata = parsed.get("metadata", {})

            record = {
                # Alignment keys
                "paper_id": paper_id,
                "file_name": filename,

                # Core extracted text
                "full_text": text,

                # Useful PDF metadata
                "num_pages": metadata.get("xmpTPg:NPages"),
                "language": metadata.get("language"),
                "content_type": metadata.get("Content-Type"),
                "created": metadata.get("Creation-Date"),
                "producer": metadata.get("pdf:producer"),

                # Simple text statistics (VERY useful later)
                "char_count": len(text),
                "word_count": len(text.split())
            }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")

        except Exception as e:
            error_record = {
                "paper_id": paper_id,
                "file_name": filename,
                "error": str(e)
            }
            out.write(json.dumps(error_record) + "\n")
            print(f"❌ Failed on {filename}: {e}")

print("✅ Tika extraction complete for all PDFs")
