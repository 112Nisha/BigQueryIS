from tika import parser
import os
import json
import tika

tika.initVM()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UNSTRUCTURED_DIR = f"{BASE_DIR}/data/unstructured"
OUTPUT_FILE = f"{BASE_DIR}/data/extracted_text.json"

extracted = []

for filename in os.listdir(UNSTRUCTURED_DIR):
    filepath = os.path.join(UNSTRUCTURED_DIR, filename)
    parsed = parser.from_file(filepath)

    extracted.append({
        "file_name": filename,
        "text": parsed.get("content", ""),
        "metadata": parsed.get("metadata", {})
    })

with open(OUTPUT_FILE, "w") as f:
    json.dump(extracted, f, indent=2)

print("Tika extraction complete - Extracted info from unstructured data")
