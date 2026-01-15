# CONVERTING THE JSON INTO A CSV

# import json
# import csv

# input_file = "/home/dell/Downloads/archive/arxiv-metadata-oai-snapshot.json"
# output_file = "arxiv_metadata.csv"

# fields = [
#     "id",
#     "title",
#     "authors",
#     "categories",
#     "abstract",
#     "journal-ref",
#     "doi"
# ]

# max_rows = 500
# row_count = 0

# with open(input_file, "r", encoding="utf-8") as fin, \
#      open(output_file, "w", encoding="utf-8", newline="") as fout:

#     writer = csv.DictWriter(fout, fieldnames=fields)
#     writer.writeheader()

#     for line in fin:
#         if row_count >= max_rows:
#             break

#         record = json.loads(line)
#         writer.writerow({f: record.get(f, "") for f in fields})
#         row_count += 1

# print(f"Successfully written {row_count} rows to {output_file}")


# DOWNLOADING THE PDFS

# import json
# import subprocess
# import os
# import re

# JSON_FILE = "/home/dell/Downloads/archive/arxiv-metadata-oai-snapshot.json"
# OUTPUT_DIR = "pdfs"
# MAX_PAPERS = 500

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# def get_latest_pdf_path(paper_id):
#     # NEW-style ID
#     if "/" not in paper_id:
#         year = paper_id.split(".")[0]
#         base_path = f"gs://arxiv-dataset/arxiv/pdf/{year}/"
#         pattern = f"{paper_id}v"
#     # OLD-style ID
#     else:
#         archive, number = paper_id.split("/")
#         base_path = f"gs://arxiv-dataset/arxiv/pdf/{archive}/"
#         pattern = f"{number}v"

#     result = subprocess.run(
#         ["gsutil", "ls", f"{base_path}{pattern}*.pdf"],
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE,
#         text=True
#     )

#     if result.returncode != 0 or not result.stdout.strip():
#         return None

#     files = result.stdout.strip().split("\n")

#     # Pick highest version number
#     def version_num(path):
#         m = re.search(r"v(\d+)\.pdf$", path)
#         return int(m.group(1)) if m else 0

#     return max(files, key=version_num)

# count = 0
# downloaded = 0

# with open(JSON_FILE, "r", encoding="utf-8") as f:
#     for line in f:
#         if count >= MAX_PAPERS:
#             break

#         record = json.loads(line)
#         paper_id = record["id"]

#         latest_pdf = get_latest_pdf_path(paper_id)
#         if not latest_pdf:
#             print(f"❌ No PDF found for {paper_id}")
#             count += 1
#             continue

#         local_name = paper_id.replace("/", "_") + ".pdf"
#         local_path = os.path.join(OUTPUT_DIR, local_name)

#         print(f"⬇️  Downloading {latest_pdf}")

#         subprocess.run(["gsutil", "cp", latest_pdf, local_path])

#         downloaded += 1
#         count += 1

# print(f"\n✅ Downloaded {downloaded} PDFs out of {count}")


# REPLACING NON EXISTING PDF ROWS

import os
import pandas as pd

CSV_FILE = "arxiv_metadata.csv"
PDF_DIR = "pdfs"
OUTPUT_FILE = "arxiv_metadata_replaced.csv"

# Collect PDF-backed IDs
pdf_ids = {
    f.replace(".pdf", "").replace("_", "/")
    for f in os.listdir(PDF_DIR)
    if f.lower().endswith(".pdf")
}

def normalize_id(x):
    """
    Canonical arXiv ID normalization:
    - pad left side to 4 digits (LEFT padding)
    - pad right side to 4 digits (RIGHT padding)
    """
    x = str(x)

    if "." not in x:
        return x

    left, right = x.split(".", 1)

    # pad left side (YYMM)
    left = left.zfill(4)

    # pad right side (sequence number)
    right = right.ljust(4, "0")

    return f"{left}.{right}"



def replace_and_print(x):
    if x in pdf_ids:
        return x
    else:
        print("Missing PDF for id:", x)
        return "MISSING_PDF"

df = pd.read_csv(CSV_FILE)


# force id to string
df["id"] = df["id"].astype(str)

# normalize ids
df["id"] = df["id"].apply(normalize_id)

# replace + print missing
df["id"] = df["id"].apply(replace_and_print)

df.to_csv(OUTPUT_FILE, index=False)

print("Done.")






