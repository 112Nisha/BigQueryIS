import json
import csv

input_file = "arxiv-data.json"
output_file = "arxiv_metadata.csv"

fields = [
    "id",
    "title",
    "authors",
    "categories",
    "abstract",
    "journal-ref",
    "doi"
]

max_rows = 500
row_count = 0

with open(input_file, "r", encoding="utf-8") as fin, \
     open(output_file, "w", encoding="utf-8", newline="") as fout:

    writer = csv.DictWriter(fout, fieldnames=fields)
    writer.writeheader()

    for line in fin:
        if row_count >= max_rows:
            break

        record = json.loads(line)
        writer.writerow({f: record.get(f, "") for f in fields})
        row_count += 1

print(f"Successfully written {row_count} rows to {output_file}")
