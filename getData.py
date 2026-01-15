import json
import subprocess
import os
import re

JSON_FILE = "arxiv-metadata-oai-snapshot.json"
OUTPUT_DIR = "pdfs"
MAX_PAPERS = 500

os.makedirs(OUTPUT_DIR, exist_ok=True)

VERSION_RE = re.compile(r"v(\d+)\.pdf$")

dir_cache = {}

def list_gcs_dir(path):
    if path in dir_cache:
        return dir_cache[path]

    result = subprocess.run(
        ["gsutil", "ls", path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        dir_cache[path] = []
    else:
        dir_cache[path] = result.stdout.strip().split("\n")

    return dir_cache[path]

def get_latest_pdf_path(paper_id):
    if "/" not in paper_id:
        year = paper_id.split(".")[0]
        base_path = f"gs://arxiv-dataset/arxiv/pdf/{year}/"
        prefix = paper_id
    else:
        archive, number = paper_id.split("/")
        base_path = f"gs://arxiv-dataset/arxiv/pdf/{archive}/"
        prefix = number

    files = list_gcs_dir(base_path)

    best = None
    best_v = -1

    for f in files:
        if f"{prefix}v" in f:
            m = VERSION_RE.search(f)
            if m:
                v = int(m.group(1))
                if v > best_v:
                    best_v = v
                    best = f

    return best

downloaded = 0
checked = 0

with open(JSON_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if downloaded >= MAX_PAPERS:
            break

        record = json.loads(line)
        paper_id = record["id"]
        checked += 1

        latest_pdf = get_latest_pdf_path(paper_id)
        if not latest_pdf:
            print(f"❌ No PDF for {paper_id}")
            continue

        local_name = paper_id.replace("/", "_") + ".pdf"
        local_path = os.path.join(OUTPUT_DIR, local_name)

        if os.path.exists(local_path):
            downloaded += 1
            continue

        print(f"⬇️  Downloading {latest_pdf}")

        subprocess.run(
            ["gsutil", "-m", "cp", latest_pdf, local_path],
            check=False
        )

        if os.path.exists(local_path):
            downloaded += 1

print(f"\n✅ Downloaded {downloaded} PDFs (checked {checked} records)")
