import os
import subprocess

SCAN_DIR = "scans"
JSON_FILE = "cases.json"

for file in os.listdir(SCAN_DIR):
    if file.endswith(".pdf"):
        pdf_path = os.path.join(SCAN_DIR, file)

        print(f"Processing {file}...")

        subprocess.run([
            "python", "pipeline_ocr.py",
            "--pdf", pdf_path,
            "--json", JSON_FILE
        ])