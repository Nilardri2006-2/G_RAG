from pdf_loader import load_pdf
from chunker import chunk_pages
import json

PDF_PATH = "data/raw/c1.pdf"
OUTPUT_PATH = "data/processed/chunks.json"


pages = load_pdf(PDF_PATH)

print(f"Pages extracted: {len(pages)}")

chunks = chunk_pages(pages)

print(f"Chunks created: {len(chunks)}")

print("\n--- SAMPLE CHUNK ---\n")

print(chunks[0]["text"])
print("\nPage:", chunks[0]["page"])

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)


print(f"Saved chunks to: {OUTPUT_PATH}")