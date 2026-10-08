"""OCR Marathi scanned PDFs -> kb/raw/<name>.ocr.txt (local, light: 200 DPI, sequential).

Usage: python kb/ocr_marathi.py
Requires: pymupdf (fitz), pytesseract + Tesseract with 'mar' language.
"""
import sys
from pathlib import Path

import fitz
import pytesseract
from PIL import Image

MEDIA = Path("media")
OUT = Path("kb/raw")
TARGETS = ["1. Paddy.pdf", "advisory-marathi.pdf"]
DPI = 200  # enough for tesseract, keeps CPU/thermal low


def ocr_pdf(pdf_path: Path, out_path: Path) -> int:
    doc = fitz.open(pdf_path)
    chunks = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=DPI)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        text = pytesseract.image_to_string(img, lang="mar+eng")
        chunks.append(f"\n===== PAGE {i + 1} =====\n{text}")
        print(f"  page {i + 1}/{doc.page_count} ({len(text)} chars)")
    out_path.write_text("\n".join(chunks), encoding="utf-8")
    return doc.page_count


def main() -> int:
    if "mar" not in pytesseract.get_languages():
        print("FATAL: tesseract 'mar' language missing")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    for name in TARGETS:
        src = MEDIA / name
        if not src.exists():
            print(f"skip (missing): {name}")
            continue
        dst = OUT / (src.stem + ".ocr.txt")
        print(f"OCR {name} -> {dst}")
        n = ocr_pdf(src, dst)
        print(f"done: {n} pages")
    return 0


if __name__ == "__main__":
    sys.exit(main())