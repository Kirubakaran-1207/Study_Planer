"""
pdf_processing/extract_text.py
--------------------------------
Extracts and structures text from a PDF file using PyMuPDF (fitz).

Responsibilities:
  - Open a PDF and iterate over every page.
  - Extract raw text per page.
  - Split text into paragraphs (double-newline separated).
  - Split paragraphs further into individual sentences.
  - Return a clean, structured dict for downstream processing.
"""

import re
import fitz  # PyMuPDF


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _clean_line(text: str) -> str:
    """Remove hyphenated line-breaks and collapse whitespace."""
    text = re.sub(r"-\n", "", text)          # remove hyphenation at line-end
    text = re.sub(r"\n", " ", text)          # flatten intra-paragraph newlines
    text = re.sub(r"\s{2,}", " ", text)      # collapse multiple spaces
    return text.strip()


def _split_sentences(paragraph: str) -> list[str]:
    """
    Naive sentence splitter.
    Splits on period / exclamation / question mark followed by
    one or more spaces and an uppercase letter.
    """
    # Keep the delimiter attached to the preceding token
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z])", paragraph)
    return [s.strip() for s in raw if len(s.strip()) > 20]


# ─────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> dict:
    """
    Open a PDF and extract text organised by page, paragraph and sentence.

    Parameters
    ----------
    pdf_path : str
        Absolute or relative path to the PDF file.

    Returns
    -------
    dict with keys:
        "raw_text"   : full concatenated text (str)
        "pages"      : list of per-page text strings
        "paragraphs" : list of paragraph strings (across all pages)
        "sentences"  : list of individual sentence strings
    """
    doc = fitz.open(pdf_path)

    all_pages: list[str] = []
    all_paragraphs: list[str] = []
    all_sentences: list[str] = []

    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text("text")          # plain text extraction

        # --- Paragraphs: split on blank lines -------------------------
        raw_paragraphs = re.split(r"\n\s*\n", page_text)

        page_clean_parts: list[str] = []
        for para in raw_paragraphs:
            para = _clean_line(para)
            if len(para) < 30:                     # skip very short fragments
                continue
            all_paragraphs.append(para)
            page_clean_parts.append(para)

            # --- Sentences: split each paragraph ----------------------
            for sentence in _split_sentences(para):
                all_sentences.append(sentence)

        all_pages.append("\n".join(page_clean_parts))

    doc.close()

    raw_text = "\n\n".join(all_pages)

    print(f"[PDF] Extracted {len(all_pages)} pages, "
          f"{len(all_paragraphs)} paragraphs, "
          f"{len(all_sentences)} sentences from '{pdf_path}'")

    return {
        "raw_text": raw_text,
        "pages": all_pages,
        "paragraphs": all_paragraphs,
        "sentences": all_sentences,
    }


# ─────────────────────────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "sample.pdf"
    result = extract_text_from_pdf(path)
    print("\nFirst 5 sentences:")
    for s in result["sentences"][:5]:
        print(" •", s)
