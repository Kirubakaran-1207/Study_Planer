"""
utils/pdf_exporter.py
----------------------
Exports the generated question paper to a formatted PDF file.

Uses fpdf2 to create a multi-section, styled document containing:
  - Header with subject name and date
  - Sections: Part A (2-mark), Part B (3-mark), Part C (12-mark), Part D (16-mark)
  - Each question followed by its answer in a styled box
  - Footer with page numbers
"""

import os
import sys
from datetime import date
from fpdf import FPDF

# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

MARK_SECTIONS = {
    2:  "Part A — 2-Mark Questions",
    3:  "Part B — 3-Mark Questions",
    12: "Part C — 12-Mark Questions",
    16: "Part D — 16-Mark Questions",
}

MARK_COLORS = {
    2:  (37,  99,  235),   # blue
    3:  (124, 58,  237),   # purple
    12: (217, 119, 6),     # amber
    16: (220, 38,  38),    # red
}


def _safe_text(text: str) -> str:
    """Replace non-latin-1 characters to avoid fpdf encoding errors."""
    return text.encode("latin-1", errors="replace").decode("latin-1")


# ─────────────────────────────────────────────────────────────────
# PDF class
# ─────────────────────────────────────────────────────────────────

class QuestionPaperPDF(FPDF):
    """Custom FPDF class with header and footer."""

    def __init__(self, subject: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subject = subject

    def header(self):
        # Background bar
        self.set_fill_color(26, 26, 46)
        self.rect(0, 0, 210, 22, "F")
        self.set_text_color(233, 69, 96)
        self.set_font("Helvetica", "B", 14)
        self.set_y(6)
        self.cell(0, 10, _safe_text(f"AI Exam Question Paper — {self.subject}"), align="C")
        self.set_text_color(180, 180, 180)
        self.set_font("Helvetica", "", 8)
        self.set_y(14)
        self.cell(0, 6, f"Generated: {date.today().strftime('%B %d, %Y')}  |  100% Local AI", align="C")
        self.ln(10)

    def footer(self):
        self.set_y(-14)
        self.set_fill_color(26, 26, 46)
        self.rect(0, self.get_y(), 210, 20, "F")
        self.set_text_color(150, 150, 150)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()} / {{nb}}", align="C")


# ─────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────

def export_question_paper(
    questions: list[dict],
    subject:   str = "Examination",
    output_dir: str = "exports",
) -> str:
    """
    Render questions to a formatted PDF.

    Parameters
    ----------
    questions  : list of dicts with keys mark, question, answer
    subject    : string used in the PDF title
    output_dir : directory where the PDF will be saved

    Returns
    -------
    str – absolute path to the saved PDF
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir,
        f"question_paper_{subject.replace(' ', '_')[:30]}.pdf"
    )

    pdf = QuestionPaperPDF(subject=subject, orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.set_margins(15, 30, 15)

    # Group questions by mark band
    from generator.question_generator import group_by_marks
    grouped = group_by_marks(questions)

    for mark in [2, 3, 12, 16]:
        qs = grouped.get(mark, [])
        if not qs:
            continue

        r, g, b = MARK_COLORS[mark]
        section_title = MARK_SECTIONS[mark]

        # ── Section heading ────────────────────────────────────────
        pdf.set_fill_color(r, g, b)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 9, _safe_text(f"  {section_title}"), fill=True, ln=True)
        pdf.ln(3)

        for i, q in enumerate(qs, 1):
            # ── Question ──────────────────────────────────────────
            pdf.set_text_color(30, 30, 30)
            pdf.set_font("Helvetica", "B", 10)
            q_text = _safe_text(f"Q{i}. [{mark} Marks]  {q['question']}")
            pdf.multi_cell(0, 6, q_text)
            pdf.ln(1)

            # ── Answer box ────────────────────────────────────────
            pdf.set_fill_color(240, 248, 255)
            pdf.set_text_color(50, 50, 100)
            pdf.set_font("Helvetica", "", 9)
            ans_text = _safe_text("Ans: " + q["answer"])
            pdf.multi_cell(0, 5.5, ans_text, fill=True, border=1)
            pdf.ln(4)

        pdf.ln(4)

    pdf.output(output_path)
    print(f"[Exporter] PDF saved → {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_qs = [
        {"mark": 2,  "question": "What is TCP/IP?",              "answer": "TCP/IP is the fundamental protocol of the Internet."},
        {"mark": 3,  "question": "Explain DNS briefly.",          "answer": "DNS translates domain names into IP addresses."},
        {"mark": 12, "question": "Explain OSI Model in detail.",  "answer": "The OSI model has 7 layers…"},
        {"mark": 16, "question": "Discuss network architecture.", "answer": "Network architecture refers to…"},
    ]
    path = export_question_paper(sample_qs, subject="Networking Test")
    print(f"Output: {path}")
