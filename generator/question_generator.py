"""
generator/question_generator.py
---------------------------------
Generates exam-style questions and answers from extracted PDF sentences.

Question types by mark value:
  2-mark  : "What is ___?" / "Define ___."
  3-mark  : "Explain ___ briefly." / "Write short notes on ___."
  12-mark : "Explain ___ in detail." / "Describe the working of ___ with an example."
  16-mark : "Discuss the architecture of ___." / "Analyse ___ and explain its components."

Strategy:
  1. Use the trained model to score each sentence for question-worthiness.
  2. Pick the top-N sentences.
  3. Extract the primary subject (keyword) from each sentence.
  4. Apply rule-based templates per mark band.
  5. Pair each question with the original sentence as the answer.
"""

import os
import sys
import random
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nlp_processing.preprocess_text import extract_keywords, preprocess_sentence
from model.train_model import load_model, prepare_text

# ── Rule-based question templates ────────────────────────────────
TEMPLATES = {
    2: [
        "What is {subject}?",
        "Define {subject}.",
        "State the meaning of {subject}.",
        "What do you mean by {subject}?",
    ],
    3: [
        "Explain {subject} briefly.",
        "Write short notes on {subject}.",
        "Describe {subject} in a few sentences.",
        "Give a brief account of {subject}.",
    ],
    12: [
        "Explain {subject} in detail.",
        "Describe the working of {subject} with an example.",
        "Elaborate on {subject} with suitable diagrams and examples.",
        "Discuss {subject} and its significance in {context_hint}.",
    ],
    16: [
        "Discuss the architecture of {subject} in detail.",
        "Analyse {subject} and explain its components with examples.",
        "With a neat diagram, explain the concept of {subject} in detail.",
        "Critically examine {subject} and compare with related concepts.",
    ],
}


def _extract_subject(sentence: str) -> str:
    """
    Extract the primary subject/noun phrase from a sentence.

    Heuristic: use the most frequent meaningful keyword.
    Falls back to the first two non-stopword tokens if no keyword found.
    """
    keywords = extract_keywords(sentence, top_n=3)
    if keywords:
        # Use title-case for the keyword phrase
        subject = " ".join(keywords[:2]).title()
        return subject

    # Fallback: first two alpha tokens
    tokens = preprocess_sentence(sentence)
    return " ".join(tokens[:2]).title() if tokens else "this concept"


def _infer_context_hint(sentence: str) -> str:
    """
    Return a broad domain hint for use in 12-mark template slot {context_hint}.
    Matches simple domain keywords in the sentence.
    """
    domains = {
        "network": "computer networks",
        "database": "database management",
        "algorithm": "algorithm design",
        "operating": "operating systems",
        "machine": "machine learning",
        "neural": "deep learning",
        "sort": "algorithm design",
        "memory": "computer architecture",
        "security": "information security",
    }
    low = sentence.lower()
    for kw, domain in domains.items():
        if kw in low:
            return domain
    return "computer science"


def generate_questions_from_sentences(
    sentences: list[str],
    model_path: str,
    top_n: int = 20,
) -> list[dict]:
    """
    Given a list of sentences extracted from a PDF:

    1. Score each sentence using the trained model.
    2. Select the top-N highest-confidence sentences.
    3. Generate questions across 2-mark, 3-mark, 12-mark, 16-mark bands.
    4. Return a list of question-answer dicts.

    Parameters
    ----------
    sentences : list[str]
        Sentences from the PDF.
    model_path : str
        Path to the saved sklearn pipeline.
    top_n : int
        Maximum total questions to generate.

    Returns
    -------
    list of dicts with keys:
        mark      : int   (2, 3, 12, or 16)
        question  : str
        answer    : str
        sentence  : str   (source sentence)
        confidence: float (model probability)
    """
    if not sentences:
        return []

    # ── Score sentences with the trained model ────────────────────
    pipeline    = load_model(model_path)
    cleaned     = [prepare_text(s) for s in sentences]
    proba       = pipeline.predict_proba(cleaned)[:, 1]   # P(question-worthy)

    scored = sorted(
        zip(proba, sentences),
        key=lambda x: x[0],
        reverse=True
    )

    # Keep only sentences the model considers question-worthy (p > 0.4)
    worthy = [(p, s) for p, s in scored if p > 0.40]
    if not worthy:
        worthy = scored[:min(top_n, len(scored))]   # fallback: top sentences

    # ── Assign mark bands ─────────────────────────────────────────
    #   Distribute across 2, 3, 12, 16 marks proportionally:
    #   40 % → 2-mark, 30 % → 3-mark, 20 % → 12-mark, 10 % → 16-mark
    n_total  = min(top_n, len(worthy))
    n2  = max(1, int(n_total * 0.40))
    n3  = max(1, int(n_total * 0.30))
    n12 = max(1, int(n_total * 0.20))
    n16 = max(1, n_total - n2 - n3 - n12)

    results: list[dict] = []

    def _add_questions(pool, mark, count):
        used = set()
        added = 0
        for conf, sent in pool:
            if added >= count:
                break
            if sent in used:
                continue
            subject      = _extract_subject(sent)
            ctx_hint     = _infer_context_hint(sent)
            template     = random.choice(TEMPLATES[mark])
            question     = template.format(subject=subject, context_hint=ctx_hint)
            answer       = _build_answer(sent, mark)
            results.append({
                "mark":       mark,
                "question":   question,
                "answer":     answer,
                "sentence":   sent,
                "confidence": round(conf, 3),
            })
            used.add(sent)
            added += 1

    # Slice pools for each band
    all_pool = worthy[:n_total]
    _add_questions(all_pool[:n2],             2,  n2)
    _add_questions(all_pool[n2:n2+n3],        3,  n3)
    _add_questions(all_pool[n2+n3:n2+n3+n12],12, n12)
    _add_questions(all_pool[n2+n3+n12:],     16, n16)

    print(f"[Generator] Generated {len(results)} questions from "
          f"{len(sentences)} sentences (top-{n_total} used).")
    return results


def _build_answer(sentence: str, mark: int) -> str:
    """
    Build an answer string scaled to the mark band.

    2-mark  : the original sentence as-is.
    3-mark  : original sentence + elaboration cue.
    12-mark : original sentence + extended explanation prompt.
    16-mark : original sentence + full analytical prompt.
    """
    if mark == 2:
        return sentence

    if mark == 3:
        return (
            f"{sentence} "
            f"This concept is fundamental to the subject and can be understood "
            f"through its core properties and practical applications."
        )

    if mark == 12:
        return (
            f"{sentence}\n\n"
            f"Detailed Explanation:\n"
            f"The above statement forms the basis of this topic. "
            f"To explain in detail, one must consider the underlying principles, "
            f"the historical context, key components, and real-world applications. "
            f"Diagrams and examples should be included to illustrate the concept "
            f"clearly. Important subtopics include the definition, working mechanism, "
            f"advantages, disadvantages, and use-cases."
        )

    # mark == 16
    return (
        f"{sentence}\n\n"
        f"Comprehensive Discussion:\n"
        f"This topic requires an in-depth analysis covering: "
        f"(1) Definition and introduction, "
        f"(2) Architectural overview with diagram, "
        f"(3) Step-by-step working mechanism, "
        f"(4) Variants and related concepts, "
        f"(5) Advantages and limitations, "
        f"(6) Comparison with alternatives, "
        f"(7) Real-world applications and case studies, "
        f"(8) Conclusion summarising key points."
    )


# ─────────────────────────────────────────────────────────────────
# Convenience function used by app.py
# ─────────────────────────────────────────────────────────────────

def group_by_marks(questions: list[dict]) -> dict[int, list[dict]]:
    """Group a flat question list by mark band."""
    groups: dict[int, list[dict]] = {2: [], 3: [], 12: [], 16: []}
    for q in questions:
        groups.setdefault(q["mark"], []).append(q)
    return groups
