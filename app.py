"""
app.py
-------
Main Streamlit application for the AI Exam Question Generator.

Workflow:
  Step 1 – Upload a PDF via sidebar.
  Step 2 – Extract text using PyMuPDF.
  Step 3 – Preprocess text with NLTK.
  Step 4 – Train (or load) the NLP model.
  Step 5 – Identify important sentences using the trained model.
  Step 6 – Generate questions (2 / 3 / 12 / 16 marks).
  Step 7 – Generate answers for each question.
  Step 8 – Display and export a formatted question paper as PDF.
"""

import os
import sys
import pickle
import tempfile

import streamlit as st

# ── Make project modules importable ───────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ── Internal imports ──────────────────────────────────────────────
from pdf_processing.extract_text import extract_text_from_pdf
from nlp_processing.preprocess_text import preprocess_corpus, split_into_sentences
from model.train_model import train, predict_importance, MODEL_PATH, DATASET_PATH
from generator.question_generator import generate_questions_from_sentences, group_by_marks
from dataset.generate_dataset import create_dataset
from utils.pdf_exporter import export_question_paper


# ─────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Exam Question Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .main-header h1 {
        color: #e94560;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
    }
    .main-header p {
        color: #a8b2c1;
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
    }
    .step-card {
        background: linear-gradient(135deg, #1e2a3a, #243447);
        border: 1px solid #2d4a6a;
        border-radius: 12px;
        padding: 1rem 1.4rem;
        margin: 0.5rem 0;
    }
    .step-card.done {
        border-color: #00d09c;
    }
    .mark-badge-2  { background: #2563eb; color: #fff; }
    .mark-badge-3  { background: #7c3aed; color: #fff; }
    .mark-badge-12 { background: #d97706; color: #fff; }
    .mark-badge-16 { background: #dc2626; color: #fff; }
    .mark-badge-2, .mark-badge-3, .mark-badge-12, .mark-badge-16 {
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .metric-card {
        background: #1e2a3a;
        border: 1px solid #2d4a6a;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .metric-card .value { font-size: 2rem; font-weight: 700; color: #e94560; }
    .metric-card .label { font-size: 0.8rem; color: #a8b2c1; margin-top: 4px; }
    div[data-testid="stExpander"] {
        border: 1px solid #2d4a6a !important;
        border-radius: 10px !important;
        background: #1a263a !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #e94560, #c0392b);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: transform 0.15s;
    }
    .stButton > button:hover { transform: scale(1.03); }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%) !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📚 AI Exam Question Generator</h1>
    <p>Upload any study PDF → AI extracts content → Generates exam-ready questions &amp; answers locally.</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# Sidebar – settings
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")
    num_questions = st.slider("Total questions to generate", 8, 40, 20, step=4)
    include_2  = st.checkbox("Include 2-Mark questions",  value=True)
    include_3  = st.checkbox("Include 3-Mark questions",  value=True)
    include_12 = st.checkbox("Include 12-Mark questions", value=True)
    include_16 = st.checkbox("Include 16-Mark questions", value=True)
    retrain    = st.checkbox("Force retrain model",       value=False)
    st.markdown("---")
    st.markdown("""
    **📋 How It Works**
    1. Upload a PDF
    2. Text extracted via PyMuPDF
    3. NLTK preprocessing
    4. TF-IDF + ML scoring
    5. Rule-based Q generation
    6. Export as PDF
    """)


# ─────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────
if "questions"      not in st.session_state: st.session_state.questions      = []
if "model_metrics"  not in st.session_state: st.session_state.model_metrics  = {}
if "pdf_data"       not in st.session_state: st.session_state.pdf_data       = None
if "nlp_data"       not in st.session_state: st.session_state.nlp_data       = None
if "export_path"    not in st.session_state: st.session_state.export_path    = None


# ─────────────────────────────────────────────────────────────────
# Step 1: Upload PDF
# ─────────────────────────────────────────────────────────────────
st.markdown("### 📤 Step 1 — Upload Study PDF")
uploaded = st.file_uploader("Choose a PDF file", type=["pdf"], label_visibility="collapsed")


if uploaded:
    # ── Step 2: Extract text ──────────────────────────────────────
    if st.button("🚀 Generate Questions", use_container_width=True):
        progress = st.progress(0, text="Starting …")

        with st.spinner(""):
            # --- Save uploaded file to temp location --------------
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            # Step 2 – Extract text
            progress.progress(10, text="Step 2 – Extracting text from PDF …")
            pdf_data = extract_text_from_pdf(tmp_path)
            st.session_state.pdf_data = pdf_data
            os.unlink(tmp_path)

            # Step 3 – Preprocess
            progress.progress(25, text="Step 3 – NLP preprocessing …")
            sentences = pdf_data["sentences"]
            if not sentences:
                # Fallback: split raw text into sentences
                sentences = split_into_sentences(pdf_data["raw_text"])
            nlp_data = preprocess_corpus(sentences)
            st.session_state.nlp_data = nlp_data

            # Step 4 – Train / load model
            progress.progress(40, text="Step 4 – Training / loading model …")
            model_exists = os.path.exists(MODEL_PATH)
            if retrain or not model_exists:
                # Generate dataset if absent
                if not os.path.exists(DATASET_PATH):
                    create_dataset(DATASET_PATH)
                metrics = train()
                st.session_state.model_metrics = metrics
            else:
                st.session_state.model_metrics = {"note": "Pre-trained model loaded."}

            # Step 5 – Identify important sentences (already done by model)
            progress.progress(60, text="Step 5 – Identifying important sentences …")

            # Step 6 & 7 – Generate questions and answers
            progress.progress(75, text="Step 6/7 – Generating questions & answers …")
            questions = generate_questions_from_sentences(
                sentences, MODEL_PATH, top_n=num_questions
            )
            # Filter by mark preferences
            allowed_marks = []
            if include_2:  allowed_marks.append(2)
            if include_3:  allowed_marks.append(3)
            if include_12: allowed_marks.append(12)
            if include_16: allowed_marks.append(16)
            questions = [q for q in questions if q["mark"] in allowed_marks]
            st.session_state.questions = questions

            progress.progress(90, text="Step 8 – Preparing export …")
            # Export path placeholder (generated on demand)
            st.session_state.export_path = None

            progress.progress(100, text="✅ Done!")

        st.success("Questions generated successfully!")


# ─────────────────────────────────────────────────────────────────
# Results display
# ─────────────────────────────────────────────────────────────────
if st.session_state.questions:
    questions = st.session_state.questions
    grouped   = group_by_marks(questions)

    # ── Metrics row ───────────────────────────────────────────────
    st.markdown("### 📊 Summary")
    cols = st.columns(5)
    total_q = len(questions)
    cols[0].metric("Total Questions", total_q)
    cols[1].metric("2-Mark",  len(grouped.get(2,  [])))
    cols[2].metric("3-Mark",  len(grouped.get(3,  [])))
    cols[3].metric("12-Mark", len(grouped.get(12, [])))
    cols[4].metric("16-Mark", len(grouped.get(16, [])))

    # ── Model metrics ─────────────────────────────────────────────
    if st.session_state.model_metrics:
        with st.expander("🔬 Model Training Metrics", expanded=False):
            m = st.session_state.model_metrics
            if "accuracy" in m:
                c1, c2 = st.columns(2)
                c1.metric("Test Accuracy",  f"{m['accuracy']*100:.2f}%")
                c2.metric("CV Accuracy",    f"{m.get('cv_mean', 0)*100:.2f}% ± {m.get('cv_std', 0)*100:.2f}%")
                if "report" in m:
                    st.code(m["report"], language="text")
            else:
                st.info(m.get("note", "Model loaded from cache."))

    st.markdown("---")
    st.markdown("### 📝 Generated Question Paper")

    # ── Display questions by section ──────────────────────────────
    MARK_INFO = {
        2:  ("Part A — 2-Mark Questions",  "2 Marks"),
        3:  ("Part B — 3-Mark Questions",  "3 Marks"),
        12: ("Part C — 12-Mark Questions", "12 Marks"),
        16: ("Part D — 16-Mark Questions", "16 Marks"),
    }

    for mark in [2, 3, 12, 16]:
        qs = grouped.get(mark, [])
        if not qs:
            continue
        section_title, badge_text = MARK_INFO[mark]
        st.markdown(f"#### {section_title}")
        for i, q in enumerate(qs, 1):
            with st.expander(f"Q{i}. {q['question']}  [{badge_text}]", expanded=(mark == 2)):
                col_q, col_a = st.columns([1, 2])
                with col_q:
                    st.markdown("**❓ Question**")
                    st.info(q["question"])
                    st.caption(f"Model confidence: {q['confidence']:.1%}")
                with col_a:
                    st.markdown("**✅ Answer**")
                    st.success(q["answer"])

    # ── Export as PDF ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📄 Export Question Paper")

    subject_name = st.text_input("Subject / Exam name (for PDF header)",
                                 value=uploaded.name.replace(".pdf", "") if uploaded else "Subject")

    if st.button("📥 Export as PDF", use_container_width=True):
        export_path = export_question_paper(
            questions=questions,
            subject=subject_name,
            output_dir=os.path.join(BASE_DIR, "exports")
        )
        st.session_state.export_path = export_path

    if st.session_state.export_path and os.path.exists(st.session_state.export_path):
        with open(st.session_state.export_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Question Paper PDF",
                data=f.read(),
                file_name=os.path.basename(st.session_state.export_path),
                mime="application/pdf",
                use_container_width=True,
            )
        st.success(f"PDF saved → `{st.session_state.export_path}`")

elif uploaded is None:
    # Landing illustration
    st.markdown("""
    <div style="text-align:center; padding: 3rem 0; color:#a8b2c1">
        <div style="font-size:5rem">📖</div>
        <h3 style="color:#e94560; margin-top:1rem">Ready to generate exam questions?</h3>
        <p>Upload a PDF from the area above and click <strong>Generate Questions</strong>.</p>
        <p style="font-size:0.85rem; margin-top:2rem">
            ✅ Runs 100% locally &nbsp;|&nbsp; ✅ No internet required &nbsp;|&nbsp;
            ✅ No API keys needed
        </p>
    </div>
    """, unsafe_allow_html=True)
