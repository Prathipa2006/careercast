"""
CareerCast - Milestone 3 Streamlit Review UI
---------------------------------------------------------------
Uses the single unified pipeline.pkl (cleaning + TF-IDF + feature
selection + classifier bundled together) - no more juggling 3
separate files that can drift out of sync.

Install:
    pip install streamlit plotly pandas

Run:
    streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import json
import re
import os

st.set_page_config(page_title="CareerCast Review UI", layout="wide")

MODEL_DIR = "model"


# Must exist here, identically, so pickle can find it when loading
# pipeline.pkl - the FunctionTransformer inside it references this
# function by name, not by copying its code.
def clean_batch(texts):
    def clean_one(t):
        t = str(t).lower()
        t = re.sub(r'\d+', ' ', t)
        t = re.sub(r'[^a-z\s]', ' ', t)
        return re.sub(r'\s+', ' ', t).strip()
    return [clean_one(t) for t in texts]


@st.cache_resource
def load_pipeline():
    with open(f"{MODEL_DIR}/pipeline.pkl", "rb") as f:
        return pickle.load(f)


# --- DEBUG: hidden behind a toggle, off by default (clean for demo) ---
show_debug = st.sidebar.checkbox("🔍 Show debug info", value=False)

if show_debug:
    st.sidebar.header("Debug Info")
    st.sidebar.code(f"Working directory:\n{os.getcwd()}", language=None)
    st.sidebar.code(f"Looking for model files in:\n{os.path.abspath(MODEL_DIR)}", language=None)

    for fname in ["pipeline.pkl", "metrics.json"]:
        path = os.path.join(MODEL_DIR, fname)
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        st.sidebar.text(f"{'✅' if exists else '❌'} {fname} ({size} bytes)")

st.title("🎯 CareerCast Review UI")
st.caption("Milestone 3 - internal review dashboard for model predictions")

pipeline = load_pipeline()

with open(f"{MODEL_DIR}/metrics.json") as f:
    metrics = json.load(f)

if show_debug:
    st.sidebar.subheader("Raw metrics.json content")
    st.sidebar.json(metrics)


# --- Sidebar: model info ---
st.sidebar.header("Model Info")
rf_accuracy = metrics.get("models", {}).get("random_forest", {}).get("accuracy", "N/A")
st.sidebar.metric("Test Accuracy (Random Forest)", f"{rf_accuracy}%")
st.sidebar.write(f"Categories: {len(metrics.get('categories', []))}")

# Same role-skill reference as your Flask app - copy from app.py if
# you've expanded this list there.
ROLE_SKILL_MAP = {
    "Data Scientist": ["python", "machine learning", "statistics", "pandas", "numpy", "sql", "data visualization", "deep learning"],
    "Machine Learning Engineer": ["python", "machine learning", "deep learning", "tensorflow", "pytorch", "docker", "aws", "git"],
    "Data Analyst": ["sql", "excel", "power bi", "tableau", "data analysis", "statistics", "data visualization", "python"],
    "Frontend Developer": ["html", "css", "javascript", "react", "angular", "git", "rest api"],
    "Backend Developer": ["java", "python", "sql", "django", "flask", "rest api", "mongodb", "git"],
    "DevOps Engineer": ["docker", "kubernetes", "aws", "azure", "ci/cd", "linux", "git"],
    "Business Analyst": ["excel", "sql", "data analysis", "power bi", "communication", "project management"],
}
SKILLS_LIST = sorted({s for skills in ROLE_SKILL_MAP.values() for s in skills})


def fix_letter_spacing(text):
    """
    Fixes a common PDF-extraction artifact (seen in Canva/design-tool
    generated resumes) where every character comes out individually
    spaced, e.g. 'D a t a  A n a l y s i s' instead of 'Data Analysis'.
    Without this, skill keyword matching silently fails.
    """
    pattern = re.compile(r'(?:\b[A-Za-z]\s){2,}[A-Za-z]\b')
    return pattern.sub(lambda m: m.group(0).replace(' ', ''), text)


def extract_text_from_upload(uploaded_file):
    """Same extraction approach as your Flask app.py"""
    name = uploaded_file.name.lower()
    data = uploaded_file.read()

    if name.endswith(".pdf"):
        from pypdf import PdfReader
        import io
        reader = PdfReader(io.BytesIO(data))
        raw = "\n".join((page.extract_text() or "") for page in reader.pages)
        return fix_letter_spacing(raw)

    if name.endswith(".docx"):
        import docx
        import io
        d = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in d.paragraphs)

    return data.decode("utf-8", errors="ignore")


def extract_skills_simple(text):
    text_l = text.lower()
    return [s for s in SKILLS_LIST if s in text_l]


# --- Main: resume input (FILE UPLOAD, not just paste) ---
st.header("📄 Analyze a Resume")

input_mode = st.radio("Input method", ["Upload file", "Paste text"], horizontal=True)

resume_text = ""
if input_mode == "Upload file":
    uploaded_file = st.file_uploader("Upload resume", type=["pdf", "docx", "txt"])
    if uploaded_file is not None:
        resume_text = extract_text_from_upload(uploaded_file)
        word_count = len(resume_text.split())
        st.caption(f"Extracted {word_count} words from the file")
        with st.expander("View extracted text"):
            st.text_area("Full extracted text", resume_text, height=300, disabled=True)
else:
    resume_text = st.text_area("Paste resume text here", height=200)

if st.button("Predict", type="primary") and resume_text.strip():
    # pipeline handles cleaning + TF-IDF + selection + prediction
    # all in one call - just pass the RAW text, don't pre-clean it
    probs = pipeline.predict_proba([resume_text])[0]
    classes = pipeline.classes_  # exposed automatically from the final classifier step

    df = pd.DataFrame({"Category": classes, "Probability": probs * 100})
    df = df.sort_values("Probability", ascending=False)  # ALL categories, not just top 10

    # Save everything needed to redraw this section into session_state,
    # so it survives Streamlit re-running the whole script when you
    # interact with the skill-gap dropdown/button further down.
    st.session_state["prediction_df"] = df
    st.session_state["extracted_skills"] = extract_skills_simple(resume_text)

# Render the prediction results from session_state (NOT only inside
# the button click above) - this is what keeps it visible even after
# interacting with widgets elsewhere on the page.
if "prediction_df" in st.session_state:
    df = st.session_state["prediction_df"]
    top_pred = df.iloc[0]

    st.subheader("Career Probability Distribution")
    num_to_show = st.slider("Number of categories to show", min_value=5, max_value=len(df), value=len(df))
    fig = px.bar(df.head(num_to_show), x="Probability", y="Category", orientation="h",
                 color="Probability", color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)

    st.success(f"**Best match: {top_pred['Category']}** ({top_pred['Probability']:.1f}% confidence)")

    # --- Report export ---
    st.subheader("📥 Export Report")
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download report as CSV",
        data=csv,
        file_name="career_prediction_report.csv",
        mime="text/csv",
    )

    report_text = f"""CareerCast Prediction Report
================================
Best Match: {top_pred['Category']}
Confidence: {top_pred['Probability']:.1f}%

All Predictions:
{df.to_string(index=False)}
"""
    st.download_button(
        label="Download report as TXT",
        data=report_text,
        file_name="career_prediction_report.txt",
        mime="text/plain",
    )

# ----------------------------------------------------------------
# Interested in a Specific Career? (mentor's requested feature)
# ----------------------------------------------------------------
st.divider()
st.header("🎯 Interested in a Specific Career?")

if "extracted_skills" not in st.session_state:
    st.info("Analyze a resume above first, then pick a target career here.")
else:
    user_skills = st.session_state["extracted_skills"]
    st.write(f"Your detected skills: **{', '.join(user_skills) if user_skills else 'none found'}**")

    target_role = st.selectbox("Select a career you're interested in", list(ROLE_SKILL_MAP.keys()))

    if st.button("Check My Skill Gap"):
        required = set(ROLE_SKILL_MAP[target_role])
        have = set(user_skills)
        matched = have & required
        missing = required - have
        pct = round(len(matched) / len(required) * 100, 1) if required else 0

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Skill Match", f"{pct}%")
            st.success(f"✓ Skills you have: {', '.join(sorted(matched)) if matched else 'None yet'}")
        with col2:
            st.warning(f"+ Skills to add: {', '.join(sorted(missing)) if missing else 'None - you have them all!'}")