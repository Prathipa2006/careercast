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

# Detailed, skill-specific suggestions - same reference as app.py.
# Streamlit's st.markdown() renders **bold** natively, so markdown
# syntax is correct here (unlike the Flask frontend, which needed HTML).
SKILL_SUGGESTIONS = {
    "python": "Build 2-3 small projects (e.g. a data pipeline or automation script) - Python is best learned by writing real code, not just tutorials.",
    "sql": "Practice on real datasets via sites like LeetCode SQL or HackerRank - focus on joins, window functions, and query optimization.",
    "machine learning": "Complete Andrew Ng's Machine Learning course (Coursera) and implement 2-3 models from scratch (regression, classification, clustering).",
    "deep learning": "Work through the fast.ai course, then build one project using a real dataset (image classification or NLP) to solidify the concepts.",
    "tensorflow": "Complete TensorFlow's official 'Get Started' tutorials, then rebuild one of your scikit-learn models using TensorFlow to compare workflows.",
    "pytorch": "PyTorch's official 60-minute blitz tutorial is the fastest on-ramp - follow it with one small end-to-end training project.",
    "docker": "Containerize one of your own existing projects - this teaches Docker faster than any course, since you'll hit real, practical issues.",
    "kubernetes": "Start with Docker first if you haven't already, then try Kubernetes' official 'Kubernetes Basics' interactive tutorial.",
    "aws": "Get the AWS Cloud Practitioner certification - it's the standard, recognized entry point and covers exactly what most job postings expect.",
    "azure": "Microsoft's free Azure Fundamentals (AZ-900) learning path is the standard starting certification for this.",
    "git": "Practice with a real repo - fork an open-source project, make a change, and submit a pull request to learn the full real-world workflow.",
    "pandas": "Work through Kaggle's free 'Pandas' micro-course, then clean and analyze a messy real-world dataset end-to-end.",
    "data analysis": "Pick a real public dataset (Kaggle has thousands) and answer 3-5 specific business questions with it - this is what's actually assessed, not just knowing definitions.",
    "numpy": "Focus on array broadcasting and vectorized operations - these are what actually show up in technical interviews and real code.",
    "statistics": "Khan Academy's Statistics course covers the fundamentals well; pair it with applying hypothesis testing on a real dataset.",
    "data visualization": "Recreate 3-5 charts from real news/data journalism (e.g. FiveThirtyEight) using matplotlib or Tableau to build a practical eye for good visuals.",
    "power bi": "Microsoft's free Power BI learning path plus building one real dashboard from a public dataset will cover most job requirements.",
    "tableau": "Tableau Public is free - recreate a dashboard from Tableau's own public gallery to learn both the tool and design conventions.",
    "excel": "Focus on pivot tables, VLOOKUP/XLOOKUP, and basic macros - these are what's actually tested in most job screenings.",
    "html": "Build one full static webpage from scratch without a framework - this cements the fundamentals before moving to React/Angular.",
    "css": "Try a CSS-focused challenge site like Frontend Mentor - it forces you to solve real layout problems, not just memorize syntax.",
    "javascript": "freeCodeCamp's JavaScript course is thorough and free; follow it with a small interactive project (to-do app, calculator).",
    "react": "Build one small app (not a tutorial clone) - a habit tracker or notes app is enough to learn components, state, and props properly.",
    "angular": "Angular's official 'Tour of Heroes' tutorial is the standard, well-structured starting point.",
    "django": "Django's official tutorial (the 'polls app') covers the core concepts well; follow with your own small project to reinforce it.",
    "flask": "Flask is lightweight - the official quickstart guide plus building one small API is usually enough to become comfortable.",
    "mongodb": "MongoDB University offers free official courses - M001 covers the essentials needed for most job requirements.",
    "mysql": "Practice schema design and complex joins on a real dataset - this is what's actually assessed in technical screens.",
    "rest api": "Build a small API from scratch (even with Flask/FastAPI) and consume it from a separate script - this teaches both sides of REST.",
    "ci/cd": "Set up a simple pipeline for one of your own repos using GitHub Actions - a working example teaches this faster than theory.",
    "linux": "Practice basic shell navigation, permissions, and process management directly in a terminal - most of this is muscle memory from use.",
    "agile": "Look into the Scrum Guide (free, short) and consider the Professional Scrum Master I (PSM I) certification if job postings require it.",
    "scrum": "Same as Agile - the official Scrum Guide is short, free, and directly referenced in most Scrum-related job requirements.",
    "project management": "Consider Google's Project Management Certificate (Coursera) - it's practical and widely recognized by employers.",
    "communication": "This is best demonstrated, not studied - highlight specific examples (presentations, documentation, cross-team collaboration) on your resume.",
}


def get_skill_suggestion(skill, target_role):
    """Returns a specific, actionable suggestion for a skill if we have
    one, otherwise a sensible fallback - never repeats the exact same
    sentence structure for every skill in a list."""
    if skill in SKILL_SUGGESTIONS:
        return f"**{skill.title()}**: {SKILL_SUGGESTIONS[skill]}"
    return f"**{skill.title()}**: Look for a well-reviewed beginner course or build a small project using it - it's commonly required for {target_role} roles."


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

    # Scale chart height with the number of bars - without this, more
    # bars get squeezed into the same fixed space and start overlapping
    # or getting visually clipped, which LOOKS like missing categories
    # even though the underlying data is complete.
    chart_height = max(350, num_to_show * 28)

    fig = px.bar(df.head(num_to_show), x="Probability", y="Category", orientation="h",
                 color="Probability", color_continuous_scale="Blues", height=chart_height)
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

        st.metric("Skill Match", f"{pct}%")

        col1, col2 = st.columns(2)
        with col1:
            st.success(f"✓ Skills you have: {', '.join(sorted(matched)) if matched else 'None yet'}")
        with col2:
            if missing:
                st.warning(f"+ {len(missing)} skill(s) to add - see details below")
            else:
                st.success("You already have all core skills for this role!")

        # --- Detailed, per-skill "how to learn" suggestions ---
        if missing:
            st.subheader("How to close this gap")
            for skill in sorted(missing):
                st.markdown(f"- {get_skill_suggestion(skill, target_role)}")
