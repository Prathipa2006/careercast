"""
CareerCast - Milestone 4 Streamlit Application
------------------------------------------------
Features:
1. Resume upload and text extraction
2. Career prediction using trained pipeline
3. Career probability visualization
4. Skill extraction
5. Skill-gap analysis
6. Career comparison
7. CSV and TXT report export
8. Cohort analytics

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
from pathlib import Path
from datetime import datetime


# ================================================================
# PAGE CONFIGURATION
# ================================================================

st.set_page_config(
    page_title="CareerCast",
    page_icon="🎯",
    layout="wide"
)


MODEL_DIR = "model"

ANALYTICS_FILE = Path("analytics_data.csv")

ANALYTICS_COLUMNS = [
    "timestamp",
    "predicted_category",
    "confidence",
    "skills"
]


# ================================================================
# TEXT CLEANING
# ================================================================

def clean_batch(texts):
    """
    Same cleaning function used when the model pipeline was created.
    """
    def clean_one(t):
        t = str(t).lower()
        t = re.sub(r'\d+', ' ', t)
        t = re.sub(r'[^a-z\s]', ' ', t)
        return re.sub(r'\s+', ' ', t).strip()

    return [clean_one(t) for t in texts]


# ================================================================
# LOAD MODEL
# ================================================================

@st.cache_resource
def load_pipeline():

    pipeline_path = os.path.join(
        MODEL_DIR,
        "pipeline.pkl"
    )

    if not os.path.exists(pipeline_path):
        st.error(
            f"Could not find {pipeline_path}. "
            "Make sure pipeline.pkl exists inside the model folder."
        )
        st.stop()

    with open(pipeline_path, "rb") as f:
        return pickle.load(f)


pipeline = load_pipeline()


# ================================================================
# LOAD METRICS
# ================================================================

metrics_path = os.path.join(
    MODEL_DIR,
    "metrics.json"
)

if not os.path.exists(metrics_path):

    st.error(
        "metrics.json was not found inside the model folder."
    )

    st.stop()


with open(metrics_path) as f:
    metrics = json.load(f)


# ================================================================
# ANALYTICS FUNCTIONS
# ================================================================

def save_analytics(
    predicted_category,
    confidence,
    skills
):
    """
    Store anonymous aggregate analytics.

    Resume text, name, email and uploaded files
    are NOT stored.
    """

    row = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(),
        "predicted_category": predicted_category,
        "confidence": float(confidence),
        "skills": ", ".join(skills)
    }])

    if ANALYTICS_FILE.exists():

        row.to_csv(
            ANALYTICS_FILE,
            mode="a",
            header=False,
            index=False
        )

    else:

        row.to_csv(
            ANALYTICS_FILE,
            index=False
        )


def load_analytics():

    if not ANALYTICS_FILE.exists():

        return pd.DataFrame(
            columns=ANALYTICS_COLUMNS
        )

    try:

        return pd.read_csv(
            ANALYTICS_FILE
        )

    except Exception:

        return pd.DataFrame(
            columns=ANALYTICS_COLUMNS
        )


# ================================================================
# CAREER / SKILL DATA
# ================================================================

ROLE_SKILL_MAP = {

    "Data Scientist": [
        "python",
        "machine learning",
        "statistics",
        "pandas",
        "numpy",
        "sql",
        "data visualization",
        "deep learning"
    ],

    "Machine Learning Engineer": [
        "python",
        "machine learning",
        "deep learning",
        "tensorflow",
        "pytorch",
        "docker",
        "aws",
        "git"
    ],

    "Data Analyst": [
        "sql",
        "excel",
        "power bi",
        "tableau",
        "data analysis",
        "statistics",
        "data visualization",
        "python"
    ],

    "Data Engineer": [
        "python",
        "sql",
        "spark",
        "hadoop",
        "big data",
        "aws",
        "docker",
        "kubernetes"
    ],

    "Frontend Developer": [
        "html",
        "css",
        "javascript",
        "react",
        "angular",
        "git",
        "rest api"
    ],

    "Backend Developer": [
        "java",
        "python",
        "sql",
        "django",
        "flask",
        "rest api",
        "mongodb",
        "git"
    ],

    "Full Stack Developer": [
        "html",
        "css",
        "javascript",
        "react",
        "node.js",
        "sql",
        "rest api",
        "git"
    ],

    "DevOps Engineer": [
        "docker",
        "kubernetes",
        "aws",
        "azure",
        "ci/cd",
        "linux",
        "git"
    ],

    "Business Analyst": [
        "excel",
        "sql",
        "data analysis",
        "power bi",
        "communication",
        "project management"
    ],

    "Project Manager": [
        "project management",
        "agile",
        "scrum",
        "leadership",
        "communication",
        "negotiation"
    ],

    "Digital Marketing Specialist": [
        "seo",
        "content writing",
        "marketing",
        "communication"
    ],

    "Sales Executive": [
        "sales",
        "negotiation",
        "communication",
        "customer service",
        "marketing"
    ],

    "HR Specialist": [
        "communication",
        "leadership",
        "negotiation",
        "project management"
    ],

    "Accountant": [
        "accounting",
        "auditing",
        "financial modeling",
        "excel"
    ],

    "Civil Engineer": [
        "civil engineering",
        "autocad",
        "project management"
    ],

    "Nurse": [
        "nursing",
        "patient care",
        "communication"
    ],

    "Teacher": [
        "teaching",
        "curriculum design",
        "communication",
        "public speaking"
    ]
}


# ================================================================
# SKILL SUGGESTIONS
# ================================================================

SKILL_SUGGESTIONS = {

    "python":
        "Build 2-3 small projects using Python.",

    "sql":
        "Practice joins, subqueries, window functions and SQL problems.",

    "machine learning":
        "Build classification and regression projects using real datasets.",

    "deep learning":
        "Build one image or NLP project using TensorFlow or PyTorch.",

    "tensorflow":
        "Follow TensorFlow beginner tutorials and build an ML project.",

    "pytorch":
        "Complete a beginner PyTorch tutorial and build a small model.",

    "docker":
        "Containerize one of your existing projects.",

    "kubernetes":
        "Learn Kubernetes fundamentals after becoming comfortable with Docker.",

    "aws":
        "Learn AWS fundamentals and deploy a small application.",

    "git":
        "Practice Git using a real GitHub repository.",

    "pandas":
        "Clean and analyze a real-world dataset using Pandas.",

    "numpy":
        "Practice NumPy arrays, broadcasting and vectorized operations.",

    "statistics":
        "Study probability, hypothesis testing and descriptive statistics.",

    "data visualization":
        "Create dashboards and charts using real datasets.",

    "power bi":
        "Build a Power BI dashboard using a public dataset.",

    "tableau":
        "Create a Tableau dashboard using Tableau Public.",

    "excel":
        "Practice Pivot Tables, XLOOKUP and data analysis.",

    "html":
        "Build a complete static webpage using HTML.",

    "css":
        "Practice responsive layouts and modern CSS.",

    "javascript":
        "Build a small interactive JavaScript application.",

    "react":
        "Build a small React application using components and state.",

    "django":
        "Build a small Django web application.",

    "flask":
        "Build a small REST API using Flask.",

    "mongodb":
        "Practice MongoDB CRUD operations and schema design.",

    "rest api":
        "Build an API and consume it from another application.",

    "linux":
        "Practice Linux commands, permissions and process management.",

    "project management":
        "Learn Agile and Scrum and practice project planning.",

    "communication":
        "Improve communication through presentations and teamwork.",

    "leadership":
        "Take responsibility for leading a small project or team.",

    "public speaking":
        "Practice presentations and public speaking regularly."
}


def get_skill_suggestion(
    skill,
    target_role
):

    if skill in SKILL_SUGGESTIONS:

        return (
            f"**{skill.title()}**: "
            f"{SKILL_SUGGESTIONS[skill]}"
        )

    return (
        f"**{skill.title()}**: "
        f"Learn this skill through a course "
        f"and build a small project related to "
        f"{target_role}."
    )


# ================================================================
# PDF / DOCX / TXT EXTRACTION
# ================================================================

def fix_letter_spacing(text):

    pattern = re.compile(
        r'(?:\b[A-Za-z]\s){2,}[A-Za-z]\b'
    )

    return pattern.sub(
        lambda m: m.group(0).replace(" ", ""),
        text
    )


def extract_text_from_upload(
    uploaded_file
):

    name = uploaded_file.name.lower()

    data = uploaded_file.read()

    # PDF
    if name.endswith(".pdf"):

        from pypdf import PdfReader
        import io

        reader = PdfReader(
            io.BytesIO(data)
        )

        raw = "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

        return fix_letter_spacing(raw)

    # DOCX
    if name.endswith(".docx"):

        import docx
        import io

        document = docx.Document(
            io.BytesIO(data)
        )

        return "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
        )

    # TXT
    return data.decode(
        "utf-8",
        errors="ignore"
    )


# ================================================================
# SKILL EXTRACTION
# ================================================================

ALL_SKILLS = sorted(
    set(
        skill
        for skills in ROLE_SKILL_MAP.values()
        for skill in skills
    )
)


def extract_skills_simple(text):

    text_lower = text.lower()

    return [
        skill
        for skill in ALL_SKILLS
        if skill in text_lower
    ]


# ================================================================
# SIDEBAR
# ================================================================

st.sidebar.title("🎯 CareerCast")

st.sidebar.write(
    "Milestone 4 Application"
)

st.sidebar.divider()

rf_accuracy = (
    metrics
    .get("models", {})
    .get("random_forest", {})
    .get("accuracy", "N/A")
)

st.sidebar.metric(
    "Random Forest Accuracy",
    f"{rf_accuracy}%"
)

st.sidebar.write(
    f"Model Categories: "
    f"{len(metrics.get('categories', []))}"
)


# ================================================================
# MAIN TITLE
# ================================================================

st.title("🎯 CareerCast")

st.caption(
    "AI-powered Career Prediction, "
    "Skill Gap Analysis & Career Comparison"
)


# ================================================================
# RESUME ANALYSIS
# ================================================================

st.header("📄 Analyze Your Resume")

input_mode = st.radio(
    "Choose input method",
    [
        "Upload file",
        "Paste text"
    ],
    horizontal=True
)


resume_text = ""


if input_mode == "Upload file":

    uploaded_file = st.file_uploader(
        "Upload your resume",
        type=[
            "pdf",
            "docx",
            "txt"
        ]
    )

    if uploaded_file is not None:

        try:

            resume_text = extract_text_from_upload(
                uploaded_file
            )

            word_count = len(
                resume_text.split()
            )

            st.success(
                f"Resume loaded successfully — "
                f"{word_count} words detected."
            )

            with st.expander(
                "View extracted resume text"
            ):

                st.text_area(
                    "Extracted text",
                    resume_text,
                    height=250,
                    disabled=True
                )

        except Exception as e:

            st.error(
                f"Could not read the file: {e}"
            )


else:

    resume_text = st.text_area(
        "Paste your resume text here",
        height=250
    )


# ================================================================
# PREDICTION
# ================================================================

if st.button(
    "🔮 Predict Career",
    type="primary"
):

    if not resume_text.strip():

        st.warning(
            "Please upload a resume or paste resume text first."
        )

    else:

        try:

            # Model prediction
            probabilities = pipeline.predict_proba(
                [resume_text]
            )[0]

            classes = pipeline.classes_

            prediction_df = pd.DataFrame({
                "Category": classes,
                "Probability": probabilities * 100
            })

            prediction_df = prediction_df.sort_values(
                "Probability",
                ascending=False
            ).reset_index(drop=True)

            # Extract skills
            extracted_skills = extract_skills_simple(
                resume_text
            )

            # Save in session
            st.session_state[
                "prediction_df"
            ] = prediction_df

            st.session_state[
                "extracted_skills"
            ] = extracted_skills

            # Save cohort analytics
            best_category = prediction_df.iloc[0][
                "Category"
            ]

            best_confidence = prediction_df.iloc[0][
                "Probability"
            ]

            save_analytics(
                best_category,
                best_confidence,
                extracted_skills
            )

            st.success(
                "Career prediction completed successfully!"
            )

        except Exception as e:

            st.error(
                f"Prediction failed: {type(e).__name__}: {e}"
            )


# ================================================================
# PREDICTION RESULTS
# ================================================================

if "prediction_df" in st.session_state:

    prediction_df = st.session_state[
        "prediction_df"
    ]

    top_prediction = prediction_df.iloc[0]

    st.divider()

    st.header(
        "🎯 Career Prediction Results"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Best Career Match",
            top_prediction["Category"]
        )

    with col2:

        st.metric(
            "Confidence",
            f"{top_prediction['Probability']:.1f}%"
        )

    st.subheader(
        "Career Probability Distribution"
    )

    number_to_show = st.slider(
        "Number of categories",
        min_value=5,
        max_value=len(prediction_df),
        value=min(10, len(prediction_df))
    )

    chart_height = max(
        400,
        number_to_show * 30
    )

    chart = px.bar(
        prediction_df.head(number_to_show),
        x="Probability",
        y="Category",
        orientation="h",
        height=chart_height,
        title="Career Prediction Probabilities"
    )

    st.plotly_chart(
        chart,
        use_container_width=True
    )

    # ------------------------------------------------------------
    # Detected skills
    # ------------------------------------------------------------

    st.subheader(
        "🛠️ Detected Skills"
    )

    detected_skills = st.session_state.get(
        "extracted_skills",
        []
    )

    if detected_skills:

        st.write(
            ", ".join(
                skill.title()
                for skill in detected_skills
            )
        )

    else:

        st.info(
            "No matching skills were detected."
        )

    # ------------------------------------------------------------
    # Export
    # ------------------------------------------------------------

    st.subheader(
        "📥 Export Prediction Report"
    )

    csv_data = prediction_df.to_csv(
        index=False
    )

    st.download_button(
        "Download CSV",
        data=csv_data,
        file_name="career_prediction_report.csv",
        mime="text/csv"
    )

    report_text = f"""
CareerCast Prediction Report
============================

Best Career Match:
{top_prediction['Category']}

Confidence:
{top_prediction['Probability']:.1f}%

Detected Skills:
{', '.join(detected_skills)}

Career Predictions:
{prediction_df.to_string(index=False)}
"""

    st.download_button(
        "Download TXT",
        data=report_text,
        file_name="career_prediction_report.txt",
        mime="text/plain"
    )


# ================================================================
# SKILL GAP ANALYSIS
# ================================================================

st.divider()

st.header(
    "🎯 Career Skill Gap Analysis"
)


if "extracted_skills" not in st.session_state:

    st.info(
        "Analyze a resume first to perform skill-gap analysis."
    )

else:

    user_skills = st.session_state[
        "extracted_skills"
    ]

    st.write(
        "**Your detected skills:** "
        + (
            ", ".join(user_skills)
            if user_skills
            else "None"
        )
    )

    target_role = st.selectbox(
        "Select your target career",
        list(ROLE_SKILL_MAP.keys()),
        key="skill_gap_role"
    )

    if st.button(
        "🔍 Check Skill Gap"
    ):

        required = set(
            ROLE_SKILL_MAP[target_role]
        )

        available = set(
            user_skills
        )

        matched = available & required

        missing = required - available

        match_percentage = round(
            len(matched)
            / len(required)
            * 100,
            1
        ) if required else 0

        st.metric(
            "Skill Match",
            f"{match_percentage}%"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.success(
                "✅ Skills You Have"
            )

            if matched:

                for skill in sorted(matched):

                    st.write(
                        f"• {skill.title()}"
                    )

            else:

                st.write(
                    "No matching skills yet."
                )

        with col2:

            st.warning(
                "📚 Skills To Add"
            )

            if missing:

                for skill in sorted(missing):

                    st.write(
                        f"• {skill.title()}"
                    )

            else:

                st.success(
                    "You already have all core skills!"
                )

        if missing:

            st.subheader(
                "📖 How To Close Your Skill Gap"
            )

            for skill in sorted(missing):

                st.markdown(
                    f"- {get_skill_suggestion(skill, target_role)}"
                )


# ================================================================
# CAREER COMPARISON
# ================================================================

st.divider()

st.header(
    "Career Comparison"
)

st.caption(
    "Compare the skills required by two different careers."
)


comparison_col1, comparison_col2 = st.columns(2)


with comparison_col1:

    career_1 = st.selectbox(
        "Career 1",
        list(ROLE_SKILL_MAP.keys()),
        index=0,
        key="career_1"
    )


with comparison_col2:

    career_2 = st.selectbox(
        "Career 2",
        list(ROLE_SKILL_MAP.keys()),
        index=1,
        key="career_2"
    )


if career_1 == career_2:

    st.warning(
        "Please select two different careers."
    )

else:

    skills_1 = set(
        ROLE_SKILL_MAP[career_1]
    )

    skills_2 = set(
        ROLE_SKILL_MAP[career_2]
    )

    common_skills = skills_1 & skills_2

    only_career_1 = skills_1 - skills_2

    only_career_2 = skills_2 - skills_1

    col1, col2, col3 = st.columns(3)

    with col1:

        st.subheader(
            " Common Skills"
        )

        if common_skills:

            for skill in sorted(common_skills):

                st.write(
                    f"• {skill.title()}"
                )

        else:

            st.write(
                "No common skills."
            )

    with col2:

        st.subheader(
            f"🔵 {career_1}"
        )

        st.write(
            "Skills specific to this career:"
        )

        if only_career_1:

            for skill in sorted(only_career_1):

                st.write(
                    f"• {skill.title()}"
                )

        else:

            st.write(
                "No additional skills."
            )

    with col3:

        st.subheader(
            f"🟢 {career_2}"
        )

        st.write(
            "Skills specific to this career:"
        )

        if only_career_2:

            for skill in sorted(only_career_2):

                st.write(
                    f"• {skill.title()}"
                )

        else:

            st.write(
                "No additional skills."
            )


# ================================================================
# COHORT ANALYTICS
# ================================================================

st.divider()

st.header(
    "📊 Cohort Analytics"
)

st.caption(
    "Aggregate statistics from resumes analyzed by CareerCast."
)


analytics_df = load_analytics()


if analytics_df.empty:

    st.info(
        "No cohort data available yet. "
        "Analyze a few resumes to populate this dashboard."
    )

else:

    # ------------------------------------------------------------
    # Summary metrics
    # ------------------------------------------------------------

    total_resumes = len(
        analytics_df
    )

    average_confidence = analytics_df[
        "confidence"
    ].mean()

    most_common_category = (
        analytics_df[
            "predicted_category"
        ]
        .value_counts()
        .idxmax()
    )

    all_skills = []

    for skills in analytics_df[
        "skills"
    ].dropna():

        if skills.strip():

            all_skills.extend(
                [
                    s.strip().lower()
                    for s in skills.split(",")
                    if s.strip()
                ]
            )

    if all_skills:

        most_common_skill = (
            pd.Series(all_skills)
            .value_counts()
            .idxmax()
        )

    else:

        most_common_skill = "N/A"

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Resumes Analyzed",
            total_resumes
        )

    with col2:

        st.metric(
            "Average Confidence",
            f"{average_confidence:.1f}%"
        )

    with col3:

        st.metric(
            "Most Common Career",
            most_common_category
        )

    with col4:

        st.metric(
            "Most Common Skill",
            most_common_skill.title()
        )

    # ------------------------------------------------------------
    # Career distribution
    # ------------------------------------------------------------

    st.subheader(
        "🎯 Career Distribution"
    )

    career_counts = (
        analytics_df[
            "predicted_category"
        ]
        .value_counts()
        .reset_index()
    )

    career_counts.columns = [
        "Career",
        "Count"
    ]

    career_chart = px.bar(
        career_counts,
        x="Career",
        y="Count",
        title="Predicted Career Categories"
    )

    st.plotly_chart(
        career_chart,
        use_container_width=True
    )

    # ------------------------------------------------------------
    # Common skills
    # ------------------------------------------------------------

    st.subheader(
        "🛠️ Most Common Skills"
    )

    if all_skills:

        skill_counts = (
            pd.Series(all_skills)
            .value_counts()
            .head(15)
            .reset_index()
        )

        skill_counts.columns = [
            "Skill",
            "Count"
        ]

        skill_chart = px.bar(
            skill_counts,
            x="Count",
            y="Skill",
            orientation="h",
            title="Top 15 Skills Across Analyzed Resumes"
        )

        st.plotly_chart(
            skill_chart,
            use_container_width=True
        )

    # ------------------------------------------------------------
    # Analytics table
    # ------------------------------------------------------------

    with st.expander(
        "View Cohort Data"
    ):

        display_df = analytics_df.copy()

        if "timestamp" in display_df.columns:

            display_df["timestamp"] = pd.to_datetime(
                display_df["timestamp"],
                errors="coerce"
            ).dt.strftime(
                "%Y-%m-%d %H:%M"
            )

        st.dataframe(
            display_df,
            use_container_width=True
        )


# ================================================================
# FOOTER
# ================================================================
st.divider()
st.caption(
    "CareerCast | Milestone 4 - Packaging, Testing & Finalization"
)   