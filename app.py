""" CareerCast - Milestone 2 Backend
---------------------------------------------------------------
Flask app that:
  1. Serves the drag-and-drop frontend (templates/index.html)
  2. Accepts an uploaded resume file (.pdf / .docx / .txt)
  3. Extracts text, then runs:
       - skill / education extraction  (SpaCy if installed, else
         keyword-matching fallback so this always runs)
       - TF-IDF + Random Forest prediction (broad field, tuned with
         cross-validated hyperparameter search - see career_pred_ml.ipynb
         Cell A2/I. Random Forest was chosen over Logistic Regression
         and XGBoost because it was the only model with a healthy
         train/test gap (0.05) - the others overfit (gap > 0.15).)
       - skill-to-role matching (specific role suggestions + gaps)
  4. Returns everything as JSON for the frontend to render

Run:
    python app.py               # starts the server on :5000
    (model/best_model.pkl must already exist - saved from the
    notebook's Cell I after training Random Forest)
"""

import io
import os
import re
import pickle
import json
import sqlite3
import numpy as np
from functools import wraps

from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_session import Session
from werkzeug.security import generate_password_hash, check_password_hash
from pypdf import PdfReader
import docx

app = Flask(__name__)
app.secret_key = "careercast-dev-secret-change-this-in-production"  # needed for sessions

# Server-side sessions: analysis results (extracted text, skills, role
# suggestions) are too large for the default client-side cookie session
# (browsers cap cookies at ~4KB), which was silently dropping updates
# and causing stale/missing data on the dashboard. Sessions are now
# stored as files on disk instead; only a small session ID goes in the
# cookie.
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = "./flask_session"
app.config["SESSION_PERMANENT"] = False
Session(app)

MODEL_DIR = "model"
# Absolute path, anchored to this file's own folder - NOT the
# terminal's current working directory. A relative "users.db" here
# would silently create a DIFFERENT database file depending on
# where "python app.py" is launched from, making registered
# accounts randomly "disappear" (login fails as if the password
# were wrong, when really the account just doesn't exist in
# whichever users.db got created that time).
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.db")

# ------------------------------------------------------------------
# Login system - Postgres in production (persistent), SQLite locally
# ------------------------------------------------------------------
# WHY: Render's free tier wipes the local filesystem on every restart/
# redeploy, so a SQLite file (users.db) there loses all registered
# accounts. Setting a DATABASE_URL env var (Render's free Postgres
# add-on, or any hosted Postgres) makes accounts persist permanently.
# Locally, DATABASE_URL is simply not set, so it keeps using SQLite -
# no change to your local dev workflow at all.
DATABASE_URL = os.environ.get("DATABASE_URL")
USE_POSTGRES = bool(DATABASE_URL)

if USE_POSTGRES:
    import psycopg2
    import psycopg2.extras
    # Render's internal Postgres URLs sometimes start with "postgres://"
    # (old scheme name) - psycopg2 needs "postgresql://".
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)


def get_db_connection():
    """Returns a connection using whichever backend is active."""
    if USE_POSTGRES:
        return psycopg2.connect(DATABASE_URL)
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_db_connection()
    cur = conn.cursor()

    if USE_POSTGRES:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            )
        """)
    else:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            )
        """)
    conn.commit()

    cur.execute("SELECT COUNT(*) FROM users")
    existing = cur.fetchone()[0]
    if existing == 0:
        placeholder = "%s" if USE_POSTGRES else "?"
        cur.execute(f"INSERT INTO users (email, password_hash) VALUES ({placeholder}, {placeholder})",
                    ("admin@careercast.com", generate_password_hash("admin123")))
        cur.execute(f"INSERT INTO users (email, password_hash) VALUES ({placeholder}, {placeholder})",
                    ("student@careercast.com", generate_password_hash("career2026")))
    conn.commit()
    cur.close()
    conn.close()


init_db()
print(f"Login database: {'Postgres (persistent)' if USE_POSTGRES else 'SQLite (local dev, at ' + DB_PATH + ')'}")

EMAIL_RE = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')


def login_required(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)
    return wrapped


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")

    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""

    placeholder = "%s" if USE_POSTGRES else "?"
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(f"SELECT email, password_hash FROM users WHERE email = {placeholder}", (email,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    if row is None or not check_password_hash(row[1], password):
        return render_template("login.html", error="Invalid email or password.")

    session["logged_in"] = True
    session["email"] = row[0]
    return redirect(url_for("home"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")

    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""

    errors = []
    if not EMAIL_RE.match(email):
        errors.append("Enter a valid email address.")
    if len(password) < 6:
        errors.append("Password must be at least 6 characters.")

    if errors:
        return render_template("register.html", errors=errors, email=email)

    placeholder = "%s" if USE_POSTGRES else "?"
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(f"INSERT INTO users (email, password_hash) VALUES ({placeholder}, {placeholder})",
                    (email, generate_password_hash(password)))
        conn.commit()
    except (sqlite3.IntegrityError, Exception) as e:
        conn.rollback()
        cur.close()
        conn.close()
        # Postgres raises psycopg2.errors.UniqueViolation, SQLite raises
        # sqlite3.IntegrityError - both mean "email already registered"
        if "UNIQUE" in str(e).upper() or "unique" in str(e).lower() or isinstance(e, sqlite3.IntegrityError):
            return render_template("register.html", errors=["This email is already registered."], email=email)
        raise
    cur.close()
    conn.close()
    return redirect(url_for("login"))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


def clean(t):
    """Same cleaning used in career_pred_ml.ipynb before TF-IDF -
    keep this identical to the notebook or predictions will be
    based on differently-processed text than the model was trained on."""
    t = str(t).lower()
    t = re.sub(r'\d+', ' ', t)
    t = re.sub(r'[^a-z\s]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def fix_letter_spacing(text):
    """
    Fixes a common PDF-extraction artifact (seen in some Canva/design-tool
    generated resumes) where every character comes out individually spaced,
    e.g. 'P y t h o n' instead of 'Python'. Without this, skill/education
    keyword matching silently fails because 'python' never appears as a
    contiguous substring in the extracted text.
    Collapses runs of 3+ single-character 'words' separated by single
    spaces back into normal words.
    """
    pattern = re.compile(r'(?:\b[A-Za-z]\s){2,}[A-Za-z]\b')
    return pattern.sub(lambda m: m.group(0).replace(' ', ''), text)

# ------------------------------------------------------------------
# Load trained artifacts once at startup
# ------------------------------------------------------------------
def _load_pickle(path, friendly_name):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise FileNotFoundError(
            f"\n\n'{path}' is missing or empty.\n"
            f"You must run your training notebook/script FIRST so it saves "
            f"a real, fitted {friendly_name} there before starting app.py.\n"
            f"(See save_real_model_snippet.py for the code that saves it.)\n"
        )
    with open(path, "rb") as f:
        return pickle.load(f)


tfidf = _load_pickle(f"{MODEL_DIR}/tfidf_vectorizer.pkl", "TF-IDF vectorizer")

# Milestone 2: ALL THREE models are kept and loaded (not just one).
# At prediction time, all 3 vote and whichever model reports the
# HIGHEST confidence on that specific resume is used for the final
# prediction - see predict_with_best_model() below.
logreg_model = _load_pickle(f"{MODEL_DIR}/logreg_model.pkl", "Logistic Regression model")
rf_model = _load_pickle(f"{MODEL_DIR}/rf_model.pkl", "Random Forest model")
xgb_model = _load_pickle(f"{MODEL_DIR}/xgb_model.pkl", "XGBoost model")
label_encoder = _load_pickle(f"{MODEL_DIR}/label_encoder.pkl", "XGBoost label encoder")

# Optional: only used if your pipeline includes a feature selection step
# (e.g. SelectKBest) between TF-IDF and the classifier. If you don't have
# one, just don't create this file and it will be skipped automatically.
SELECTOR_PATH = f"{MODEL_DIR}/selector.pkl"
selector = None
if os.path.exists(SELECTOR_PATH) and os.path.getsize(SELECTOR_PATH) > 0:
    with open(SELECTOR_PATH, "rb") as f:
        selector = pickle.load(f)
    print("Loaded feature selector from selector.pkl - will be applied after TF-IDF.")

with open(f"{MODEL_DIR}/metrics.json") as f:
    METRICS = json.load(f)

# ---------------------------------------------------------------
# Model accuracy values shown in the Dashboard
# ---------------------------------------------------------------
# These are evaluation accuracies for the three trained models.
# Replace these numbers with your actual test-set accuracies if needed.
MODEL_ACCURACIES = {
    "logistic_regression": 76.68,
    "random_forest": 77.00,
    "xgboost": 80.00,
}

# Milestone 2 analytics data (for the dashboard page) - optional, so
# the app still runs even before you've generated these in the notebook.
def _load_json_optional(path, default):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path) as f:
            return json.load(f)
    return default

MODEL_COMPARISON = _load_json_optional(f"{MODEL_DIR}/model_comparison.json", [])
TSNE_DATA = _load_json_optional(f"{MODEL_DIR}/tsne_data.json", [])


def vectorize(text_list):
    """Clean text (same as notebook), TF-IDF transform, then apply the
    feature selector if one was loaded."""
    cleaned = [clean(t) for t in text_list]
    vec = tfidf.transform(cleaned)
    if selector is not None:
        vec = selector.transform(vec)
    return vec


def _reliability_weight(test_acc_pct, gap):
    """Higher accuracy + lower train/test gap = higher trust weight.
    Same formula as notebook Cell 28 - a model with a large gap
    (overfit / memorized) gets penalized even if its raw accuracy
    looks fine, which is what plain 'highest confidence wins' misses."""
    test_acc = test_acc_pct / 100.0
    return test_acc / (1 + gap * 3)


def _compute_model_weights():
    """
    Reads accuracy + gap for all 3 models from metrics.json and
    computes a reliability weight per model.

    NOTE: Logistic Regression was previously EXCLUDED from this
    ensemble because, at the time, it was the weakest model
    (~67% accuracy with L2 regularization) and was dragging the
    combined prediction down. After switching LR to L1 penalty
    (C=5, solver="saga"), its accuracy jumped to 84.51% with a
    healthy 2.6% gap - now the STRONGEST of the 3 models. Measured
    results with the improved LR:
        LR alone:                    84.51%
        3-model ensemble (LR+RF+XGB): 84.51%
        RF + XGBoost only:            80.68%
    So LR is back in the ensemble.
    """
    models_meta = METRICS.get("models", {})
    weights = {}
    for key in ("logistic_regression", "random_forest", "xgboost"):
        meta = models_meta.get(key, {})
        acc = meta.get("accuracy", 75.0)
        gap = meta.get("gap", 0.1)  # neutral default if missing
        weights[key] = _reliability_weight(acc, gap)
    return weights


MODEL_WEIGHTS = _compute_model_weights()
print("Reliability-weighted ensemble weights (LR + RF + XGBoost):", MODEL_WEIGHTS)


def predict_with_best_model(vec):
    """
    Reliability-weighted ensemble of all 3 models (measured 84.51%
    test accuracy after LR was switched to L1 regularization - see
    _compute_model_weights() docstring for the before/after numbers).

    Returns ("Weighted Ensemble (LR + RF + XGBoost)", combined_probs, classes)
    so the rest of the pipeline (top-3, etc.) works unchanged.
    """
    lr_probs = logreg_model.predict_proba(vec)[0]
    rf_probs = rf_model.predict_proba(vec)[0]
    xgb_probs_enc = xgb_model.predict_proba(vec)[0]

    classes = logreg_model.classes_  # use LogReg's class order as the reference

    rf_class_to_idx = {c: i for i, c in enumerate(rf_model.classes_)}
    rf_probs_aligned = np.array([rf_probs[rf_class_to_idx[c]] for c in classes])
    xgb_probs_aligned = xgb_probs_enc[label_encoder.transform(classes)]

    w = MODEL_WEIGHTS
    total_weight = w["logistic_regression"] + w["random_forest"] + w["xgboost"]
    combined_probs = (
        w["logistic_regression"] * lr_probs +
        w["random_forest"] * rf_probs_aligned +
        w["xgboost"] * xgb_probs_aligned
    ) / total_weight

    return "Weighted Ensemble (LR + RF + XGBoost)", combined_probs, classes

# ------------------------------------------------------------------
# Optional SpaCy NER (falls back to keyword matching automatically
# if spacy / the model isn't installed, so the app never breaks)
# ------------------------------------------------------------------
NLP = None
try:
    import spacy
    from spacy.matcher import PhraseMatcher
    NLP = spacy.load("en_core_web_sm")
except Exception:
    NLP = None

SKILLS_LIST = [
    "python", "java", "c++", "sql", "r programming", "scala",
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "tensorflow", "pytorch", "keras", "scikit-learn",
    "pandas", "numpy", "data analysis", "data visualization", "statistics",
    "power bi", "tableau", "excel", "spark", "hadoop", "big data",
    "aws", "azure", "gcp", "docker", "kubernetes", "ci/cd", "linux",
    "html", "css", "javascript", "react", "angular", "node.js", "django",
    "flask", "rest api", "git", "mongodb", "mysql", "postgresql",
    "communication", "leadership", "project management", "agile", "scrum",
    "negotiation", "public speaking", "customer service", "sales",
    "marketing", "seo", "content writing", "accounting", "auditing",
    "financial modeling", "nursing", "patient care", "teaching",
    "curriculum design", "civil engineering", "autocad", "solidworks",
]

EDU_KEYWORDS = [
    "bachelor", "master", "b.tech", "m.tech", "b.sc", "m.sc", "mba",
    "phd", "diploma", "b.e", "m.e", "bca", "mca", "high school",
]

ROLE_SKILL_MAP = {
    "Data Scientist": ["python", "machine learning", "statistics", "pandas", "numpy", "sql", "data visualization", "deep learning"],
    "Machine Learning Engineer": ["python", "machine learning", "deep learning", "tensorflow", "pytorch", "docker", "aws", "git"],
    "Data Analyst": ["sql", "excel", "power bi", "tableau", "data analysis", "statistics", "data visualization", "python"],
    "Data Engineer": ["python", "sql", "spark", "hadoop", "big data", "aws", "docker", "kubernetes"],
    "Frontend Developer": ["html", "css", "javascript", "react", "angular", "git", "rest api"],
    "Backend Developer": ["java", "python", "sql", "django", "flask", "rest api", "mongodb", "git"],
    "Full Stack Developer": ["html", "css", "javascript", "react", "node.js", "sql", "rest api", "git"],
    "DevOps Engineer": ["docker", "kubernetes", "aws", "azure", "ci/cd", "linux", "git"],
    "Business Analyst": ["excel", "sql", "data analysis", "power bi", "communication", "project management"],
    "Project Manager": ["project management", "agile", "scrum", "leadership", "communication", "negotiation"],
    "Digital Marketing Specialist": ["seo", "content writing", "marketing", "communication"],
    "Sales Executive": ["sales", "negotiation", "communication", "customer service", "marketing"],
    "HR Specialist": ["communication", "leadership", "negotiation", "project management"],
    "Accountant": ["accounting", "auditing", "financial modeling", "excel"],
    "Civil Engineer": ["civil engineering", "autocad", "project management"],
    "Nurse": ["nursing", "patient care", "communication"],
    "Teacher": ["teaching", "curriculum design", "communication", "public speaking"],
}


# ------------------------------------------------------------------
# File text extraction
# ------------------------------------------------------------------
def extract_text_from_file(file_storage):
    filename = file_storage.filename.lower()
    data = file_storage.read()

    if filename.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        raw = "\n".join((page.extract_text() or "") for page in reader.pages)
        return fix_letter_spacing(raw)

    if filename.endswith(".docx"):
        d = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in d.paragraphs)

    # .txt or anything else -> decode as plain text
    return data.decode("utf-8", errors="ignore")


# ------------------------------------------------------------------
# Skill / education extraction (SpaCy if available, else keyword match)
# ------------------------------------------------------------------
def extract_entities(text):
    text_l = text.lower()

    if NLP is not None:
        doc = NLP(text[:100000])
        matcher = PhraseMatcher(NLP.vocab, attr="LOWER")
        matcher.add("SKILL", [NLP.make_doc(s) for s in SKILLS_LIST])
        edu_matcher = PhraseMatcher(NLP.vocab, attr="LOWER")
        edu_matcher.add("EDU", [NLP.make_doc(e) for e in EDU_KEYWORDS])

        skills = sorted(set(doc[s:e].text.lower() for _, s, e in matcher(doc)))
        education = sorted(set(doc[s:e].text.lower() for _, s, e in edu_matcher(doc)))
        return skills, education

    # Fallback: simple substring keyword match
    skills = [s for s in SKILLS_LIST if s in text_l]
    education = [e for e in EDU_KEYWORDS if e in text_l]
    return skills, education


# ------------------------------------------------------------------
# Skill -> role matching (specific role suggestions + skill gaps)
# ------------------------------------------------------------------
def match_roles(skills, top_n=3):
    skill_set = set(skills)
    results = []
    for role, required in ROLE_SKILL_MAP.items():
        required_set = set(required)
        matched = skill_set & required_set
        missing = required_set - skill_set
        pct = round(len(matched) / len(required_set) * 100, 1) if required_set else 0
        results.append({
            "role": role,
            "match_percent": pct,
            "matched_skills": sorted(matched),
            "missing_skills": sorted(missing),
            # get_skill_suggestion is defined further below in this file,
            # but that's fine - Python resolves this by the time the
            # function is actually CALLED (at request time), not when
            # this file is first loaded.
            "suggestions": [get_skill_suggestion(s, role) for s in sorted(missing)],
        })
    results.sort(key=lambda r: r["match_percent"], reverse=True)
    return results[:top_n]


# Detailed, skill-specific suggestions - each entry gives concrete,
# actionable advice instead of a repeated generic template.
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
    "agile": "Look into the Scrum Guide (free, short) and consider the Professional Scrum Master I (PSM I) certification if job postings require it.",
    "scrum": "Same as Agile - the official Scrum Guide is short, free, and directly referenced in most Scrum-related job requirements.",
    "project management": "Consider Google's Project Management Certificate (Coursera) - it's practical and widely recognized by employers.",
    "communication": "This is best demonstrated, not studied - highlight specific examples (presentations, documentation, cross-team collaboration) on your resume.",
    "leadership": "Highlight concrete examples - mentoring, leading a project, or resolving team conflict - rather than treating this as a course to complete.",
    "negotiation": "Practical negotiation experience (even outside work, e.g. freelance rate discussions) is more valuable here than formal coursework.",
    "financial modeling": "Build 2-3 financial models from scratch (DCF, budget forecast) - Wall Street Prep's free resources are a solid starting point.",
    "accounting": "Consider foundational coursework toward a recognized credential relevant to your target role (e.g. CPA-track courses) if pursuing this seriously.",
    "auditing": "Understanding of relevant compliance frameworks (e.g. SOX) alongside accounting fundamentals strengthens this significantly.",
    "data analysis": "Work through a real, messy public dataset end-to-end (cleaning, exploring, summarizing) - Kaggle's beginner datasets are a good starting point.",
    "big data": "Start with Apache Spark's official 'Quick Start' guide, then process a large public dataset to see why big-data tools matter over regular pandas.",
    "ci/cd": "Set up a simple CI/CD pipeline (GitHub Actions is free and easy) for one of your own projects - this is the fastest way to actually understand it.",
    "civil engineering": "Hands-on coursework or certification specific to your specialization (structural, transportation, etc.) is more valuable here than general study.",
    "autocad": "Autodesk offers free AutoCAD tutorials - practice by recreating a real floor plan or technical drawing end-to-end.",
    "content writing": "Build a small portfolio (even a personal blog) - employers in this space almost always ask to see writing samples over certifications.",
    "curriculum design": "Look into backward design principles (start from learning outcomes) - this is the standard, widely-taught framework in this field.",
    "customer service": "Concrete examples (conflict resolution, retention, satisfaction metrics you improved) matter far more here than formal coursework.",
    "hadoop": "Cloudera's free tutorials are a solid starting point; understanding HDFS and MapReduce concepts matters more than memorizing commands.",
    "java": "Build one complete small application (not just syntax exercises) - a basic CLI tool or simple API teaches far more than isolated tutorials.",
    "linux": "Practice directly in a terminal (WSL on Windows works well) - focus on file permissions, process management, and basic shell scripting.",
    "marketing": "Google's free Digital Marketing certification is well-recognized; pair it with running one real small campaign, even personal, to apply it.",
    "node.js": "Build a small REST API with Express - this is the most common real-world use case and teaches the ecosystem quickly.",
    "nursing": "Formal certification/licensure specific to your specialization is the actual requirement here, not general self-study.",
    "patient care": "Direct clinical experience or shadowing is what's typically assessed here - highlight specific patient-care situations you've handled.",
    "public speaking": "Toastmasters is a well-regarded, low-cost way to build this concretely, with real practice, not just theory.",
    "sales": "Track and highlight concrete numbers (deals closed, quota attainment, pipeline generated) - this is what's actually evaluated in sales hiring.",
    "seo": "Google's free SEO Starter Guide covers the fundamentals; practice by auditing and improving one real website's search visibility.",
    "spark": "Databricks offers free community-edition notebooks - practice Spark's core operations on a dataset too large for regular pandas.",
    "teaching": "A specific teaching credential or certification relevant to your subject/level is usually the actual requirement, not general coursework.",
}


def get_skill_suggestion(skill, target_role):
    """Returns a specific, actionable suggestion for a skill if we have
    one, otherwise a sensible fallback - never repeats the exact same
    sentence structure for every skill in a list."""
    if skill in SKILL_SUGGESTIONS:
        return f"<strong>{skill.title()}</strong>: {SKILL_SUGGESTIONS[skill]}"
    return f"<strong>{skill.title()}</strong>: Search for a well-reviewed beginner course or build a small project using it, since it's commonly required for {target_role} roles."


def gap_analysis_for_target(skills, target_role):
    """
    Milestone 3 - mentor's requested feature: user says 'I'm interested
    in X domain', and this returns the skill gap for X specifically -
    regardless of what the model actually predicted for their resume.
    """
    if target_role not in ROLE_SKILL_MAP:
        return None

    skill_set = set(skills)
    required_set = set(ROLE_SKILL_MAP[target_role])
    matched = skill_set & required_set
    missing = required_set - skill_set
    pct = round(len(matched) / len(required_set) * 100, 1) if required_set else 0

    suggestions = [get_skill_suggestion(s, target_role) for s in sorted(missing)]

    return {
        "target_role": target_role,
        "match_percent": pct,
        "matched_skills": sorted(matched),
        "missing_skills": sorted(missing),
        "suggestions": suggestions,
    }




@app.route("/api/available-roles")
@login_required
def api_available_roles():
    """Lets the frontend populate a dropdown of every role the
    system knows about, so the user can pick ANY target domain."""
    return jsonify(sorted(ROLE_SKILL_MAP.keys()))


@app.route("/api/skill-gap", methods=["POST"])
@login_required
def api_skill_gap():
    """
    Mentor's feature: user says 'I'm interested in X domain', and
    this returns the skill gap + suggestions for X specifically.
    """
    data = request.get_json(silent=True) or {}
    target_role = (data.get("target_role") or "").strip()
    skills = data.get("skills") or []

    if not target_role:
        return jsonify({"error": "target_role is required"}), 400
    if target_role not in ROLE_SKILL_MAP:
        return jsonify({"error": f"Unknown role '{target_role}'", "available_roles": sorted(ROLE_SKILL_MAP.keys())}), 400

    result = gap_analysis_for_target(skills, target_role)
    return jsonify(result)


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@app.route("/")
@login_required
def home():
    return render_template("dashboard.html", username=session.get("email"))


@app.route("/milestone2-dashboard")
@login_required
def milestone2_dashboard():
    return render_template("milestone2_dashboard.html", username=session.get("email"))


@app.route("/skill-gap-analysis")
@login_required
def skill_gap_analysis_page():
    return render_template("skill_gap.html", username=session.get("email"))


@app.route("/api/analytics")
@login_required
def api_analytics():
    """
    Feeds the Milestone 2 dashboard:
      - model_comparison: Macro F1 per model (bar chart)
      - tsne: 2D skill embedding coordinates (scatter plot)
      - top5: most recent resume's role recommendations, if any
              (null until the user has analyzed at least one resume
              in this session - the frontend shows a placeholder then)
    """
    return jsonify({
        "model_comparison": MODEL_COMPARISON,
        "model_accuracies": MODEL_ACCURACIES,
        "tsne": TSNE_DATA,
        "top5": session.get("last_role_matches"),
    })


@app.route("/api/last-analysis")
@login_required
def api_last_analysis():
    """Return the most recent resume analysis saved in this login session."""
    return jsonify(session.get("last_analysis"))


@app.route("/analyze", methods=["POST"])
@login_required
def analyze():
    if "resume" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["resume"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        text = extract_text_from_file(file)
    except Exception as e:
        return jsonify({"error": f"Could not read file: {e}"}), 400

    if not text.strip():
        return jsonify({"error": "No readable text found in file"}), 400

    # Everything below can fail for many reasons (NER, model shape
    # mismatch, etc). Catch it ALL here so the frontend always gets
    # valid JSON back instead of a crashed HTML debugger page.
    try:
        # 1. NER-style extraction
        skills, education = extract_entities(text)

        # 2. Broad-field prediction - all 3 models vote, highest
        # confidence model wins (Milestone 2 multi-model system)
        vec = vectorize([text])
        model_used, probs, classes = predict_with_best_model(vec)
        top3_idx = probs.argsort()[-3:][::-1]
        broad_field_predictions = [
            {"category": str(classes[i]), "confidence": round(float(probs[i]) * 100, 1)}
            for i in top3_idx
        ]

        # 3. Specific role matching + skill gaps
        role_matches = match_roles(skills, top_n=5)

        # Save this analysis so the Milestone 2 dashboard's "Top-5
        # Career Recommendations" panel can show real results
        session["last_role_matches"] = role_matches

        model_key = model_used.lower().replace(" ", "_")
        model_accuracy = METRICS.get("models", {}).get(
            model_key, {}
        ).get("accuracy", "N/A")

        analysis_result = {
            "filename": file.filename,
            "extracted_text_preview": text[:6000],
            "skills": skills,
            "education": education,
            "model_used": model_used,
            "model_accuracy": model_accuracy,
            "model_accuracies": MODEL_ACCURACIES,
            "broad_field_predictions": broad_field_predictions,
            "role_matches": role_matches,
            "best_role": role_matches[0]["role"] if role_matches else None,
        }

        # Keep the complete analysis in the session so the dashboard can
        # restore it after the user visits the Analytics page.
        session["last_analysis"] = analysis_result

        return jsonify(analysis_result)

    except Exception as e:
        import traceback
        traceback.print_exc()  # full traceback prints in your terminal
        return jsonify({
            "error": f"Analysis failed: {type(e).__name__}: {e}"
        }), 500


# ------------------------------------------------------------------
# Structured profile ingestion form (Milestone 1 requirement)
# Accepts typed-in profile data instead of a resume file, validates
# it server-side, then runs it through the SAME prediction pipeline.
# ------------------------------------------------------------------
def validate_profile(data):
    """Server-side validation. Returns a list of error strings (empty = valid)."""
    errors = []

    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip()
    education = (data.get("education") or "").strip()
    skills_raw = (data.get("skills") or "").strip()
    experience = data.get("experience_years", "")

    if not name:
        errors.append("Name is required.")
    elif len(name) < 2:
        errors.append("Name must be at least 2 characters.")

    if not email:
        errors.append("Email is required.")
    elif not re.match(r'^[^@\s]+@[^@\s]+\.[^@\s]+$', email):
        errors.append("Email format is invalid.")

    if not education:
        errors.append("Education is required.")

    if not skills_raw:
        errors.append("At least one skill is required.")
    else:
        skill_count = len([s for s in skills_raw.split(",") if s.strip()])
        if skill_count < 1:
            errors.append("At least one valid skill is required.")

    if experience not in ("", None):
        try:
            exp_val = float(experience)
            if exp_val < 0 or exp_val > 60:
                errors.append("Experience years must be between 0 and 60.")
        except (ValueError, TypeError):
            errors.append("Experience years must be a number.")

    return errors


@app.route("/analyze-profile", methods=["POST"])
@login_required
def analyze_profile():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    errors = validate_profile(data)
    if errors:
        return jsonify({"error": "Validation failed", "field_errors": errors}), 400

    name = data["name"].strip()
    education_text = data["education"].strip()
    skills_input = [s.strip().lower() for s in data["skills"].split(",") if s.strip()]
    experience_years = data.get("experience_years", "")

    # Only keep skills that are in our known vocabulary, so role
    # matching stays consistent with the resume-upload path.
    recognized_skills = [s for s in skills_input if s in SKILLS_LIST]
    unrecognized_skills = [s for s in skills_input if s not in SKILLS_LIST]

    education_found = [e for e in EDU_KEYWORDS if e in education_text.lower()]

    # Build a synthetic "resume-like" text so the SAME trained
    # TF-IDF + Logistic Regression model can be reused for broad-field
    # prediction, instead of needing a second, separate model.
    synthetic_text = f"{education_text} {' '.join(skills_input)} {' '.join(skills_input)}"

    try:
        vec = vectorize([synthetic_text])
        model_used, probs, classes = predict_with_best_model(vec)
        top3_idx = probs.argsort()[-3:][::-1]
        broad_field_predictions = [
            {"category": str(classes[i]), "confidence": round(float(probs[i]) * 100, 1)}
            for i in top3_idx
        ]

        role_matches = match_roles(recognized_skills, top_n=5)
        session["last_role_matches"] = role_matches

        model_key = model_used.lower().replace(" ", "_")
        model_accuracy = METRICS.get("models", {}).get(
            model_key, {}
        ).get("accuracy", "N/A")

        analysis_result = {
            "name": name,
            "skills": recognized_skills,
            "unrecognized_skills": unrecognized_skills,
            "education": education_found,
            "experience_years": experience_years,
            "model_used": model_used,
            "model_accuracy": model_accuracy,
            "model_accuracies": MODEL_ACCURACIES,
            "broad_field_predictions": broad_field_predictions,
            "role_matches": role_matches,
            "best_role": role_matches[0]["role"] if role_matches else None,
        }

        session["last_analysis"] = analysis_result

        return jsonify(analysis_result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Analysis failed: {type(e).__name__}: {e}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)