"""
CareerCast - Milestone 1 Backend
---------------------------------------------------------------
Flask app that:
  1. Serves the drag-and-drop frontend (templates/index.html)
  2. Accepts an uploaded resume file (.pdf / .docx / .txt)
  3. Extracts text, then runs:
       - skill / education extraction  (SpaCy if installed, else
         keyword-matching fallback so this always runs)
       - TF-IDF + Logistic Regression prediction (broad field,
         trained by train_model.py)
       - skill-to-role matching (specific role suggestions + gaps)
  4. Returns everything as JSON for the frontend to render

Run:
    python train_model.py      # once, to create model/ artifacts
    python app.py               # starts the server on :5000
"""

import io
import os
import re
import pickle
import json
import sqlite3
from functools import wraps

from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from pypdf import PdfReader
import docx

app = Flask(__name__)
app.secret_key = "careercast-dev-secret-change-this-in-production"  # needed for sessions

MODEL_DIR = "model"
DB_PATH = "users.db"

# ------------------------------------------------------------------
# Login system - SQLite database with hashed passwords (email-based)
# ------------------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    """)
    # Seed two demo accounts, only if the table is empty, so restarting
    # the app doesn't reset custom accounts you register later.
    existing = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    if existing == 0:
        conn.execute("INSERT INTO users (email, password_hash) VALUES (?, ?)",
                     ("admin@careercast.com", generate_password_hash("admin123")))
        conn.execute("INSERT INTO users (email, password_hash) VALUES (?, ?)",
                     ("student@careercast.com", generate_password_hash("career2026")))
    conn.commit()
    conn.close()


init_db()

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

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()

    if user is None or not check_password_hash(user["password_hash"], password):
        return render_template("login.html", error="Invalid email or password.")

    session["logged_in"] = True
    session["email"] = user["email"]
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

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("INSERT INTO users (email, password_hash) VALUES (?, ?)",
                     (email, generate_password_hash(password)))
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return render_template("register.html", errors=["This email is already registered."], email=email)
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
logreg = _load_pickle(f"{MODEL_DIR}/logreg_model.pkl", "Logistic Regression model")

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

OVERALL_ACCURACY = METRICS["accuracy"]


def vectorize(text_list):
    """Clean text (same as notebook), TF-IDF transform, then apply the
    feature selector if one was loaded."""
    cleaned = [clean(t) for t in text_list]
    vec = tfidf.transform(cleaned)
    if selector is not None:
        vec = selector.transform(vec)
    return vec

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
    "python", "java", "c++", "sql", "r programming", "scala","c"
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
        })
    results.sort(key=lambda r: r["match_percent"], reverse=True)
    return results[:top_n]


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@app.route("/")
@login_required
def home():
    return render_template("dashboard.html", username=session.get("email"))


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

        # 2. Broad-field prediction (trained Logistic Regression)
        vec = vectorize([text])
        probs = logreg.predict_proba(vec)[0]
        classes = logreg.classes_
        top3_idx = probs.argsort()[-3:][::-1]
        broad_field_predictions = [
            {"category": str(classes[i]), "confidence": round(float(probs[i]) * 100, 1)}
            for i in top3_idx
        ]

        # 3. Specific role matching + skill gaps
        role_matches = match_roles(skills, top_n=3)

        return jsonify({
            "filename": file.filename,
            "extracted_text_preview": text[:6000],  # generous cap, just to protect against extreme outliers
            "skills": skills,
            "education": education,
            "model_accuracy": OVERALL_ACCURACY,
            "broad_field_predictions": broad_field_predictions,
            "role_matches": role_matches,
            "best_role": role_matches[0]["role"] if role_matches else None,
        })

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
        probs = logreg.predict_proba(vec)[0]
        classes = logreg.classes_
        top3_idx = probs.argsort()[-3:][::-1]
        broad_field_predictions = [
            {"category": str(classes[i]), "confidence": round(float(probs[i]) * 100, 1)}
            for i in top3_idx
        ]

        role_matches = match_roles(recognized_skills, top_n=3)

        return jsonify({
            "name": name,
            "skills": recognized_skills,
            "unrecognized_skills": unrecognized_skills,
            "education": education_found,
            "experience_years": experience_years,
            "model_accuracy": OVERALL_ACCURACY,
            "broad_field_predictions": broad_field_predictions,
            "role_matches": role_matches,
            "best_role": role_matches[0]["role"] if role_matches else None,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Analysis failed: {type(e).__name__}: {e}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
