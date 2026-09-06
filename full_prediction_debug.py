"""
Run this to see the FULL probability breakdown (all 24 categories,
not just the top 10 chart) for your exact resume text.

Run:
    python full_prediction_debug.py
"""
import pickle
import re
import os

MODEL_DIR = "model"

with open(f"{MODEL_DIR}/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)
with open(f"{MODEL_DIR}/rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)
selector = None
if os.path.exists(f"{MODEL_DIR}/selector.pkl"):
    with open(f"{MODEL_DIR}/selector.pkl", "rb") as f:
        selector = pickle.load(f)


def clean(t):
    t = str(t).lower()
    t = re.sub(r'\d+', ' ', t)
    t = re.sub(r'[^a-z\s]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()


# PASTE YOUR EXACT RESUME TEXT HERE - same one that gave weird results
resume_text = """
GenAI-Powered Data Analytics Virtual Internship
Completed a virtual internship focused on data analysis and visualization.
Used analytics tools to generate insights and support business decisions.
YUVA | Data Analytics Intern (April 2026)
Analyzed Titanic dataset using Python (Pandas), Visualized survival trends using
Matplotlib/Seaborn, Cleaned missing values and handled outliers
CERTIFICATIONS:
Data Science Certification (Python & Data Visualization) - CIRF
Completed Data Analytics certification from Simplilearn in Python
"""

print("--- What clean() actually produces (this is what the model sees) ---")
cleaned = clean(resume_text)
print(cleaned)
print(f"\nWord count after cleaning: {len(cleaned.split())}")

vec = tfidf.transform([cleaned])
print(f"\nNon-zero TF-IDF features matched: {vec.nnz}")
if selector is not None:
    vec = selector.transform(vec)
    print(f"Non-zero features after selection: {vec.nnz}")

probs = rf_model.predict_proba(vec)[0]
classes = rf_model.classes_

# Sort ALL 24 categories by probability, print every single one
ranked = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)

print("\n--- FULL ranking, all categories ---")
for i, (cat, prob) in enumerate(ranked, 1):
    marker = " <-- INFORMATION-TECHNOLOGY" if cat == "INFORMATION-TECHNOLOGY" else ""
    print(f"{i:2d}. {cat:25s} {prob*100:5.1f}%{marker}")