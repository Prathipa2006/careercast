"""
Run this in your project folder to check if your model files are
actually consistent with each other (same training run).

Run:
    python check_model_consistency.py
"""
import pickle
import json
import os

MODEL_DIR = "model"

print("=== File check ===")
for fname in ["tfidf_vectorizer.pkl", "selector.pkl", "rf_model.pkl", "metrics.json"]:
    path = f"{MODEL_DIR}/{fname}"
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    print(f"{fname}: exists={exists}, size={size} bytes")

print("\n=== Feature dimension check ===")
with open(f"{MODEL_DIR}/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)
print(f"TF-IDF vocabulary size: {len(tfidf.vocabulary_)}")

selector = None
if os.path.exists(f"{MODEL_DIR}/selector.pkl"):
    with open(f"{MODEL_DIR}/selector.pkl", "rb") as f:
        selector = pickle.load(f)
    print(f"Selector expects input features: {selector.n_features_in_}")
    print(f"Selector outputs k features: {selector.k}")

with open(f"{MODEL_DIR}/rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)
print(f"RF model expects input features: {rf_model.n_features_in_}")

print("\n=== Consistency check ===")
if selector is not None:
    if tfidf.vocabulary_ and len(tfidf.vocabulary_) != selector.n_features_in_:
        print("MISMATCH: TF-IDF vocab size != what selector expects - files are from DIFFERENT runs!")
    else:
        print("OK: TF-IDF -> selector dimensions match")

    if selector.k != rf_model.n_features_in_:
        print(f"MISMATCH: selector outputs {selector.k} features but RF model expects {rf_model.n_features_in_} - BROKEN PIPELINE")
    else:
        print("OK: selector -> RF model dimensions match")
else:
    if len(tfidf.vocabulary_) != rf_model.n_features_in_:
        print(f"MISMATCH: TF-IDF outputs {len(tfidf.vocabulary_)} features but RF model expects {rf_model.n_features_in_} features (no selector found)")
    else:
        print("OK: TF-IDF -> RF model dimensions match directly (no selector)")

print("\n=== Real test: known IT-heavy text should predict confidently ===")
import re
def clean(t):
    t = str(t).lower()
    t = re.sub(r'\d+', ' ', t)
    t = re.sub(r'[^a-z\s]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()

test_text = "python sql machine learning data analysis pandas numpy data visualization tableau excel statistics"
cleaned = clean(test_text)
vec = tfidf.transform([cleaned])
if selector is not None:
    vec = selector.transform(vec)

probs = rf_model.predict_proba(vec)[0]
top_idx = probs.argsort()[-3:][::-1]
print("Top 3 predictions for obviously data-analyst-like text:")
for i in top_idx:
    print(f"  {rf_model.classes_[i]}: {probs[i]*100:.1f}%")
print("\nIf these are also near-random (~4-7% each), the model itself is broken.")
print("If these show a CLEAR, confident, sensible top prediction, the model is")
print("fine and the issue is specific to the resume text you tested with Streamlit.")