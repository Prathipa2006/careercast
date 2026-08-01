"""
Inspect what's inside your .pkl files - since they're binary and
can't be opened directly in VS Code, this loads them with pickle
and prints their contents in a readable way.

Run:
    python inspect_pkl.py
"""

import pickle

print("=" * 60)
print("TF-IDF VECTORIZER")
print("=" * 60)
with open("model/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)
print(f"Vocabulary size: {len(tfidf.vocabulary_)}")
print(f"Sample terms: {list(tfidf.vocabulary_.keys())[:15]}")
print(f"max_features: {tfidf.max_features}")
print(f"ngram_range: {tfidf.ngram_range}")

print("\n" + "=" * 60)
print("SELECTOR (SelectKBest)")
print("=" * 60)
try:
    with open("model/selector.pkl", "rb") as f:
        selector = pickle.load(f)
    print(f"k (features kept): {selector.k}")
    print(f"Original features in: {selector.n_features_in_}")
except FileNotFoundError:
    print("No selector.pkl found (not used in this pipeline).")

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION MODEL")
print("=" * 60)
with open("model/logreg_model.pkl", "rb") as f:
    model = pickle.load(f)
print(f"Classes (categories): {list(model.classes_)}")
print(f"Number of classes: {len(model.classes_)}")
print(f"C (regularization): {model.C}")
print(f"max_iter: {model.max_iter}")
