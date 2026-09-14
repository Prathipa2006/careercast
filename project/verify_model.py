"""
CareerCast - Model Verification Script
---------------------------------------------------------------
Sanity-checks your LIVE running app (not just the notebook) by
sending real resumes from Resume.csv through /analyze and checking
if the broad-field prediction matches the known true Category.

Run this WHILE app.py is already running in another terminal:
    python verify_model.py
"""

import random
import requests
import pandas as pd
from sklearn.model_selection import train_test_split

APP_URL = "http://127.0.0.1:5000/analyze"
SAMPLE_SIZE = 20  # how many resumes to test

df = pd.read_csv("Resume.csv").dropna(subset=["Resume_str", "Category"])
x = df["Resume_str"]
y = df["Category"]

# IMPORTANT: this must use the EXACT SAME split params as your notebook
# (test_size=0.2, random_state=42, stratify=y) so we only test on
# resumes the model has NEVER seen during training. Sampling from the
# full Resume.csv would include training data and give a fake, inflated
# accuracy (this is what happened before - 100% because it was testing
# on memorized data, not real held-out data).
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

test_df = pd.DataFrame({"Resume_str": x_test, "Category": y_test})
sample = test_df.sample(n=SAMPLE_SIZE, random_state=7)

correct = 0
results = []

for idx, row in sample.iterrows():
    text = row["Resume_str"]
    true_label = row["Category"]

    # write resume text to a temp .txt file and upload it, just like
    # the real frontend would
    with open("_temp_resume.txt", "w", encoding="utf-8") as f:
        f.write(text)

    with open("_temp_resume.txt", "rb") as f:
        try:
            res = requests.post(APP_URL, files={"resume": f}, timeout=30)
            data = res.json()
        except Exception as e:
            print(f"Resume {idx}: request failed -> {e}")
            continue

    if "error" in data:
        print(f"Resume {idx} ({true_label}): app returned an error -> {data['error']}")
        continue

    predicted_label = data["broad_field_predictions"][0]["category"]
    confidence = data["broad_field_predictions"][0]["confidence"]
    is_correct = (predicted_label == true_label)
    correct += int(is_correct)

    results.append({
        "id": idx,
        "true": true_label,
        "predicted": predicted_label,
        "confidence": confidence,
        "correct": is_correct,
    })

    status = "✅" if is_correct else "❌"
    print(f"{status} True: {true_label:<25} Predicted: {predicted_label:<25} ({confidence}%)")

print(f"\n--- Verification Summary (held-out TEST SET ONLY) ---")
print(f"Tested: {len(results)} resumes (from the 20% test split, never seen in training)")
print(f"Correct: {correct}")
print(f"Live accuracy on this sample: {correct/len(results)*100:.1f}%")
print(f"\nCompare this to your notebook's reported test accuracy (81.49%).")
print(f"This time both numbers are measuring the SAME thing: performance")
print(f"on data the model never trained on. They should now be reasonably close.")
