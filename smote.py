# ============================================================
# SMOTE - fixes small-class problem WITHOUT merging categories
# (unlike the merge experiment, this keeps all 24 real categories,
# just synthetically balances how many training examples each has)
#
# ONE-TIME INSTALL first, in a terminal:
#   pip install imbalanced-learn
# ============================================================
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# --- Use your ORIGINAL dataset, NOT the merged one ---
df3 = pd.read_csv("Resume.csv")

X3 = df3["Resume_str"]
y3 = df3["Category"]
xtr, xte, ytr, yte = train_test_split(X3, y3, test_size=0.2, random_state=42, stratify=y3)

def clean3(t):
    t = str(t).lower()
    t = re.sub(r'\d+', ' ', t)
    t = re.sub(r'[^a-z\s]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()

xtr = xtr.apply(clean3)
xte = xte.apply(clean3)

tfidf3 = TfidfVectorizer(stop_words='english', max_features=15000, ngram_range=(1,2),
                          sublinear_tf=True, min_df=2, max_df=0.9)
xtr_vec = tfidf3.fit_transform(xtr)
xte_vec = tfidf3.transform(xte)

selector3 = SelectKBest(chi2, k=1500)
xtr_sel = selector3.fit_transform(xtr_vec, ytr)
xte_sel = selector3.transform(xte_vec)   # test set NEVER touched by SMOTE - important

# --- SMOTE: only applied to TRAINING data, never test data ---
# k_neighbors must be less than the smallest class size - Automobile
# has ~29 samples in training (36 total * 0.8), so keep k_neighbors low
smote = SMOTE(random_state=42, k_neighbors=3)
xtr_balanced, ytr_balanced = smote.fit_resample(xtr_sel, ytr)

print("Before SMOTE:", xtr_sel.shape[0], "training samples")
print("After SMOTE:", xtr_balanced.shape[0], "training samples (classes now balanced)")

results = []

# --- Logistic Regression ---
lr3 = LogisticRegression(max_iter=3000, C=1.0, random_state=42)
lr3.fit(xtr_balanced, ytr_balanced)
lr_train_acc = accuracy_score(ytr_balanced, lr3.predict(xtr_balanced))
lr_test_acc = accuracy_score(yte, lr3.predict(xte_sel))
results.append(("Logistic Regression", lr_train_acc, lr_test_acc))

# --- Random Forest ---
rf3 = RandomForestClassifier(n_estimators=300, max_depth=16, min_samples_split=10,
                              min_samples_leaf=5, random_state=42, n_jobs=1)
rf3.fit(xtr_balanced, ytr_balanced)
rf_train_acc = accuracy_score(ytr_balanced, rf3.predict(xtr_balanced))
rf_test_acc = accuracy_score(yte, rf3.predict(xte_sel))
results.append(("Random Forest", rf_train_acc, rf_test_acc))

# --- XGBoost ---
le3 = LabelEncoder()
ytr_bal_enc = le3.fit_transform(ytr_balanced)
xgb3 = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, subsample=0.7,
                      colsample_bytree=0.6, min_child_weight=5, reg_alpha=1.0, reg_lambda=3.0,
                      objective="multi:softprob", num_class=len(le3.classes_),
                      eval_metric="mlogloss", tree_method="hist", random_state=42, n_jobs=1)
xgb3.fit(xtr_balanced, ytr_bal_enc)
xgb_train_acc = accuracy_score(ytr_bal_enc, xgb3.predict(xtr_balanced))
xgb_test_acc = accuracy_score(yte, le3.inverse_transform(xgb3.predict(xte_sel)))
results.append(("XGBoost", xgb_train_acc, xgb_test_acc))

print("\n" + "="*65)
print(f"{'Model':<22} {'Train':>8} {'Test':>8} {'Gap':>8}")
print("="*65)
for name, tr, te in results:
    gap = tr - te
    flag = "OVERFIT" if gap > 0.15 else "healthy" if gap < 0.08 else "borderline"
    print(f"{name:<22} {tr:>8.4f} {te:>8.4f} {gap:>8.4f}  {flag}")
print("="*65)