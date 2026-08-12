# CareerCast — Milestone 1

## Setup
```
pip install -r requirements.txt
```

## 1. Place your dataset
Copy `Resume.csv` into this folder (same level as `app.py`).

## 2. Train the baseline model (creates model/*.pkl)
```
python train_model.py
```

## 3. Run the app
```
python app.py
```
Open (https://careercast.onrender.com/) in your browser.

## What it does
- Drag-and-drop a resume (.pdf / .docx / .txt)
- Backend extracts text, pulls out skills & education
- Runs the trained TF-IDF + Logistic Regression model → broad field prediction (e.g. IT, Finance)
- Runs skill-to-role matching → specific role suggestion (e.g. Data Scientist) with match % and skill gaps
- Frontend renders everything: highlighted skills, accuracy, ranked roles, best-match verdict

## implemented random forest and xgboost

