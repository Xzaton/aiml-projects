# 🤖 Project 3: ML Dark Pattern Analyzer
**Difficulty:** Medium | **Tech:** Python + Flask + scikit-learn (TF-IDF + Logistic Regression)  
**By:** Piyush Mishra | SAP: 590024892

## What it does
Uses a trained ML model (TF-IDF vectorizer + Logistic Regression) to classify
each sentence of input text as "Dark Pattern" or "Clean" with confidence scores.

## Setup & Run
```bash
pip install flask scikit-learn numpy pandas
python app.py
# Open http://localhost:5002
```

## Features
- ML pipeline: TF-IDF (n-gram 1–3) + Logistic Regression
- Sentence-level classification with confidence %
- Overall manipulation score + risk level
- Model auto-trains on first run, saves to disk
- /model-info endpoint returns accuracy & classification report
- 60 labeled training examples built-in

## How it Works
1. Input text is split into sentences
2. Each sentence is vectorized using TF-IDF
3. Logistic Regression predicts label + probability
4. Results aggregated into overall risk score
