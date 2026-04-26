# ⚡ Project 4: Advanced Dark Pattern Detection System
**Difficulty:** Hard | **Tech:** Python + Flask + scikit-learn (4 ML models) + 8-class classification  
**By:** Piyush Mishra | SAP: 590024892

## What it does
A full multi-model ML system for detecting and classifying dark patterns into 8 categories.
Trains 4 classifiers, compares their performance, and lets you analyze text with any model.

## Setup & Run
```bash
pip install flask scikit-learn numpy pandas
python app.py
# Open http://localhost:5003
```

## Features
- 8-class classification: Scarcity, Confirm Shaming, Forced Continuity, Hidden Costs,
  Roach Motel, Misdirection, Privacy Manipulation, Clean
- 4 ML models: Logistic Regression, Naive Bayes, Linear SVM, Random Forest
- Auto-selects best performing model
- Model comparison table (Accuracy, Precision, Recall, F1, Cross-Val)
- Sentence-level predictions with confidence scores
- 3 tabs: Analyze / Model Comparison / System Info
- REST API endpoints: /analyze, /model-metrics, /dataset-stats

## API Endpoints
- GET  /              → Web UI
- POST /analyze       → {text, model} → predictions
- GET  /model-metrics → model performance stats
- GET  /dataset-stats → dataset distribution

## Architecture
Raw Text → Sentence Split → TF-IDF (n-gram 1–3) → Classifier → Category + Confidence
