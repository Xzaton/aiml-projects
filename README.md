# 🛡️ Dark Pattern Detection — AIML Projects
**Student:** Piyush Mishra | **SAP ID:** 590024892  
**Submitted to:** Dr. Tanupriya Chaudhary  
**Department:** Computer Science Engineering

---

## 📁 Project Overview

| # | Project | Difficulty | Port | Tech |
|---|---------|-----------|------|------|
| 1 | Dark Pattern Awareness Quiz | 🟢 Easy | 5000 | Flask |
| 2 | Dark Pattern Text Detector | 🟢 Easy | 5001 | Flask + Regex |
| 3 | ML Sentiment Analyzer | 🟡 Medium | 5002 | Flask + scikit-learn |
| 4 | Advanced Detection System | 🔴 Hard | 5003 | Flask + 4 ML Models |
| 6 | Full Dashboard | 🔴 Hard | 5004 | Flask + Chart.js |

---

## 🚀 Quick Start (All Projects)

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/dark-patterns-aiml
cd dark-patterns-aiml

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run any project
cd project1_easy_quiz    && python app.py   # http://localhost:5000
cd project2_easy_detector && python app.py  # http://localhost:5001
cd project3_medium_sentiment && python app.py # http://localhost:5002
cd project4_hard_darkpattern_ml && python app.py # http://localhost:5003
cd project4_hard_darkpattern_ml && python dashboard.py # http://localhost:5004
```

---

## 🎯 Project 1: Dark Pattern Awareness Quiz (Easy)
**`project1_easy_quiz/app.py`** → `http://localhost:5000`

Interactive 10-question quiz covering all major dark pattern types.
- Randomized question order
- Instant feedback with color coding
- Explanations for every answer
- Score card with badges

---

## 🔍 Project 2: Dark Pattern Text Detector (Easy)
**`project2_easy_detector/app.py`** → `http://localhost:5001`

Rule-based regex detector for dark patterns in any UI copy or website text.
- 7 pattern categories
- Manipulation score (0–100)
- Snippet highlighting
- 4 built-in sample texts
- REST API: `POST /analyze`

---

## 🤖 Project 3: ML Dark Pattern Analyzer (Medium)
**`project3_medium_sentiment/app.py`** → `http://localhost:5002`

Trained ML model (TF-IDF + Logistic Regression) for sentence-level classification.
- Binary classification: Dark Pattern / Clean
- Confidence scores per sentence
- Auto-trains and saves model on first run
- `/model-info` endpoint for metrics

---

## ⚡ Project 4: Advanced Detection System (Hard)
**`project4_hard_darkpattern_ml/app.py`** → `http://localhost:5003`

Full multi-model, 8-class dark pattern classification system.
- 8 categories: Scarcity, Confirm Shaming, Forced Continuity, Hidden Costs, Roach Motel, Misdirection, Privacy, Clean
- 4 ML models: Logistic Regression, Naive Bayes, Linear SVM, Random Forest
- Auto-selects best model by F1 score
- Model comparison tab with metrics table
- REST API: `POST /analyze`, `GET /model-metrics`, `GET /dataset-stats`

---

## 📊 Project 6: Full Dashboard
**`project4_hard_darkpattern_ml/dashboard.py`** → `http://localhost:5004`

Full GUI dashboard with 4 panels:
- **Overview:** KPI cards + 4 Chart.js charts (distribution, model comparison, radar, scatter)
- **Analyzer:** Real-time text analysis with donut chart + sentence breakdown
- **Models:** Comparison table with bar charts
- **History:** Session analysis log

**Screenshots for submission:**
> Open http://localhost:5004 → Take screenshot of each panel

---

## 🔬 ML Architecture

```
Input Text
    ↓
Sentence Splitting (regex)
    ↓
TF-IDF Vectorization (n-gram 1–3, max 5000 features)
    ↓
Classifier (LR / NB / SVM / RF)
    ↓
Category + Confidence Score
    ↓
Risk Level (High / Medium / Low)
```

---

## 📚 Dark Pattern Categories

| Category | Description |
|----------|-------------|
| Scarcity Pressure | False urgency via stock/time limits |
| Confirm Shaming | Guilt-tripping opt-out buttons |
| Forced Continuity | Silent trial-to-paid conversion |
| Hidden Costs | Fees revealed only at checkout |
| Roach Motel | Easy in, hard to exit |
| Misdirection | Visual steering toward business option |
| Privacy Manipulation | Opaque data-sharing consent |
| Clean | Honest, transparent UI copy |

---

## 📜 References
- Brignull, H. (2010). Dark Patterns
- Gray et al. (2018). CHI Conference on HCI
- Mathur et al. (2019). Princeton University Study
- FTC Report (2022). Bringing Dark Patterns to Light
- GDPR Article 7 — Conditions for Consent
