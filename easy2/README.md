# 💬 Sentiment Analyzer

Lexicon-based NLP sentiment analyzer. Detects positive, negative, and neutral sentiment with word-level analysis.

## 📌 Project Info
- **Type:** Easy (AIML / Python / NLP)
- **Tech:** Python, Flask, HTML/CSS/JS
- **No ML libraries required** — pure lexicon-based

## 🚀 Run Locally

```bash
pip install flask
python app.py
```
Open: http://localhost:5001

## ✨ Features
- Single text + batch multi-line analysis
- Compound sentiment score (-1 to +1)
- Positive/Negative word highlighting
- Handles negation ("not good" → negative)
- Handles intensifiers ("very good" → stronger positive)
- Word count, sentence count stats

## 📁 Structure
```
easy2/
├── app.py
├── requirements.txt
└── templates/
    └── index.html
```
