# 📰 Fake News Detector

ML-powered fake news detection using TF-IDF vectorization and Logistic Regression classifier.

## 📌 Project Info
- **Type:** Medium (AIML / Python / Machine Learning)
- **Tech:** Python, Flask, Scikit-learn, TF-IDF, Logistic Regression
- **Topic:** NLP, Misinformation Detection, Text Classification

## 🚀 Run Locally

```bash
pip install -r requirements.txt
python train.py     # Train and save the model
python app.py       # Start the web app
```
Open: http://localhost:5002

> **Note:** `app.py` auto-trains if no model file is found.

## 🤖 How it works
1. Text is preprocessed (lowercased, cleaned)
2. TF-IDF vectorizer converts text to feature vectors (bigrams, 5000 features)
3. Logistic Regression predicts Real (0) or Fake (1)
4. Heuristic rules add explainability (clickbait signals, conspiracy language, attribution)

## 📁 Structure
```
medium1/
├── train.py            # Training script + dataset
├── app.py              # Flask API + prediction logic
├── requirements.txt
├── model/              # Created after training
│   └── pipeline.pkl
└── templates/
    └── index.html
```

## 📈 Upgrade path (for better accuracy)
- Replace synthetic data with LIAR dataset (Kaggle)
- Use LSTM or BERT for production accuracy
