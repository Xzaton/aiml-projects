# 🤖 DarkBot — AI Intent Classification Chatbot

An AI-powered chatbot that answers questions about dark patterns using intent classification (SVM + TF-IDF), entity extraction, and context-aware response generation.

## 📌 Project Info
- **Type:** Hard (AIML / Python / NLP / ML)
- **Tech:** Python, Flask, Scikit-learn, LinearSVC, TF-IDF, Session Management
- **Topic:** Chatbot, Intent Classification, NLP, Dark Patterns

## 🏗️ Architecture
```
User Input
    ↓
Preprocessing (lowercase, clean)
    ↓
TF-IDF Vectorizer (char n-grams, 1-3)
    ↓
LinearSVC Classifier (calibrated probabilities)
    ↓
Intent → Response Selection
    ↓
Context Manager (conversation history)
    ↓
Entity Extractor (URLs, prices)
    ↓
Response + Follow-up suggestion
```

## 🚀 Run Locally

```bash
pip install -r requirements.txt
python app.py   # auto-trains model on first run
```
Open: http://localhost:5003

## 🎯 Intents Supported (12)
- greeting, farewell, thanks, help
- dark_pattern_info, dark_pattern_examples
- roach_motel, confirm_shaming, privacy_manipulation
- how_to_detect, legal_info, recommendations

## ✨ Features
- Intent classification with confidence scores
- Context-aware follow-up suggestions
- Conversation history (per session)
- Entity extraction (URLs, prices)
- Typing indicator animation
- Quick-reply buttons
- Reset conversation

## 📁 Structure
```
hard1/
├── train.py            # ML training + intent data
├── app.py              # Flask app + context manager
├── requirements.txt
├── model/              # Auto-created after training
│   ├── chatbot_pipeline.pkl
│   ├── label_encoder.pkl
│   └── intents.json
└── templates/
    └── index.html      # Chat UI
```

## 🔧 How to extend
- Add new intents in `train.py` INTENTS dict
- Re-run `python train.py` to retrain
- Swap LinearSVC with BERT for higher accuracy
