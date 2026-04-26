# 🕵️ Dark Pattern Detector

A web application that analyzes UI/UX text for manipulative dark patterns using rule-based NLP.

## 📌 Project Info
- **Type:** Easy (AIML / Python)
- **Tech:** Python, Flask, HTML/CSS/JS
- **Topic:** Dark Patterns, Consumer Protection, HCI

## 🚀 Run Locally

```bash
pip install -r requirements.txt
python app.py
```
Open: http://localhost:5000

## 🔍 What it detects
- Scarcity Pressure ("Only 2 left!")
- Urgency / Countdown ("Offer expires in...")
- Confirm Shaming ("No thanks, I hate saving money")
- Hidden Costs (service fees revealed late)
- Forced Continuity (auto-renew traps)
- Misdirection (pre-selected options)
- Roach Motel (call to cancel)
- Social Proof Manipulation

## 📁 Structure
```
easy1/
├── app.py              # Flask backend + detection logic
├── requirements.txt
└── templates/
    └── index.html      # Frontend UI
```
