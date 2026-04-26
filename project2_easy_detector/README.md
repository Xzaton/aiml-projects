# 🔍 Project 2: Dark Pattern Text Detector
**Difficulty:** Easy | **Tech:** Python + Flask + Regex NLP  
**By:** Piyush Mishra | SAP: 590024892

## What it does
Paste any website copy, email, or UI text — the tool detects dark patterns using
rule-based regex matching across 7 dark pattern categories and gives a Manipulation Score.

## Setup & Run
```bash
pip install flask
python app.py
# Open http://localhost:5001
```

## Features
- 7 dark pattern categories with keyword rules
- Manipulation score (0–100)
- Highlights matched text snippets
- 4 built-in sample texts to test
- REST API endpoint: POST /analyze

## Pattern Categories Detected
1. Scarcity Pressure
2. Confirm Shaming
3. Forced Continuity
4. Hidden Costs
5. Misdirection
6. Privacy Manipulation
7. Social Proof Manipulation
