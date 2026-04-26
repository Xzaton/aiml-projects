"""
AI Chatbot with Intent Classification — Hard AIML Project
Architecture: TF-IDF + SVM for intent classification
              Rule-based entity extraction
              Context-aware response generation
"""

import os, pickle, json, re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

# ── Intent Training Data ──
INTENTS = {
    "greeting": {
        "patterns": [
            "hello", "hi", "hey", "good morning", "good afternoon",
            "good evening", "howdy", "what's up", "greetings", "hi there",
            "hello there", "hey there", "sup", "hiya", "good day"
        ],
        "responses": [
            "Hello! How can I help you today?",
            "Hi there! What can I do for you?",
            "Hey! I'm here to help. What do you need?",
            "Good to see you! How can I assist?"
        ]
    },
    "farewell": {
        "patterns": [
            "bye", "goodbye", "see you", "take care", "farewell",
            "see you later", "good night", "catch you later", "later",
            "i'm leaving", "that's all", "thanks bye", "exit", "quit"
        ],
        "responses": [
            "Goodbye! Have a great day!",
            "See you later! Take care!",
            "Bye! Feel free to come back anytime.",
            "Farewell! It was nice chatting with you."
        ]
    },
    "dark_pattern_info": {
        "patterns": [
            "what is a dark pattern", "explain dark patterns", "what are dark patterns",
            "define dark pattern", "dark pattern meaning", "what does dark pattern mean",
            "tell me about dark patterns", "dark patterns definition",
            "what is deceptive design", "explain deceptive ux"
        ],
        "responses": [
            "Dark patterns are deceptive UI/UX design techniques used to manipulate users into actions they didn't intend — like hidden charges, fake urgency, or difficult cancellation flows. The term was coined by Harry Brignull in 2010.",
            "A dark pattern is a deliberately misleading interface element that benefits the company at the user's expense. Examples include pre-checked boxes, hidden fees, and subscribe traps.",
            "Dark patterns are interface tricks designed to manipulate user behavior — exploiting cognitive biases like loss aversion, default bias, and decision fatigue to serve business interests over user interests."
        ]
    },
    "dark_pattern_examples": {
        "patterns": [
            "give me examples of dark patterns", "dark pattern examples",
            "what are some dark patterns", "list dark patterns", "types of dark patterns",
            "name some dark patterns", "common dark patterns", "dark pattern types"
        ],
        "responses": [
            "Common dark patterns include:\n• Roach Motel (easy to sign up, hard to cancel)\n• Confirm Shaming ('No thanks, I hate savings')\n• Hidden Costs (fees at final checkout)\n• Forced Continuity (silent trial-to-paid conversion)\n• Scarcity Pressure ('Only 2 left!')\n• Privacy Zuckering (confusing consent settings)",
            "Key dark pattern types:\n1. Hidden Costs\n2. Roach Motel\n3. Confirm Shaming\n4. Misdirection\n5. Forced Continuity\n6. Privacy Manipulation\n7. Scarcity/Urgency Pressure\n8. Social Proof Manipulation"
        ]
    },
    "roach_motel": {
        "patterns": [
            "what is roach motel", "roach motel pattern", "explain roach motel",
            "hard to cancel subscription", "difficult to unsubscribe", "cancel subscription problem"
        ],
        "responses": [
            "The Roach Motel pattern means it's easy to sign up but extremely difficult to cancel. For example, a streaming service lets you subscribe in 2 clicks but requires calling support during business hours to cancel.",
            "Roach Motel: You can check in but can't check out. It's when companies make enrollment instant but bury the cancellation option under support calls, long forms, or waiting periods."
        ]
    },
    "confirm_shaming": {
        "patterns": [
            "what is confirm shaming", "confirm shaming example", "guilt trip in ux",
            "shaming decline button", "insulting reject button", "manipulative decline option"
        ],
        "responses": [
            "Confirm shaming uses emotionally manipulative text on the decline option — like 'No thanks, I don't want to save money' — to guilt users into accepting rather than clicking no.",
            "Confirm shaming is when the 'reject' button says something like 'No, I prefer paying full price' — designed to make you feel bad about declining an offer."
        ]
    },
    "privacy_manipulation": {
        "patterns": [
            "privacy dark pattern", "gdpr dark pattern", "cookie consent manipulation",
            "data collection manipulation", "forced data sharing", "privacy zuckering",
            "consent dark pattern", "manipulative cookie banner"
        ],
        "responses": [
            "Privacy Zuckering involves interfaces that confuse users into sharing more data than intended. Asymmetric cookie banners — 'Accept all' in one click, 'Reject all' buried in 5 sub-menus — are the most common form. GDPR requires consent to be freely given and equally easy.",
            "Privacy dark patterns include pre-enabled tracking, confusing consent dialogs, and making 'accept all' far more prominent than 'reject'. The EDPB issued guidelines in 2022 specifically targeting these on social media platforms."
        ]
    },
    "how_to_detect": {
        "patterns": [
            "how to detect dark patterns", "identify dark patterns", "spot dark patterns",
            "how to find dark patterns", "dark pattern detection", "check for dark patterns"
        ],
        "responses": [
            "To detect dark patterns:\n• Ask: Is rejecting as easy as accepting?\n• Check if costs are shown early\n• Look for asymmetric button sizes\n• Search for urgency/scarcity language\n• Check cancellation flow complexity\n• Use automated tools or browser extensions",
            "Detection methods include:\n1. Expert UX audit\n2. User testing (repeated confusion = dark pattern)\n3. ML classifiers (TF-IDF on UI text)\n4. Heuristic checklists\n5. Browser extensions that flag manipulative elements"
        ]
    },
    "legal_info": {
        "patterns": [
            "are dark patterns illegal", "dark pattern law", "gdpr dark patterns",
            "ftc dark patterns", "legal dark patterns", "regulation dark patterns",
            "digital services act dark patterns", "laws against dark patterns"
        ],
        "responses": [
            "Dark patterns face increasing legal scrutiny:\n• GDPR: Consent via dark patterns is invalid (EU)\n• FTC: Published 2022 report making dark patterns an enforcement priority (US)\n• DSA (Digital Services Act): Explicitly bans dark patterns on large platforms (EU, 2022)\n• UK: Age Appropriate Design Code protects minors\nEnforcement remains challenging as interfaces change rapidly.",
            "Legally, GDPR (EU) and FTC guidelines (US) both prohibit deceptive patterns. The EU's Digital Services Act (2022) directly names dark patterns as prohibited for large platforms. Ireland's DPC fined Meta over €1.2 billion in 2023 partly for consent manipulation."
        ]
    },
    "recommendations": {
        "patterns": [
            "how to avoid dark patterns", "design ethically", "ethical ux design",
            "prevent dark patterns", "good ux practices", "transparent design",
            "user friendly design", "honest design principles"
        ],
        "responses": [
            "Ethical design principles to avoid dark patterns:\n• Equal prominence for accept and reject options\n• Show all costs upfront\n• Make cancellation as easy as sign-up\n• Privacy-protective defaults\n• Clear, non-shaming language\n• Regular ethical design audits",
            "To prevent dark patterns: implement symmetric design, transparent pricing, frictionless exit flows, privacy by default, and establish an internal ethics review process. Measure success with trust metrics, not just conversions."
        ]
    },
    "thanks": {
        "patterns": [
            "thank you", "thanks", "thank you so much", "thanks a lot",
            "appreciate it", "cheers", "helpful", "that helped", "great help"
        ],
        "responses": [
            "You're welcome! Happy to help.",
            "Glad I could help! Anything else?",
            "Anytime! Feel free to ask more questions.",
            "Happy to assist! Let me know if you need more info."
        ]
    },
    "help": {
        "patterns": [
            "help", "what can you do", "what do you know", "your capabilities",
            "topics", "what topics", "menu", "options", "features"
        ],
        "responses": [
            "I can help you with:\n• What dark patterns are\n• Types and examples of dark patterns\n• Specific patterns (Roach Motel, Confirm Shaming, etc.)\n• How to detect dark patterns\n• Legal and regulatory info (GDPR, FTC, DSA)\n• Ethical design recommendations\n\nJust ask away!"
        ]
    }
}

def build_training_data():
    X, y = [], []
    for intent, data in INTENTS.items():
        for pattern in data["patterns"]:
            X.append(pattern.lower())
            y.append(intent)
    return X, y

def train_and_save():
    print("Building training data...")
    X, y = build_training_data()
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1,3), max_features=3000,
            analyzer='char_wb', sublinear_tf=True
        )),
        ('clf', CalibratedClassifierCV(LinearSVC(max_iter=2000, C=1.0)))
    ])

    print(f"Training on {len(X)} samples across {len(le.classes_)} intents...")
    pipeline.fit(X, y_enc)
    preds = pipeline.predict(X)
    acc = accuracy_score(y_enc, preds)
    print(f"Training accuracy: {acc:.2%}")

    os.makedirs('model', exist_ok=True)
    with open('model/chatbot_pipeline.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    with open('model/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    with open('model/intents.json', 'w') as f:
        json.dump(INTENTS, f, indent=2)

    print("Model saved!")
    return pipeline, le, INTENTS

if __name__ == '__main__':
    train_and_save()
