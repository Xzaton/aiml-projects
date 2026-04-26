"""
Fake News Detector — Medium AIML Project
Uses TF-IDF + Logistic Regression / Naive Bayes
Run: python train.py  (trains and saves model)
Then: python app.py   (launches web app)
"""

import os, pickle, re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

# ── Synthetic training data (balanced, realistic) ──
# In production: replace with LIAR dataset or FakeNewsNet
REAL_HEADLINES = [
    "Scientists publish peer-reviewed study on climate change impacts",
    "Government announces new budget allocation for healthcare",
    "Stock markets close higher after economic data release",
    "University researchers develop new cancer detection method",
    "City council approves infrastructure improvement plan",
    "Central bank raises interest rates by 0.25 percent",
    "New trade agreement signed between two nations",
    "Sports team wins championship after strong season",
    "Tech company reports quarterly earnings above expectations",
    "Health ministry issues guidelines for seasonal flu prevention",
    "Parliament passes bill on renewable energy subsidies",
    "International court issues ruling on territorial dispute",
    "Research shows moderate exercise improves mental health",
    "Airline introduces new direct routes to major destinations",
    "Education board revises curriculum following public consultation",
    "Police arrest suspect in connection with bank robbery",
    "Hospital opens new emergency department wing",
    "Election commission announces date for regional elections",
    "Scientists confirm discovery of new species in rainforest",
    "Food safety authority recalls product due to contamination risk",
    "Prime minister meets foreign delegation to discuss trade",
    "Company announces merger pending regulatory approval",
    "Flooding affects thousands in coastal region",
    "New study links air pollution to respiratory diseases",
    "Government releases annual economic growth report",
    "Museum opens exhibition of ancient artifacts",
    "Startup raises funding to expand operations",
    "Weather agency forecasts above normal rainfall this season",
    "Court upholds decision on environmental protection law",
    "Public transit authority expands bus service",
]

FAKE_HEADLINES = [
    "SHOCKING: Government secretly putting mind control chemicals in water supply",
    "EXPOSED: Vaccines contain microchips to track citizens, whistleblower reveals",
    "You won't BELIEVE what they are hiding about the moon landing",
    "BREAKING: Secret society controls all world governments, documents prove",
    "Doctors don't want you to know this one cure for all diseases",
    "BOMBSHELL: Entire political establishment involved in massive cover-up",
    "URGENT: Share before they delete this — truth about 5G towers revealed",
    "Billionaires meeting in secret to plan global population reduction",
    "This miracle fruit cures cancer in 24 hours, big pharma is furious",
    "LEAKED: Government plan to ban all cash and control every purchase",
    "Scientist who discovered climate is a hoax found dead mysteriously",
    "PROOF: Elections are completely rigged by a global shadow organization",
    "Ancient aliens built the pyramids, finally admitted by archaeologists",
    "This household chemical kills coronavirus instantly, doctors confirm",
    "CELEBRITY reveals illuminati tried to silence them for speaking truth",
    "They are spraying chemicals from planes to control weather and minds",
    "WARNING: New law will allow government to enter your home without permission",
    "BANNED VIDEO: What they don't teach you in school about history",
    "Eating this one food every day makes you immune to all viruses",
    "WHISTLEBLOWER: Major banks plan to steal your savings next week",
    "Secret government program creates weather disasters to sell insurance",
    "THIS is the real reason they want you vaccinated — it will shock you",
    "CONFIRMED: Famous actor is actually a robot built by tech company",
    "The earth is actually flat and NASA admits it in leaked emails",
    "SHARE NOW: Government is reading all your messages without warrant",
    "New world order plan leaked — elites planning to reduce population",
    "Doctor fired for revealing that common pain medicine causes cancer",
    "WATCH: Politician caught on camera admitting elections are fake",
    "Ancient remedy big pharma doesn't want you to know about",
    "They hid this from us for decades — the truth about moon water",
]

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def build_dataset():
    texts = [preprocess(h) for h in REAL_HEADLINES + FAKE_HEADLINES]
    labels = [0]*len(REAL_HEADLINES) + [1]*len(FAKE_HEADLINES)  # 0=real, 1=fake
    return texts, labels

def train_and_save():
    print("Building dataset...")
    texts, labels = build_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    # TF-IDF + Logistic Regression pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1,2), max_features=5000,
            stop_words='english', sublinear_tf=True
        )),
        ('clf', LogisticRegression(max_iter=1000, C=1.0, random_state=42))
    ])
    print("Training model...")
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.2%}")
    print(classification_report(y_test, preds, target_names=['Real','Fake']))

    os.makedirs('model', exist_ok=True)
    with open('model/pipeline.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    print("Model saved to model/pipeline.pkl")
    return pipeline

if __name__ == '__main__':
    train_and_save()
