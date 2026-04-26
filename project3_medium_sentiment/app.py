"""
Project 3: Dark Pattern Sentiment & Manipulation Analyzer (ML-based)
Difficulty: MEDIUM
Tech: Python + Flask + scikit-learn + TF-IDF
Run: pip install flask scikit-learn numpy pandas && python app.py
"""

from flask import Flask, render_template_string, request, jsonify
import numpy as np
import json
import re
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

app = Flask(__name__)

# ── Training Data ──────────────────────────────────────────────────────────────
TRAINING_DATA = [
    # (text, label)  label: 0=clean, 1=dark_pattern
    ("Only 3 left in stock — order soon!", 1),
    ("Hurry! This deal expires in 10 minutes.", 1),
    ("847 people are viewing this item right now.", 1),
    ("Limited time offer — ends tonight!", 1),
    ("Act now before it's too late!", 1),
    ("Only 1 seat remaining — book immediately!", 1),
    ("Selling fast — 92% claimed!", 1),
    ("No thanks, I don't want to save money.", 1),
    ("No thanks, I hate free shipping.", 1),
    ("I prefer to pay full price.", 1),
    ("No, I don't want to improve my skills.", 1),
    ("Skip this offer — I'm fine missing out.", 1),
    ("Your free trial will automatically convert to a paid plan.", 1),
    ("We will charge your card ₹999/month after 7 days unless cancelled.", 1),
    ("Auto-renewal is enabled. Cancel anytime (but we make it hard).", 1),
    ("By signing up you agree to receive marketing from our partners.", 1),
    ("Price shown does not include booking fee, service fee, and taxes.", 1),
    ("Additional charges may apply at checkout.", 1),
    ("The recommended plan is pre-selected for your convenience.", 1),
    ("We share your data with third-party advertising partners by default.", 1),
    ("10,000 customers joined this week. Don't be left out!", 1),
    ("Best seller — most popular choice among our customers.", 1),
    ("Uncheck this box if you do not wish to not receive our newsletter.", 1),
    ("By continuing to browse you consent to all cookies.", 1),
    ("Your account will be charged automatically each month.", 1),
    ("Free cancellation — but only if done 30 days before arrival.", 1),
    ("Welcome to our store. Browse our collection.", 0),
    ("Add items to your cart and proceed to checkout.", 0),
    ("Our prices include all taxes and fees.", 0),
    ("Cancel your subscription anytime in two clicks.", 0),
    ("Your data is never shared with third parties.", 0),
    ("Privacy is our priority. All tracking is opt-in.", 0),
    ("Thank you for your purchase. Your order will arrive in 3–5 days.", 0),
    ("Read our full terms before subscribing.", 0),
    ("All prices shown are final with no hidden charges.", 0),
    ("You can delete your account at any time from Settings.", 0),
    ("We will send you a reminder 3 days before your trial ends.", 0),
    ("Opt out of marketing emails using the link below.", 0),
    ("Compare all our plans before deciding.", 0),
    ("Our team is available 24/7 to assist with cancellations.", 0),
    ("This product has 4.2 stars from 1,240 verified buyers.", 0),
    ("Please review your order before confirming.", 0),
    ("No credit card required for the free plan.", 0),
    ("You are in control of your privacy settings.", 0),
    ("Items ship in 2–3 business days. Free returns within 30 days.", 0),
    ("Subscribe to our newsletter — unsubscribe anytime.", 0),
    ("Prices may vary by location. See full pricing page.", 0),
    ("We use analytics to improve your experience. Manage cookies here.", 0),
    ("Your subscription renews annually. We'll remind you 30 days before.", 0),
    ("Download limit: 5 files per day on the free plan.", 0),
    ("Only 2 rooms left at this price! Book now before they're gone.", 1),
    ("Flash sale: 80% off — 00:04:32 remaining!", 1),
    ("Warning: Your cart will expire in 5 minutes.", 1),
    ("Other buyers are waiting. Checkout before you lose your items.", 1),
    ("Your exclusive discount expires today at midnight.", 1),
    ("We noticed you tried to leave. Here's 10% off if you stay.", 1),
    ("Joining is free. Cancellation requires calling our helpdesk.", 1),
    ("By using our service you automatically consent to targeted ads.", 1),
    ("Upgrade now — our most popular plan is already selected for you.", 1),
    ("Don't miss out — your neighbors are already saving!", 1),
]

# ── Model Training ─────────────────────────────────────────────────────────────
MODEL_PATH = "dp_model.pkl"

def train_model():
    texts = [t for t, _ in TRAINING_DATA]
    labels = [l for _, l in TRAINING_DATA]
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,3), max_features=3000, sublinear_tf=True)),
        ('clf', LogisticRegression(max_iter=500, C=1.5, random_state=42))
    ])
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Model trained | Accuracy: {acc:.2%}")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(pipeline, f)
    return pipeline, acc

def load_or_train():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f), None
    return train_model()

MODEL, TRAIN_ACC = load_or_train()

# ── Sentence Splitter ──────────────────────────────────────────────────────────
def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 8]

# ── Analysis ───────────────────────────────────────────────────────────────────
def analyze_text(text):
    sentences = split_sentences(text)
    if not sentences:
        return {"error": "Too short"}

    results = []
    for sent in sentences:
        prob = MODEL.predict_proba([sent])[0]
        pred = int(MODEL.predict([sent])[0])
        results.append({
            "sentence": sent,
            "label": "Dark Pattern" if pred == 1 else "Clean",
            "confidence": float(max(prob)),
            "dark_prob": float(prob[1]),
            "clean_prob": float(prob[0]),
        })

    dark_count = sum(1 for r in results if r["label"] == "Dark Pattern")
    overall_score = round(np.mean([r["dark_prob"] for r in results]) * 100, 1)
    risk = "High" if overall_score > 60 else "Medium" if overall_score > 35 else "Low"

    return {
        "sentences": results,
        "total": len(results),
        "dark_count": dark_count,
        "clean_count": len(results) - dark_count,
        "overall_score": overall_score,
        "risk_level": risk,
    }

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>ML Dark Pattern Analyzer</title>
<style>
* { box-sizing:border-box; margin:0; padding:0; }
body { font-family:'Segoe UI',sans-serif; background:#f5f7fa; }
.header { background:linear-gradient(135deg,#1a1a2e,#16213e); color:#fff; padding:24px 40px; }
.header h1 { font-size:1.8rem; } .header p { color:#aaa; margin-top:4px; }
.container { max-width:920px; margin:36px auto; padding:0 20px; }
.card { background:#fff; border-radius:16px; padding:28px; margin-bottom:24px; box-shadow:0 2px 12px rgba(0,0,0,0.07); }
h2 { font-size:1.05rem; color:#1a1a2e; margin-bottom:14px; }
textarea { width:100%; border:2px solid #e0e0e0; border-radius:10px; padding:16px; font-size:0.95rem;
  min-height:150px; resize:vertical; outline:none; font-family:inherit; }
textarea:focus { border-color:#e94560; }
.btn { background:linear-gradient(135deg,#e94560,#c0392b); color:#fff; border:none; border-radius:10px;
  padding:14px 36px; font-size:1rem; font-weight:700; cursor:pointer; margin-top:14px; width:100%; }
.btn:hover { opacity:0.88; }
.samples { display:flex; gap:10px; flex-wrap:wrap; margin-top:12px; }
.sbtn { background:#f0f0f0; border:none; border-radius:8px; padding:7px 14px; font-size:0.82rem; cursor:pointer; color:#444; }
.sbtn:hover { background:#1a1a2e; color:#fff; }
#results { display:none; }
.metrics { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:16px; margin-bottom:24px; }
.metric { background:#fff; border-radius:14px; padding:20px; text-align:center; box-shadow:0 2px 10px rgba(0,0,0,0.06); }
.metric-val { font-size:2.4rem; font-weight:900; }
.metric-label { color:#888; font-size:0.85rem; margin-top:4px; }
.risk-high { color:#e74c3c; } .risk-medium { color:#e67e22; } .risk-low { color:#27ae60; }
.sent-item { border-radius:10px; padding:14px 18px; margin-bottom:10px; border-left:5px solid; display:flex; align-items:flex-start; gap:14px; }
.sent-dark { background:#fde8e8; border-color:#e74c3c; }
.sent-clean { background:#e8f8f0; border-color:#27ae60; }
.sent-text { flex:1; font-size:0.95rem; color:#222; line-height:1.5; }
.sent-badge { font-size:0.75rem; font-weight:700; border-radius:20px; padding:3px 12px; white-space:nowrap; }
.badge-dark { background:#e74c3c; color:#fff; }
.badge-clean { background:#27ae60; color:#fff; }
.conf-bar { height:5px; border-radius:4px; margin-top:6px; background:#e0e0e0; }
.conf-fill-dark { background:#e74c3c; height:100%; border-radius:4px; }
.conf-fill-clean { background:#27ae60; height:100%; border-radius:4px; }
</style>
</head>
<body>
<div class="header">
  <h1>🤖 ML Dark Pattern Analyzer</h1>
  <p>Project 3 &nbsp;|&nbsp; AIML &nbsp;|&nbsp; Piyush Mishra — SAP 590024892 &nbsp;|&nbsp; TF-IDF + Logistic Regression</p>
</div>
<div class="container">
  <div class="card">
    <h2>Paste any UI copy or website text for ML-based sentence-level analysis:</h2>
    <textarea id="inp" placeholder="Paste multiple sentences. Each sentence is analyzed independently by the ML model..."></textarea>
    <div class="samples">
      <span style="color:#888;font-size:0.82rem;align-self:center">Samples:</span>
      <button class="sbtn" onclick="ls(0)">High Risk Text</button>
      <button class="sbtn" onclick="ls(1)">Mixed Text</button>
      <button class="sbtn" onclick="ls(2)">Clean Text</button>
    </div>
    <button class="btn" onclick="run()">🔬 Analyze with ML Model</button>
  </div>
  <div id="results"></div>
</div>
<script>
const ss = [
  "Only 3 seats left — act now! This offer expires tonight at midnight. 847 people are currently viewing this page. No thanks, I hate saving money. Your free trial automatically converts to ₹999/month unless cancelled. Price shown excludes service fee and GST.",
  "Welcome to our store. Only 2 items left in stock — hurry! Our prices include all taxes. Auto-renewal is enabled on your account. You can cancel anytime by calling our support line. Thank you for your order.",
  "Browse our complete product catalog at your own pace. All prices include taxes and fees. You can cancel your subscription in two clicks from your account settings. We will never share your data without your consent. Free returns within 30 days."
];
function ls(i){ document.getElementById('inp').value=ss[i]; }
function run(){
  const t=document.getElementById('inp').value.trim();
  if(!t){alert('Enter text first');return;}
  fetch('/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:t})})
  .then(r=>r.json()).then(d=>{
    const R=document.getElementById('results'); R.style.display='block';
    const rClass=d.risk_level==='High'?'risk-high':d.risk_level==='Medium'?'risk-medium':'risk-low';
    let html=`<div class="metrics">
      <div class="metric"><div class="metric-val ${rClass}">${d.overall_score}%</div><div class="metric-label">Manipulation Score</div></div>
      <div class="metric"><div class="metric-val ${rClass}">${d.risk_level}</div><div class="metric-label">Risk Level</div></div>
      <div class="metric"><div class="metric-val" style="color:#e74c3c">${d.dark_count}</div><div class="metric-label">Dark Sentences</div></div>
      <div class="metric"><div class="metric-val" style="color:#27ae60">${d.clean_count}</div><div class="metric-label">Clean Sentences</div></div>
      <div class="metric"><div class="metric-val" style="color:#2980b9">${d.total}</div><div class="metric-label">Total Analyzed</div></div>
    </div><div class="card"><h2>Sentence-level Analysis:</h2>`;
    d.sentences.forEach(s=>{
      const isDark=s.label==='Dark Pattern';
      const pct=Math.round(s.confidence*100);
      html+=`<div class="sent-item ${isDark?'sent-dark':'sent-clean'}">
        <div class="sent-text">"${s.sentence}"
          <div class="conf-bar"><div class="${isDark?'conf-fill-dark':'conf-fill-clean'}" style="width:${pct}%"></div></div>
          <small style="color:#888">${pct}% confidence</small>
        </div>
        <span class="sent-badge ${isDark?'badge-dark':'badge-clean'}">${s.label}</span>
      </div>`;
    });
    html+='</div>';
    R.innerHTML=html;
  });
}
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    result = analyze_text(data.get("text", ""))
    return jsonify(result)

@app.route("/model-info")
def model_info():
    texts = [t for t, _ in TRAINING_DATA]
    labels = [l for _, l in TRAINING_DATA]
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
    y_pred = MODEL.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Clean", "Dark Pattern"], output_dict=True)
    return jsonify({"accuracy": f"{acc:.2%}", "report": report, "training_samples": len(TRAINING_DATA)})

if __name__ == "__main__":
    print("🚀 ML Dark Pattern Analyzer running at http://localhost:5002")
    app.run(debug=True, port=5002)
