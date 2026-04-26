"""
Project 4: Advanced Dark Pattern Detection System
Difficulty: HARD
Tech: Python + Flask + scikit-learn + Multiple ML Models + Charts + REST API
Run: pip install flask scikit-learn numpy pandas matplotlib seaborn && python app.py
"""

from flask import Flask, render_template_string, request, jsonify
import numpy as np
import json
import re
import pickle
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, accuracy_score,
                              confusion_matrix, precision_recall_fscore_support)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ── Extended Dataset (8 categories) ───────────────────────────────────────────
DATASET = [
    # (text, category)
    # SCARCITY
    ("Only 3 left in stock!", "scarcity"),
    ("Hurry! Offer expires in 10 minutes.", "scarcity"),
    ("847 people viewing this right now.", "scarcity"),
    ("Limited time offer — ends tonight!", "scarcity"),
    ("Act now! Last 2 seats available.", "scarcity"),
    ("Flash sale ends in 00:04:23!", "scarcity"),
    ("Only today at this price!", "scarcity"),
    ("Selling fast — 95% claimed already.", "scarcity"),
    ("Your cart expires in 5 minutes.", "scarcity"),
    ("Book now — prices increase tomorrow!", "scarcity"),
    # CONFIRM SHAMING
    ("No thanks, I hate saving money.", "confirm_shaming"),
    ("No, I prefer to pay full price.", "confirm_shaming"),
    ("I'm fine missing this exclusive deal.", "confirm_shaming"),
    ("No thanks, I don't want free shipping.", "confirm_shaming"),
    ("No, I don't want to improve my life.", "confirm_shaming"),
    ("Skip this — I hate being healthy.", "confirm_shaming"),
    ("No, I enjoy paying more.", "confirm_shaming"),
    ("I'll pass on the savings.", "confirm_shaming"),
    # FORCED CONTINUITY
    ("Free trial — auto-renews at ₹999/month.", "forced_continuity"),
    ("We will charge your card after 7 days unless cancelled.", "forced_continuity"),
    ("Your subscription continues automatically after the trial.", "forced_continuity"),
    ("Auto-renewal enabled. Cancel before period ends.", "forced_continuity"),
    ("Trial converts to paid plan on day 30.", "forced_continuity"),
    ("Your plan auto-renews annually until cancelled.", "forced_continuity"),
    ("Credit card required for free trial.", "forced_continuity"),
    ("Billing begins automatically after free period.", "forced_continuity"),
    # HIDDEN COSTS
    ("Price excludes booking fee, service fee, and GST.", "hidden_costs"),
    ("Additional charges apply at checkout.", "hidden_costs"),
    ("Taxes and fees not included in displayed price.", "hidden_costs"),
    ("Final price shown at checkout only.", "hidden_costs"),
    ("Processing fee added at payment step.", "hidden_costs"),
    ("Shown price does not include mandatory resort fee.", "hidden_costs"),
    ("Luggage and seat selection billed separately.", "hidden_costs"),
    ("Price may vary. See checkout for total.", "hidden_costs"),
    # ROACH MOTEL
    ("Sign up in one click. Cancel by calling helpdesk.", "roach_motel"),
    ("Joining is instant. Deletion requires 30-day notice.", "roach_motel"),
    ("Easy registration. Account removal: contact support.", "roach_motel"),
    ("Subscribe now. Cancellation requires written request.", "roach_motel"),
    ("Join in seconds. To leave, fill the 5-page exit form.", "roach_motel"),
    ("One-click signup. Call us to cancel.", "roach_motel"),
    # MISDIRECTION
    ("Recommended plan is pre-selected for you.", "misdirection"),
    ("Best value option already chosen.", "misdirection"),
    ("Our most popular plan is highlighted and pre-filled.", "misdirection"),
    ("Accept all is the default and primary button.", "misdirection"),
    ("Upgrade is displayed prominently; downgrade is hidden.", "misdirection"),
    ("Continue with premium selected by default.", "misdirection"),
    # PRIVACY ZUCKERING
    ("We share your data with advertising partners by default.", "privacy"),
    ("By continuing you consent to all data collection.", "privacy"),
    ("Opt out of data sharing requires contacting us by mail.", "privacy"),
    ("Your browsing data is automatically shared with third parties.", "privacy"),
    ("Data sharing is enabled unless you disable it in advanced settings.", "privacy"),
    ("We may sell your data to improve our services.", "privacy"),
    ("Your activity is tracked across all our partner sites.", "privacy"),
    # CLEAN
    ("Browse our complete catalog at your own pace.", "clean"),
    ("All prices shown include taxes and fees.", "clean"),
    ("Cancel your subscription anytime in two clicks.", "clean"),
    ("We will never share your data without consent.", "clean"),
    ("Your order will arrive in 3 to 5 business days.", "clean"),
    ("Free returns within 30 days, no questions asked.", "clean"),
    ("Manage your privacy settings easily from your profile.", "clean"),
    ("We remind you 7 days before your trial ends.", "clean"),
    ("Compare all plans before making a decision.", "clean"),
    ("No credit card required for the free tier.", "clean"),
    ("Delete your account instantly from account settings.", "clean"),
    ("Our customer service is available 24/7 for help.", "clean"),
    ("All fees displayed upfront before you commit.", "clean"),
    ("Your data is encrypted and never sold.", "clean"),
    ("Opt out of emails with one click at the bottom.", "clean"),
]

CATEGORIES = ["scarcity", "confirm_shaming", "forced_continuity",
               "hidden_costs", "roach_motel", "misdirection", "privacy", "clean"]

CATEGORY_META = {
    "scarcity":          {"label": "Scarcity Pressure",    "color": "#e74c3c", "icon": "⏰"},
    "confirm_shaming":   {"label": "Confirm Shaming",      "color": "#9b59b6", "icon": "😔"},
    "forced_continuity": {"label": "Forced Continuity",    "color": "#e67e22", "icon": "🔄"},
    "hidden_costs":      {"label": "Hidden Costs",         "color": "#c0392b", "icon": "💰"},
    "roach_motel":       {"label": "Roach Motel",          "color": "#2c3e50", "icon": "🪤"},
    "misdirection":      {"label": "Misdirection",         "color": "#2980b9", "icon": "👁️"},
    "privacy":           {"label": "Privacy Manipulation", "color": "#16a085", "icon": "🔒"},
    "clean":             {"label": "Clean / Honest",       "color": "#27ae60", "icon": "✅"},
}

MODEL_CONFIGS = {
    "Logistic Regression": Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,3), max_features=5000, sublinear_tf=True)),
        ('clf', LogisticRegression(max_iter=1000, C=2.0, random_state=42))
    ]),
    "Naive Bayes": Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=4000)),
        ('clf', MultinomialNB(alpha=0.5))
    ]),
    "Linear SVM": Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,3), max_features=5000, sublinear_tf=True)),
        ('clf', LinearSVC(max_iter=2000, C=1.5, random_state=42))
    ]),
    "Random Forest": Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=3000)),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ]),
}

TRAINED_MODELS = {}
MODEL_METRICS = {}

def train_all_models():
    global TRAINED_MODELS, MODEL_METRICS
    texts  = [t for t, _ in DATASET]
    labels = [l for _, l in DATASET]
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels)

    for name, pipeline in MODEL_CONFIGS.items():
        t0 = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - t0
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
        cv_scores = cross_val_score(pipeline, texts, labels, cv=3, scoring='accuracy')
        TRAINED_MODELS[name] = pipeline
        MODEL_METRICS[name] = {
            "accuracy":    round(acc * 100, 1),
            "precision":   round(p * 100, 1),
            "recall":      round(r * 100, 1),
            "f1":          round(f1 * 100, 1),
            "cv_mean":     round(cv_scores.mean() * 100, 1),
            "cv_std":      round(cv_scores.std() * 100, 2),
            "train_time":  round(train_time * 1000, 1),
        }
        print(f"  {name}: Acc={acc:.2%}  F1={f1:.2%}  Time={train_time*1000:.0f}ms")

    best = max(MODEL_METRICS, key=lambda k: MODEL_METRICS[k]["f1"])
    print(f"✅ Best model: {best}")
    return best

print("Training all models...")
BEST_MODEL_NAME = train_all_models()

def predict_text(text, model_name=None):
    model_name = model_name or BEST_MODEL_NAME
    model = TRAINED_MODELS[model_name]
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    results = []
    for sent in sentences:
        pred = model.predict([sent])[0]
        meta = CATEGORY_META[pred]
        # get probability if available
        try:
            probs = model.predict_proba([sent])[0]
            classes = model.classes_
            prob_dict = {c: round(float(p)*100, 1) for c, p in zip(classes, probs)}
            confidence = round(float(max(probs))*100, 1)
        except:
            prob_dict = {}
            confidence = 90.0
        results.append({
            "sentence": sent,
            "category": pred,
            "label": meta["label"],
            "color": meta["color"],
            "icon": meta["icon"],
            "confidence": confidence,
            "probs": prob_dict,
            "is_dark": pred != "clean",
        })
    dark_count = sum(1 for r in results if r["is_dark"])
    score = round(dark_count / len(results) * 100) if results else 0
    cats_found = {}
    for r in results:
        if r["is_dark"]:
            cats_found[r["label"]] = cats_found.get(r["label"], 0) + 1
    return {
        "sentences": results,
        "total": len(results),
        "dark_count": dark_count,
        "score": score,
        "risk": "High" if score > 60 else "Medium" if score > 30 else "Low",
        "categories_found": cats_found,
        "model_used": model_name,
        "model_metrics": MODEL_METRICS.get(model_name, {}),
    }

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Advanced Dark Pattern Detection System</title>
<style>
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:'Segoe UI',sans-serif;background:#0a0a14;color:#e0e0e0;min-height:100vh;}
.header{background:linear-gradient(135deg,#0d0d1a,#1a1a2e);padding:28px 40px;border-bottom:2px solid #e94560;}
.header h1{font-size:1.9rem;color:#fff;}.header p{color:#888;margin-top:5px;}
.tabs{display:flex;gap:0;background:#111;border-bottom:2px solid #222;}
.tab{padding:14px 28px;cursor:pointer;font-size:0.95rem;color:#888;border-bottom:3px solid transparent;transition:all 0.2s;}
.tab.active{color:#e94560;border-bottom-color:#e94560;background:#0d0d1a;}
.tab:hover{color:#fff;}
.container{max-width:1100px;margin:32px auto;padding:0 20px;}
.panel{display:none;}.panel.active{display:block;}
.card{background:#12121e;border:1px solid #1e1e30;border-radius:16px;padding:28px;margin-bottom:22px;}
.card h2{font-size:1.05rem;color:#ccc;margin-bottom:16px;font-weight:600;}
textarea{width:100%;background:#0a0a14;border:1.5px solid #2a2a4a;border-radius:10px;padding:16px;
  font-size:0.95rem;color:#e0e0e0;min-height:140px;resize:vertical;outline:none;font-family:inherit;}
textarea:focus{border-color:#e94560;}
.model-select{background:#0a0a14;border:1.5px solid #2a2a4a;border-radius:8px;padding:10px 16px;
  color:#ccc;font-size:0.95rem;width:100%;margin-top:10px;outline:none;}
.model-select:focus{border-color:#e94560;}
.btn{background:linear-gradient(135deg,#e94560,#c0392b);color:#fff;border:none;border-radius:10px;
  padding:14px 36px;font-size:1rem;font-weight:700;cursor:pointer;margin-top:14px;width:100%;}
.btn:hover{opacity:0.88;}
.sbtn{background:#1a1a2e;border:1px solid #2a2a4a;border-radius:8px;padding:7px 14px;font-size:0.82rem;
  cursor:pointer;color:#aaa;margin:4px 4px 0 0;}
.sbtn:hover{background:#e94560;color:#fff;border-color:#e94560;}
.metrics-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:14px;margin-bottom:22px;}
.metric{background:#12121e;border:1px solid #1e1e30;border-radius:12px;padding:20px;text-align:center;}
.metric-val{font-size:2.2rem;font-weight:900;}
.metric-label{color:#666;font-size:0.8rem;margin-top:4px;}
.sent-item{border-radius:10px;padding:14px 18px;margin-bottom:9px;border-left:5px solid;display:flex;align-items:flex-start;gap:12px;}
.sent-text{flex:1;font-size:0.93rem;line-height:1.5;}
.sent-badge{font-size:0.72rem;font-weight:700;border-radius:20px;padding:3px 12px;white-space:nowrap;color:#fff;}
.conf-bar{height:4px;border-radius:3px;background:#1e1e30;margin-top:6px;}
.conf-fill{height:100%;border-radius:3px;}
.comp-table{width:100%;border-collapse:collapse;font-size:0.92rem;}
.comp-table th{padding:12px 14px;text-align:left;color:#888;font-weight:600;border-bottom:2px solid #1e1e30;}
.comp-table td{padding:12px 14px;border-bottom:1px solid #1a1a2e;color:#ccc;}
.comp-table tr.best td{color:#fff;background:#0d1a0d;}
.comp-table tr:hover td{background:#1a1a2e;}
.badge-best{background:#27ae60;color:#fff;border-radius:12px;padding:2px 10px;font-size:0.72rem;}
.cats-chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px;}
.chip{border-radius:20px;padding:5px 14px;font-size:0.82rem;font-weight:600;color:#fff;}
</style>
</head>
<body>
<div class="header">
  <h1>⚡ Advanced Dark Pattern Detection System</h1>
  <p>Project 4 — Hard | AIML | Piyush Mishra — SAP 590024892 | Multi-Model Comparison + 8-Class Classification</p>
</div>
<div class="tabs">
  <div class="tab active" onclick="switchTab('analyze')">🔍 Analyze Text</div>
  <div class="tab" onclick="switchTab('compare')">📊 Model Comparison</div>
  <div class="tab" onclick="switchTab('about')">ℹ️ System Info</div>
</div>
<div class="container">

<!-- ANALYZE PANEL -->
<div class="panel active" id="panel-analyze">
  <div class="card">
    <h2>Input Text for Analysis</h2>
    <textarea id="inp" placeholder="Paste website copy, email, UI text..."></textarea>
    <div style="margin-top:10px;">
      <label style="color:#888;font-size:0.85rem;">Select Model:</label>
      <select class="model-select" id="modelSel">
        <option value="auto">Auto (Best Model)</option>
        <option value="Logistic Regression">Logistic Regression</option>
        <option value="Naive Bayes">Naive Bayes</option>
        <option value="Linear SVM">Linear SVM</option>
        <option value="Random Forest">Random Forest</option>
      </select>
    </div>
    <div style="margin-top:12px;">
      <button class="sbtn" onclick="ls(0)">High Risk</button>
      <button class="sbtn" onclick="ls(1)">Mixed</button>
      <button class="sbtn" onclick="ls(2)">Clean</button>
    </div>
    <button class="btn" onclick="run()">⚡ Run Analysis</button>
  </div>
  <div id="results"></div>
</div>

<!-- COMPARE PANEL -->
<div class="panel" id="panel-compare">
  <div class="card">
    <h2>Model Performance Comparison</h2>
    <table class="comp-table" id="compTable"></table>
  </div>
</div>

<!-- ABOUT PANEL -->
<div class="panel" id="panel-about">
  <div class="card">
    <h2>System Architecture</h2>
    <p style="color:#aaa;line-height:1.8;font-size:0.95rem;">
      This system implements <strong style="color:#e94560">multi-class dark pattern classification</strong> across 8 categories
      using 4 ML models trained on a curated dataset.<br><br>
      <strong style="color:#ccc">Pipeline:</strong> Raw Text → Sentence Splitting → TF-IDF Vectorization (n-gram 1–3) → Classifier → Category + Confidence<br><br>
      <strong style="color:#ccc">Models:</strong> Logistic Regression, Naive Bayes, Linear SVM, Random Forest<br>
      <strong style="color:#ccc">Dataset:</strong> 70 labeled examples across 8 categories<br>
      <strong style="color:#ccc">Evaluation:</strong> Train/Test split (75/25) + 3-fold cross-validation<br><br>
      <strong style="color:#ccc">8 Categories:</strong> Scarcity Pressure · Confirm Shaming · Forced Continuity ·
      Hidden Costs · Roach Motel · Misdirection · Privacy Manipulation · Clean
    </p>
  </div>
</div>
</div>

<script>
const samples=[
  "Only 2 seats left — act now! This offer expires in 04:32. 500 people are viewing this. No thanks, I hate saving money. Your trial auto-renews at ₹1999/month. Price shown excludes booking fee.",
  "Welcome to our platform. Only 3 rooms left at this price! All our prices include taxes. Your free trial converts automatically after 14 days. Cancel anytime easily from your dashboard. Thank you.",
  "Browse our catalog freely. All prices include all fees. Cancel subscription instantly in Settings. We will never sell your data. Free returns within 30 days. Our support team is here for you."
];
function ls(i){document.getElementById('inp').value=samples[i];}
function switchTab(t){
  document.querySelectorAll('.tab').forEach((el,i)=>el.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(el=>el.classList.remove('active'));
  document.getElementById('panel-'+t).classList.add('active');
  event.target.classList.add('active');
  if(t==='compare') loadCompare();
}
function loadCompare(){
  fetch('/model-metrics').then(r=>r.json()).then(d=>{
    const best=Object.entries(d).sort((a,b)=>b[1].f1-a[1].f1)[0][0];
    let html=`<tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>CV Mean</th><th>Train Time</th></tr>`;
    Object.entries(d).sort((a,b)=>b[1].f1-a[1].f1).forEach(([name,m])=>{
      const isBest=name===best;
      html+=`<tr class="${isBest?'best':''}">
        <td>${name} ${isBest?'<span class="badge-best">BEST</span>':''}</td>
        <td>${m.accuracy}%</td><td>${m.precision}%</td><td>${m.recall}%</td>
        <td><strong>${m.f1}%</strong></td><td>${m.cv_mean}±${m.cv_std}%</td><td>${m.train_time}ms</td></tr>`;
    });
    document.getElementById('compTable').innerHTML=html;
  });
}
function run(){
  const t=document.getElementById('inp').value.trim();
  const m=document.getElementById('modelSel').value;
  if(!t){alert('Enter text first');return;}
  fetch('/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:t,model:m})})
  .then(r=>r.json()).then(d=>{
    const R=document.getElementById('results'); R.style.display='block';
    const rCol=d.risk==='High'?'#e74c3c':d.risk==='Medium'?'#e67e22':'#27ae60';
    let html=`<div class="metrics-grid">
      <div class="metric"><div class="metric-val" style="color:${rCol}">${d.score}%</div><div class="metric-label">Dark Score</div></div>
      <div class="metric"><div class="metric-val" style="color:${rCol}">${d.risk}</div><div class="metric-label">Risk Level</div></div>
      <div class="metric"><div class="metric-val" style="color:#e74c3c">${d.dark_count}</div><div class="metric-label">Dark Sentences</div></div>
      <div class="metric"><div class="metric-val" style="color:#27ae60">${d.total-d.dark_count}</div><div class="metric-label">Clean Sentences</div></div>
      <div class="metric"><div class="metric-val" style="color:#2980b9">${d.model_used.split(' ')[0]}</div><div class="metric-label">Model Used</div></div>
    </div>`;
    if(Object.keys(d.categories_found).length>0){
      html+=`<div class="card"><h2>Dark Pattern Types Found:</h2><div class="cats-chips">`;
      Object.entries(d.categories_found).forEach(([cat,cnt])=>{
        html+=`<span class="chip" style="background:#e94560">${cat} ×${cnt}</span>`;
      });
      html+=`</div></div>`;
    }
    html+=`<div class="card"><h2>Sentence-level Results (Model: ${d.model_used}):</h2>`;
    d.sentences.forEach(s=>{
      html+=`<div class="sent-item" style="background:${s.color}18;border-color:${s.color}">
        <div class="sent-text">${s.icon} <strong style="color:${s.color}">${s.label}</strong><br>"${s.sentence}"
          <div class="conf-bar"><div class="conf-fill" style="width:${s.confidence}%;background:${s.color}"></div></div>
          <small style="color:#666">${s.confidence}% confidence</small>
        </div>
        <span class="sent-badge" style="background:${s.color}">${s.is_dark?'⚠️ Dark':'✅ Clean'}</span>
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
    model = data.get("model", "auto")
    if model == "auto":
        model = BEST_MODEL_NAME
    result = predict_text(data.get("text", ""), model)
    return jsonify(result)

@app.route("/model-metrics")
def metrics():
    return jsonify(MODEL_METRICS)

@app.route("/dataset-stats")
def dataset_stats():
    from collections import Counter
    counts = Counter(l for _, l in DATASET)
    return jsonify({"total": len(DATASET), "by_category": dict(counts)})

if __name__ == "__main__":
    print(f"\n🚀 Advanced Dark Pattern System running at http://localhost:5003")
    print(f"   Best model: {BEST_MODEL_NAME}")
    app.run(debug=True, port=5003)
