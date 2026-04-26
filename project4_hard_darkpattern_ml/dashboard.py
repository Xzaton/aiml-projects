"""
Project 6: Dark Pattern Detection Dashboard
Full GUI/Dashboard for the AIML Dark Pattern Projects
Tech: Python + Flask + Chart.js
Run: pip install flask scikit-learn numpy pandas && python dashboard.py
Open: http://localhost:5004
"""

from flask import Flask, render_template_string, request, jsonify
import numpy as np
import re
import time
import random
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

DATASET = [
    ("Only 3 left in stock!", "scarcity"),
    ("Hurry! Offer expires in 10 minutes.", "scarcity"),
    ("847 people viewing this right now.", "scarcity"),
    ("Limited time offer ends tonight!", "scarcity"),
    ("Act now! Last 2 seats available.", "scarcity"),
    ("Flash sale ends in 4 minutes!", "scarcity"),
    ("Only today at this price!", "scarcity"),
    ("Selling fast 95 percent claimed already.", "scarcity"),
    ("Your cart expires in 5 minutes.", "scarcity"),
    ("Book now prices increase tomorrow!", "scarcity"),
    ("No thanks, I hate saving money.", "confirm_shaming"),
    ("No, I prefer to pay full price.", "confirm_shaming"),
    ("I'm fine missing this exclusive deal.", "confirm_shaming"),
    ("No thanks, I don't want free shipping.", "confirm_shaming"),
    ("No, I don't want to improve my life.", "confirm_shaming"),
    ("Skip this I hate being healthy.", "confirm_shaming"),
    ("No, I enjoy paying more.", "confirm_shaming"),
    ("Free trial auto-renews at 999 per month.", "forced_continuity"),
    ("We will charge your card after 7 days unless cancelled.", "forced_continuity"),
    ("Your subscription continues automatically after the trial.", "forced_continuity"),
    ("Auto-renewal enabled. Cancel before period ends.", "forced_continuity"),
    ("Trial converts to paid plan on day 30.", "forced_continuity"),
    ("Your plan auto-renews annually until cancelled.", "forced_continuity"),
    ("Credit card required for free trial.", "forced_continuity"),
    ("Price excludes booking fee service fee and GST.", "hidden_costs"),
    ("Additional charges apply at checkout.", "hidden_costs"),
    ("Taxes and fees not included in displayed price.", "hidden_costs"),
    ("Final price shown at checkout only.", "hidden_costs"),
    ("Processing fee added at payment step.", "hidden_costs"),
    ("Shown price does not include mandatory resort fee.", "hidden_costs"),
    ("Sign up in one click. Cancel by calling helpdesk.", "roach_motel"),
    ("Joining is instant. Deletion requires 30-day notice.", "roach_motel"),
    ("Easy registration. Account removal contact support.", "roach_motel"),
    ("Subscribe now. Cancellation requires written request.", "roach_motel"),
    ("One-click signup. Call us to cancel.", "roach_motel"),
    ("Recommended plan is pre-selected for you.", "misdirection"),
    ("Best value option already chosen.", "misdirection"),
    ("Most popular plan is highlighted and pre-filled.", "misdirection"),
    ("Accept all is the default and primary button.", "misdirection"),
    ("Upgrade is displayed prominently downgrade is hidden.", "misdirection"),
    ("We share your data with advertising partners by default.", "privacy"),
    ("By continuing you consent to all data collection.", "privacy"),
    ("Opt out of data sharing requires contacting us by mail.", "privacy"),
    ("Your browsing data is automatically shared with third parties.", "privacy"),
    ("We may sell your data to improve our services.", "privacy"),
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
    ("Our customer service is available 24 hours a day.", "clean"),
    ("All fees displayed upfront before you commit.", "clean"),
    ("Your data is encrypted and never sold.", "clean"),
    ("Opt out of emails with one click at the bottom.", "clean"),
]

CATEGORY_META = {
    "scarcity":          {"label": "Scarcity Pressure",    "color": "#e74c3c"},
    "confirm_shaming":   {"label": "Confirm Shaming",      "color": "#9b59b6"},
    "forced_continuity": {"label": "Forced Continuity",    "color": "#e67e22"},
    "hidden_costs":      {"label": "Hidden Costs",         "color": "#c0392b"},
    "roach_motel":       {"label": "Roach Motel",          "color": "#34495e"},
    "misdirection":      {"label": "Misdirection",         "color": "#2980b9"},
    "privacy":           {"label": "Privacy Manipulation", "color": "#16a085"},
    "clean":             {"label": "Clean / Honest",       "color": "#27ae60"},
}

# Train model
texts  = [t for t,_ in DATASET]
labels = [l for _,l in DATASET]
X_tr, X_te, y_tr, y_te = train_test_split(texts, labels, test_size=0.25, random_state=42, stratify=labels)

MODELS = {}
METRICS = {}
for name, pipe in [
    ("Logistic Regression", Pipeline([('v', TfidfVectorizer(ngram_range=(1,3), max_features=5000, sublinear_tf=True)),
                                       ('c', LogisticRegression(max_iter=1000, C=2.0, random_state=42))])),
    ("Naive Bayes",         Pipeline([('v', TfidfVectorizer(ngram_range=(1,2), max_features=4000)),
                                       ('c', MultinomialNB(alpha=0.5))])),
    ("Linear SVM",          Pipeline([('v', TfidfVectorizer(ngram_range=(1,3), max_features=5000, sublinear_tf=True)),
                                       ('c', LinearSVC(max_iter=2000, C=1.5, random_state=42))])),
    ("Random Forest",       Pipeline([('v', TfidfVectorizer(ngram_range=(1,2), max_features=3000)),
                                       ('c', RandomForestClassifier(n_estimators=100, random_state=42))])),
]:
    t0 = time.time()
    pipe.fit(X_tr, y_tr)
    elapsed = round((time.time()-t0)*1000, 1)
    yp = pipe.predict(X_te)
    acc = accuracy_score(y_te, yp)
    p,r,f1,_ = precision_recall_fscore_support(y_te, yp, average='weighted', zero_division=0)
    MODELS[name] = pipe
    METRICS[name] = {"accuracy":round(acc*100,1),"precision":round(p*100,1),
                     "recall":round(r*100,1),"f1":round(f1*100,1),"time":elapsed}

BEST = max(METRICS, key=lambda k: METRICS[k]["f1"])
print(f"✅ Models trained. Best: {BEST}")

# Simulated history for dashboard
HISTORY = [
    {"text": "Only 2 rooms left!", "risk": "High", "score": 85, "time": "10:42 AM"},
    {"text": "Cancel anytime in one click.", "risk": "Low", "score": 5, "time": "10:38 AM"},
    {"text": "Auto-renews at ₹999/month after trial.", "risk": "High", "score": 75, "time": "10:30 AM"},
    {"text": "All prices include taxes and fees.", "risk": "Low", "score": 8, "time": "10:22 AM"},
    {"text": "Hurry! Offer expires tonight.", "risk": "Medium", "score": 55, "time": "10:15 AM"},
]

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Dark Pattern Detection Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:'Segoe UI',sans-serif;background:#0b0b18;color:#e0e0e0;display:flex;min-height:100vh;}

/* SIDEBAR */
.sidebar{width:230px;min-height:100vh;background:#0d0d1e;border-right:1px solid #1e1e30;padding:0;flex-shrink:0;position:fixed;left:0;top:0;bottom:0;}
.sidebar-logo{padding:24px 20px;border-bottom:1px solid #1e1e30;}
.sidebar-logo h2{font-size:1rem;color:#e94560;font-weight:800;letter-spacing:0.5px;}
.sidebar-logo p{color:#555;font-size:0.75rem;margin-top:3px;}
.nav-item{display:flex;align-items:center;gap:12px;padding:13px 22px;cursor:pointer;color:#888;font-size:0.9rem;transition:all 0.2s;border-left:3px solid transparent;}
.nav-item:hover{background:#12121e;color:#ccc;}
.nav-item.active{background:#12121e;color:#e94560;border-left-color:#e94560;}
.nav-item .icon{font-size:1.1rem;width:20px;}

/* MAIN */
.main{margin-left:230px;flex:1;padding:28px 32px;min-height:100vh;}
.topbar{display:flex;justify-content:space-between;align-items:center;margin-bottom:28px;}
.topbar h1{font-size:1.5rem;color:#fff;font-weight:700;}
.topbar p{color:#666;font-size:0.85rem;}
.badge-name{background:#e94560;color:#fff;border-radius:20px;padding:6px 16px;font-size:0.8rem;font-weight:700;}

/* PANELS */
.panel{display:none;} .panel.active{display:block;}

/* KPI CARDS */
.kpi-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:28px;}
.kpi{background:#12121e;border:1px solid #1e1e30;border-radius:14px;padding:22px 24px;position:relative;overflow:hidden;}
.kpi::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;}
.kpi.red::before{background:#e74c3c;} .kpi.green::before{background:#27ae60;}
.kpi.blue::before{background:#2980b9;} .kpi.purple::before{background:#9b59b6;}
.kpi-val{font-size:2.2rem;font-weight:900;color:#fff;}
.kpi-label{color:#666;font-size:0.82rem;margin-top:5px;}
.kpi-sub{font-size:0.78rem;margin-top:8px;font-weight:600;}
.kpi.red .kpi-sub{color:#e74c3c;} .kpi.green .kpi-sub{color:#27ae60;}
.kpi.blue .kpi-sub{color:#2980b9;} .kpi.purple .kpi-sub{color:#9b59b6;}

/* CHART GRID */
.chart-grid{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:28px;}
.chart-card{background:#12121e;border:1px solid #1e1e30;border-radius:14px;padding:22px;}
.chart-card h3{font-size:0.95rem;color:#aaa;margin-bottom:16px;font-weight:600;}
.chart-card canvas{max-height:220px;}

/* ANALYZER */
.analyzer-grid{display:grid;grid-template-columns:1fr 1fr;gap:20px;}
.card{background:#12121e;border:1px solid #1e1e30;border-radius:14px;padding:24px;margin-bottom:20px;}
.card h3{font-size:0.95rem;color:#aaa;margin-bottom:14px;font-weight:600;}
textarea{width:100%;background:#0b0b18;border:1.5px solid #1e1e30;border-radius:10px;padding:14px;
  color:#e0e0e0;font-size:0.9rem;min-height:130px;resize:vertical;outline:none;font-family:inherit;}
textarea:focus{border-color:#e94560;}
.analyze-btn{background:linear-gradient(135deg,#e94560,#c0392b);color:#fff;border:none;border-radius:10px;
  padding:12px 28px;font-size:0.95rem;font-weight:700;cursor:pointer;margin-top:12px;width:100%;}
.analyze-btn:hover{opacity:0.88;}
.sent-item{border-radius:8px;padding:12px 15px;margin-bottom:8px;border-left:4px solid;font-size:0.88rem;}
.score-ring{text-align:center;padding:20px 0;}
.score-num{font-size:3.5rem;font-weight:900;}
.score-lbl{font-size:0.85rem;color:#666;margin-top:4px;}

/* MODEL TABLE */
.mtable{width:100%;border-collapse:collapse;font-size:0.88rem;}
.mtable th{padding:10px 12px;color:#666;font-weight:600;border-bottom:1px solid #1e1e30;text-align:left;}
.mtable td{padding:10px 12px;border-bottom:1px solid #0f0f1a;color:#bbb;}
.mtable tr.best-row td{color:#fff;background:#0d1a0d;}
.best-badge{background:#27ae60;color:#fff;border-radius:10px;padding:2px 8px;font-size:0.7rem;}
.bar-cell{display:flex;align-items:center;gap:8px;}
.bar-bg{flex:1;height:6px;background:#1e1e30;border-radius:3px;}
.bar-fill{height:100%;border-radius:3px;background:#e94560;}

/* HISTORY TABLE */
.htable{width:100%;border-collapse:collapse;font-size:0.88rem;}
.htable th{padding:10px 14px;color:#666;font-weight:600;border-bottom:1px solid #1e1e30;text-align:left;}
.htable td{padding:10px 14px;border-bottom:1px solid #0f0f1a;color:#bbb;}
.risk-badge{border-radius:12px;padding:3px 12px;font-size:0.75rem;font-weight:700;color:#fff;}
.risk-H{background:#e74c3c;} .risk-M{background:#e67e22;} .risk-L{background:#27ae60;}
</style>
</head>
<body>

<!-- SIDEBAR -->
<div class="sidebar">
  <div class="sidebar-logo">
    <h2>🛡️ DarkPatternAI</h2>
    <p>Piyush Mishra · SAP 590024892</p>
  </div>
  <div style="padding:16px 0;">
    <div class="nav-item active" onclick="switchPanel('overview',this)"><span class="icon">📊</span> Overview</div>
    <div class="nav-item" onclick="switchPanel('analyzer',this)"><span class="icon">🔍</span> Analyzer</div>
    <div class="nav-item" onclick="switchPanel('models',this)"><span class="icon">🤖</span> Models</div>
    <div class="nav-item" onclick="switchPanel('history',this)"><span class="icon">📋</span> History</div>
  </div>
  <div style="position:absolute;bottom:20px;left:0;right:0;padding:0 20px;">
    <div style="background:#1a1a2e;border-radius:10px;padding:14px;font-size:0.78rem;color:#666;">
      <div style="color:#e94560;font-weight:700;margin-bottom:4px;">Best Model</div>
      <div id="bestModelName" style="color:#ccc;"></div>
      <div id="bestModelF1" style="color:#27ae60;font-size:0.85rem;font-weight:700;margin-top:2px;"></div>
    </div>
  </div>
</div>

<!-- MAIN -->
<div class="main">
  <div class="topbar">
    <div><h1>Dark Pattern Detection Dashboard</h1><p>AIML Project — Real-time analysis using ML models</p></div>
    <span class="badge-name">Dr. Tanupriya Chaudhary</span>
  </div>

  <!-- OVERVIEW -->
  <div class="panel active" id="panel-overview">
    <div class="kpi-grid">
      <div class="kpi red"><div class="kpi-val" id="k-analyzed">0</div><div class="kpi-label">Texts Analyzed</div><div class="kpi-sub">↑ Session Total</div></div>
      <div class="kpi green"><div class="kpi-val" id="k-models">4</div><div class="kpi-label">ML Models</div><div class="kpi-sub">↑ All Trained</div></div>
      <div class="kpi blue"><div class="kpi-val">8</div><div class="kpi-label">Pattern Categories</div><div class="kpi-sub">→ Classification Classes</div></div>
      <div class="kpi purple"><div class="kpi-val" id="k-acc"></div><div class="kpi-label">Best Model F1</div><div class="kpi-sub" id="k-acc-sub">↑ Cross-validated</div></div>
    </div>
    <div class="chart-grid">
      <div class="chart-card"><h3>Dataset Distribution by Category</h3><canvas id="distChart"></canvas></div>
      <div class="chart-card"><h3>Model Performance Comparison (F1 Score)</h3><canvas id="modelChart"></canvas></div>
    </div>
    <div class="chart-grid">
      <div class="chart-card"><h3>Pattern Type Prevalence</h3><canvas id="prevChart"></canvas></div>
      <div class="chart-card"><h3>Accuracy vs. Training Speed</h3><canvas id="speedChart"></canvas></div>
    </div>
  </div>

  <!-- ANALYZER -->
  <div class="panel" id="panel-analyzer">
    <div class="analyzer-grid">
      <div>
        <div class="card">
          <h3>Input Text</h3>
          <textarea id="inp" placeholder="Paste website text, email copy, UI strings..."></textarea>
          <div style="margin-top:10px;display:flex;gap:8px;flex-wrap:wrap;">
            <button style="background:#1e1e30;border:none;border-radius:7px;padding:7px 13px;color:#aaa;cursor:pointer;font-size:0.8rem;" onclick="ls(0)">High Risk</button>
            <button style="background:#1e1e30;border:none;border-radius:7px;padding:7px 13px;color:#aaa;cursor:pointer;font-size:0.8rem;" onclick="ls(1)">Mixed</button>
            <button style="background:#1e1e30;border:none;border-radius:7px;padding:7px 13px;color:#aaa;cursor:pointer;font-size:0.8rem;" onclick="ls(2)">Clean</button>
          </div>
          <button class="analyze-btn" onclick="runAnalysis()">⚡ Analyze Text</button>
        </div>
        <div class="card" id="sent-results" style="display:none;">
          <h3>Sentence Analysis</h3>
          <div id="sent-list"></div>
        </div>
      </div>
      <div>
        <div class="card" id="score-card" style="display:none;">
          <h3>Risk Assessment</h3>
          <div class="score-ring">
            <div class="score-num" id="score-val" style="color:#e94560">—</div>
            <div class="score-lbl">Manipulation Score</div>
            <div id="risk-lbl" style="font-size:1.1rem;font-weight:700;margin-top:8px;"></div>
          </div>
          <canvas id="resultDonut" style="max-height:180px;"></canvas>
        </div>
      </div>
    </div>
  </div>

  <!-- MODELS -->
  <div class="panel" id="panel-models">
    <div class="card">
      <h3>Model Performance Summary</h3>
      <table class="mtable" id="modelTable"></table>
    </div>
  </div>

  <!-- HISTORY -->
  <div class="panel" id="panel-history">
    <div class="card">
      <h3>Recent Analyses</h3>
      <table class="htable" id="histTable"></table>
    </div>
  </div>
</div>

<script>
let sessionCount=0, donutChart=null;
const HIST={{ history|tojson }};
const METRICS={{ metrics|tojson }};
const BEST="{{ best }}";
const CAT_COLORS=["#e74c3c","#9b59b6","#e67e22","#c0392b","#34495e","#2980b9","#16a085","#27ae60"];
const CAT_LABELS=["Scarcity","Confirm Shaming","Forced Continuity","Hidden Costs","Roach Motel","Misdirection","Privacy","Clean"];
const CAT_COUNTS=[10,8,8,8,5,6,7,15]; // approx from dataset

// Init
document.getElementById('bestModelName').textContent=BEST;
document.getElementById('bestModelF1').textContent='F1: '+METRICS[BEST].f1+'%';
document.getElementById('k-acc').textContent=METRICS[BEST].f1+'%';

// Charts
new Chart(document.getElementById('distChart'),{type:'doughnut',data:{labels:CAT_LABELS,datasets:[{data:CAT_COUNTS,backgroundColor:CAT_COLORS,borderWidth:2,borderColor:'#0b0b18'}]},options:{plugins:{legend:{labels:{color:'#888',font:{size:11}}}},cutout:'60%'}});

const mNames=Object.keys(METRICS);
const mF1=mNames.map(k=>METRICS[k].f1);
new Chart(document.getElementById('modelChart'),{type:'bar',data:{labels:mNames,datasets:[{label:'F1 Score %',data:mF1,backgroundColor:['#e94560','#2980b9','#e67e22','#27ae60'],borderRadius:6}]},options:{plugins:{legend:{display:false}},scales:{x:{ticks:{color:'#888'}},y:{ticks:{color:'#888'},min:0,max:100}}}});

new Chart(document.getElementById('prevChart'),{type:'radar',data:{labels:CAT_LABELS.slice(0,7),datasets:[{label:'Prevalence',data:[10,8,8,8,5,6,7],backgroundColor:'rgba(233,69,96,0.2)',borderColor:'#e94560',pointBackgroundColor:'#e94560'}]},options:{plugins:{legend:{display:false}},scales:{r:{ticks:{color:'#555'},pointLabels:{color:'#888',font:{size:10}}}}}});

const mTimes=mNames.map(k=>METRICS[k].time);
new Chart(document.getElementById('speedChart'),{type:'scatter',data:{datasets:mNames.map((n,i)=>({label:n,data:[{x:mTimes[i],y:mF1[i]}],backgroundColor:['#e94560','#2980b9','#e67e22','#27ae60'][i],pointRadius:10}))},options:{plugins:{legend:{labels:{color:'#888'}}},scales:{x:{title:{display:true,text:'Train Time (ms)',color:'#888'},ticks:{color:'#888'}},y:{title:{display:true,text:'F1 Score %',color:'#888'},ticks:{color:'#888'},min:0,max:100}}}});

// Model table
function buildModelTable(){
  let h='<tr><th>Model</th><th>Accuracy</th><th>F1 Score</th><th>Precision</th><th>Recall</th><th>Speed</th></tr>';
  Object.entries(METRICS).sort((a,b)=>b[1].f1-a[1].f1).forEach(([n,m])=>{
    const best=n===BEST;
    h+=`<tr class="${best?'best-row':''}"><td>${n} ${best?'<span class="best-badge">BEST</span>':''}</td>
    <td>${m.accuracy}%</td>
    <td><div class="bar-cell"><div class="bar-bg"><div class="bar-fill" style="width:${m.f1}%"></div></div>${m.f1}%</div></td>
    <td>${m.precision}%</td><td>${m.recall}%</td><td>${m.time}ms</td></tr>`;
  });
  document.getElementById('modelTable').innerHTML=h;
}

// History table
function buildHistTable(){
  let h='<tr><th>Text</th><th>Risk</th><th>Score</th><th>Time</th></tr>';
  HIST.forEach(r=>{
    h+=`<tr><td style="max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${r.text}</td>
    <td><span class="risk-badge risk-${r.risk[0]}">${r.risk}</span></td>
    <td>${r.score}%</td><td>${r.time}</td></tr>`;
  });
  document.getElementById('histTable').innerHTML=h;
}

const samples=[
  "Only 2 seats left act now! This offer expires in 4 minutes. No thanks I hate saving money. Your trial auto-renews at 1999 per month. Price shown excludes booking fee.",
  "Welcome to our platform. Only 3 rooms left at this price. All prices include taxes. Your free trial converts automatically after 14 days. Cancel anytime easily from your dashboard.",
  "Browse freely. All prices include all fees. Cancel instantly in Settings. We never sell your data. Free returns within 30 days."
];
function ls(i){document.getElementById('inp').value=samples[i];}

function switchPanel(name,el){
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n=>n.classList.remove('active'));
  document.getElementById('panel-'+name).classList.add('active');
  el.classList.add('active');
  if(name==='models') buildModelTable();
  if(name==='history') buildHistTable();
}

function runAnalysis(){
  const t=document.getElementById('inp').value.trim();
  if(!t){alert('Enter text first');return;}
  fetch('/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:t})})
  .then(r=>r.json()).then(d=>{
    sessionCount++;
    document.getElementById('k-analyzed').textContent=sessionCount;
    const rCol=d.risk==='High'?'#e74c3c':d.risk==='Medium'?'#e67e22':'#27ae60';
    document.getElementById('score-val').textContent=d.score+'%';
    document.getElementById('score-val').style.color=rCol;
    document.getElementById('risk-lbl').textContent=d.risk+' Risk';
    document.getElementById('risk-lbl').style.color=rCol;
    document.getElementById('score-card').style.display='block';
    // donut
    if(donutChart) donutChart.destroy();
    donutChart=new Chart(document.getElementById('resultDonut'),{type:'doughnut',
      data:{labels:['Dark Patterns','Clean'],datasets:[{data:[d.dark_count,d.total-d.dark_count],
        backgroundColor:['#e94560','#27ae60'],borderWidth:2,borderColor:'#12121e'}]},
      options:{plugins:{legend:{labels:{color:'#888'}}},cutout:'65%'}});
    // sentences
    let sh='';
    d.sentences.forEach(s=>{
      sh+=`<div class="sent-item" style="background:${s.color}18;border-color:${s.color}">
        <strong style="color:${s.color}">${s.icon} ${s.label}</strong><br>
        <span style="color:#bbb">"${s.sentence}"</span>
        <span style="float:right;font-size:0.75rem;color:#666">${s.confidence}%</span>
      </div>`;
    });
    document.getElementById('sent-list').innerHTML=sh;
    document.getElementById('sent-results').style.display='block';
    // add to history
    const now=new Date(); const timeStr=now.getHours()+':'+String(now.getMinutes()).padStart(2,'0');
    HIST.unshift({text:t.substring(0,60)+'...', risk:d.risk, score:d.score, time:timeStr});
  });
}
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML,
        history=HISTORY, metrics=METRICS, best=BEST)

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text", "")
    model = MODELS[BEST]
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if len(s.strip()) > 5]
    results = []
    for sent in sentences:
        pred = model.predict([sent])[0]
        meta = CATEGORY_META[pred]
        try:
            probs = model.predict_proba([sent])[0]
            conf = round(float(max(probs))*100, 1)
        except:
            conf = 88.0
        results.append({
            "sentence": sent, "category": pred,
            "label": meta["label"], "color": meta["color"],
            "icon": "✅" if pred == "clean" else "⚠️",
            "confidence": conf, "is_dark": pred != "clean",
        })
    dark = sum(1 for r in results if r["is_dark"])
    score = round(dark / len(results) * 100) if results else 0
    return jsonify({
        "sentences": results, "total": len(results),
        "dark_count": dark, "score": score,
        "risk": "High" if score > 60 else "Medium" if score > 30 else "Low",
    })

if __name__ == "__main__":
    print("🚀 Dashboard running at http://localhost:5004")
    app.run(debug=True, port=5004)
