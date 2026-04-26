"""
Project 2: Dark Pattern Text Detector
Difficulty: EASY
Tech: Python + Flask + Keyword/Rule-based NLP
Run: pip install flask && python app.py
"""

from flask import Flask, render_template_string, request, jsonify
import re

app = Flask(__name__)

# Rule-based dark pattern detector
DARK_PATTERN_RULES = {
    "Scarcity Pressure": {
        "color": "#e74c3c",
        "icon": "⏰",
        "keywords": [
            r"\bonly\s+\d+\s+(left|remaining|available|in stock)\b",
            r"\blimited\s+(time|offer|stock|seats|spots)\b",
            r"\bhurry\b", r"\bexpires?\s+(soon|in|today)\b",
            r"\blast\s+(chance|few|one)\b", r"\bact\s+now\b",
            r"\bends?\s+(today|tonight|soon)\b",
            r"\bselling\s+fast\b", r"\b\d+\s+people\s+(viewing|watching)\b",
        ],
        "explanation": "Creates artificial urgency or false scarcity to rush decisions."
    },
    "Confirm Shaming": {
        "color": "#9b59b6",
        "icon": "😔",
        "keywords": [
            r"\bno\s+thanks?,?\s+i\s+(don'?t|hate|prefer not|dislike)\b",
            r"\bi\s+don'?t\s+want\s+to\s+(save|improve|benefit|grow)\b",
            r"\bno\s+thanks?,?\s+i('?ll|'?m|'?d)\s+(pass|skip|miss out)\b",
            r"\bno\s+thanks?,?\s+i\s+hate\b",
        ],
        "explanation": "Uses guilt or shame in opt-out button labels to pressure acceptance."
    },
    "Forced Continuity": {
        "color": "#e67e22",
        "icon": "🔄",
        "keywords": [
            r"\bfree\s+trial\b.*\bcredit\s+card\b",
            r"\bauto[- ]?renew(s|al|ed)?\b",
            r"\bautomatically\s+(charge|billed?|renew)\b",
            r"\bunless\s+(you\s+cancel|cancelled?)\b",
            r"\bno\s+cancellation\s+fee\b.*\bbut\b",
            r"\bcontinues?\s+(after|until)\b",
        ],
        "explanation": "Silently converts trials to paid subscriptions without adequate warning."
    },
    "Hidden Costs": {
        "color": "#c0392b",
        "icon": "💰",
        "keywords": [
            r"\b(service|booking|processing|handling|convenience)\s+fee\b",
            r"\bexclud(ing|es?)\s+(tax|vat|gst)\b",
            r"\b\+\s*(tax|fee|charges?)\b",
            r"\badditional\s+charges?\s+(apply|may)\b",
            r"\bfees?\s+(not\s+included|apply|added)\b",
        ],
        "explanation": "Hides the real price until late in the checkout process."
    },
    "Misdirection": {
        "color": "#2980b9",
        "icon": "👁️",
        "keywords": [
            r"\brecommended\b.*\bplan\b",
            r"\bbest\s+(value|deal|choice|offer)\b.*\bselected\b",
            r"\bpre[- ]?selected\b", r"\bdefault\s+option\b",
        ],
        "explanation": "Visually emphasizes the business-preferred option to steer user choice."
    },
    "Privacy Manipulation": {
        "color": "#16a085",
        "icon": "🔒",
        "keywords": [
            r"\bshare\s+(your\s+)?data\s+with\s+(partners|third.parties)\b",
            r"\bopt[- ]?out\b.*\bdata\s+(collection|sharing|selling)\b",
            r"\bby\s+(using|continuing|signing)\b.*\byou\s+agree\b",
            r"\bwe\s+(may\s+)?(share|sell|transfer)\s+(your\s+)?data\b",
        ],
        "explanation": "Obscures data sharing practices or makes opt-out unnecessarily difficult."
    },
    "Social Proof Manipulation": {
        "color": "#8e44ad",
        "icon": "👥",
        "keywords": [
            r"\b\d{3,}\+?\s+(people|users|customers)\s+(bought|viewing|watching|joined)\b",
            r"\bmost\s+popular\b", r"\bbest\s+seller\b",
            r"\btrusted\s+by\s+\d+\b", r"\b\d+\s+reviews?\b",
        ],
        "explanation": "Uses (sometimes fabricated) social proof to pressure conformity."
    }
}

def detect_dark_patterns(text):
    text_lower = text.lower()
    findings = []
    for pattern_name, config in DARK_PATTERN_RULES.items():
        matched_snippets = []
        for keyword in config["keywords"]:
            matches = re.finditer(keyword, text_lower, re.IGNORECASE)
            for m in matches:
                start = max(0, m.start()-30)
                end = min(len(text), m.end()+30)
                snippet = text[start:end].strip()
                if snippet not in matched_snippets:
                    matched_snippets.append(snippet)
        if matched_snippets:
            findings.append({
                "name": pattern_name,
                "color": config["color"],
                "icon": config["icon"],
                "explanation": config["explanation"],
                "count": len(matched_snippets),
                "snippets": matched_snippets[:3]
            })
    score = min(100, len(findings) * 18 + sum(f["count"] * 5 for f in findings))
    return findings, score

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Dark Pattern Detector</title>
<style>
* { box-sizing: border-box; margin:0; padding:0; }
body { font-family: 'Segoe UI', sans-serif; background: #f4f6f9; min-height:100vh; }
.header { background: linear-gradient(135deg, #1a1a2e, #16213e); color:#fff; padding:24px 40px; }
.header h1 { font-size:1.8rem; } .header p { color:#aaa; margin-top:4px; }
.container { max-width:900px; margin:36px auto; padding:0 20px; }
.card { background:#fff; border-radius:16px; padding:28px; margin-bottom:24px; box-shadow:0 2px 12px rgba(0,0,0,0.07); }
.card h2 { font-size:1.1rem; color:#1a1a2e; margin-bottom:16px; font-weight:700; }
textarea { width:100%; border:2px solid #e0e0e0; border-radius:10px; padding:16px; font-size:0.95rem;
  resize:vertical; min-height:160px; font-family:inherit; transition:border 0.2s; outline:none; }
textarea:focus { border-color:#e94560; }
.samples { display:flex; gap:10px; flex-wrap:wrap; margin-top:12px; }
.sample-btn { background:#f0f0f0; border:none; border-radius:8px; padding:8px 14px; font-size:0.82rem;
  cursor:pointer; color:#444; transition:all 0.2s; }
.sample-btn:hover { background:#e94560; color:#fff; }
.analyze-btn { background:linear-gradient(135deg,#e94560,#c0392b); color:#fff; border:none; border-radius:10px;
  padding:14px 36px; font-size:1rem; font-weight:700; cursor:pointer; margin-top:16px; width:100%; transition:opacity 0.2s; }
.analyze-btn:hover { opacity:0.9; }
.results { display:none; }
.score-banner { border-radius:14px; padding:24px 28px; margin-bottom:20px; display:flex; align-items:center; gap:20px; }
.score-clean { background:#e8f8f0; border:2px solid #27ae60; }
.score-mild { background:#fff8e1; border:2px solid #f5a623; }
.score-bad { background:#fde8e8; border:2px solid #e74c3c; }
.score-num { font-size:3.2rem; font-weight:900; }
.score-clean .score-num { color:#27ae60; } .score-mild .score-num { color:#e67e22; } .score-bad .score-num { color:#e74c3c; }
.score-label { font-size:1.2rem; font-weight:700; color:#1a1a2e; } .score-sub { color:#666; margin-top:4px; }
.finding { border-radius:12px; padding:20px 22px; margin-bottom:14px; border-left:5px solid; }
.finding-title { font-size:1.05rem; font-weight:700; display:flex; align-items:center; gap:8px; }
.finding-exp { color:#555; font-size:0.9rem; margin:8px 0 12px; }
.snippet { background:rgba(0,0,0,0.06); border-radius:6px; padding:6px 12px; font-size:0.85rem;
  font-family:monospace; margin-top:6px; color:#333; }
.clean-msg { text-align:center; padding:32px; color:#27ae60; font-size:1.1rem; }
</style>
</head>
<body>
<div class="header">
  <h1>🔍 Dark Pattern Text Detector</h1>
  <p>Project 2 &nbsp;|&nbsp; AIML &nbsp;|&nbsp; Piyush Mishra — SAP 590024892</p>
</div>
<div class="container">
  <div class="card">
    <h2>Paste any website text, email, or UI copy to analyze:</h2>
    <textarea id="inputText" placeholder="e.g. 'Only 3 seats left! Hurry — this offer expires tonight. Free trial — credit card required, auto-renews at ₹999/month unless cancelled...'"></textarea>
    <div class="samples">
      <span style="color:#888;font-size:0.82rem;align-self:center">Try samples:</span>
      <button class="sample-btn" onclick="loadSample(0)">E-Commerce</button>
      <button class="sample-btn" onclick="loadSample(1)">Subscription</button>
      <button class="sample-btn" onclick="loadSample(2)">Cookie Banner</button>
      <button class="sample-btn" onclick="loadSample(3)">Clean Text</button>
    </div>
    <button class="analyze-btn" onclick="analyze()">🔍 Analyze for Dark Patterns</button>
  </div>
  <div class="results" id="results"></div>
</div>

<script>
const samples = [
  "Only 2 items left in stock! Hurry — 847 people are viewing this right now. Best seller! Trusted by 50,000+ customers. Price shown excludes booking fee and processing fee. Act now before this deal expires tonight!",
  "Start your FREE trial today! No credit card? Think again — credit card required. Your subscription auto-renews at ₹999/month automatically unless you cancel. By signing up you agree to share your data with our partners.",
  "We use cookies to enhance your experience. [Accept All Cookies] ... [Manage Preferences — takes 7 steps]. By continuing to use this site you agree to our data collection practices. We may share your data with third parties.",
  "Welcome to our store. Browse our collection of products. Add items to your cart and proceed to checkout. Our customer service team is available Monday to Friday. Thank you for shopping with us."
];

function loadSample(i) { document.getElementById('inputText').value = samples[i]; }

function analyze() {
  const text = document.getElementById('inputText').value.trim();
  if (!text) { alert('Please enter some text first.'); return; }
  fetch('/analyze', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({text})})
    .then(r=>r.json()).then(data=>{
      const r = document.getElementById('results');
      r.style.display='block';
      if (data.findings.length === 0) {
        r.innerHTML = `<div class="card"><div class="clean-msg">✅ No dark patterns detected in this text.<br><span style="color:#888;font-size:0.9rem">This text appears to be straightforward and honest.</span></div></div>`;
        return;
      }
      const sc = data.score;
      const cls = sc < 30 ? 'score-mild' : sc < 60 ? 'score-mild' : 'score-bad';
      const lbl = sc < 30 ? 'Low Risk' : sc < 60 ? 'Moderate Manipulation' : 'High Manipulation Detected';
      const sub = sc < 30 ? 'Minor concerns found' : sc < 60 ? 'Several deceptive patterns present' : 'This text is highly manipulative';
      let html = `<div class="card score-banner ${cls}">
        <div class="score-num">${sc}</div>
        <div><div class="score-label">Manipulation Score — ${lbl}</div>
        <div class="score-sub">${sub} — ${data.findings.length} pattern type(s) found</div></div></div>`;
      data.findings.forEach(f=>{
        html += `<div class="finding" style="background:${f.color}18;border-color:${f.color}">
          <div class="finding-title" style="color:${f.color}">${f.icon} ${f.name} <span style="font-size:0.78rem;background:${f.color};color:#fff;border-radius:12px;padding:2px 10px">${f.count} match(es)</span></div>
          <div class="finding-exp">${f.explanation}</div>
          ${f.snippets.map(s=>`<div class="snippet">…${s}…</div>`).join('')}
        </div>`;
      });
      r.innerHTML = html;
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
    text = data.get("text", "")
    findings, score = detect_dark_patterns(text)
    return jsonify({"findings": findings, "score": score})

if __name__ == "__main__":
    print("🚀 Dark Pattern Detector running at http://localhost:5001")
    app.run(debug=True, port=5001)
