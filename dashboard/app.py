from flask import Flask, render_template, request, jsonify
import re, json
from datetime import datetime
from collections import defaultdict

app = Flask(__name__)

# Reuse detection logic from easy1
DARK_PATTERNS = {
    "Scarcity Pressure":     {"keywords":["only left","limited stock","hurry","almost gone","selling fast","few left","last chance"],"color":"#e74c3c","icon":"⚠️"},
    "Urgency / Countdown":   {"keywords":["offer expires","today only","ends in","limited time","deal ends","expires soon","act now","time is running out"],"color":"#e67e22","icon":"⏰"},
    "Confirm Shaming":       {"keywords":["no thanks i don't","no i don't want","i prefer to pay more","no thanks i hate","i don't want to save"],"color":"#9b59b6","icon":"😢"},
    "Hidden Costs":          {"keywords":["service fee","booking fee","processing fee","convenience fee","additional charge","taxes not included","fees apply"],"color":"#c0392b","icon":"💸"},
    "Forced Continuity":     {"keywords":["free trial","auto renew","automatically charged","cancel anytime","recurring charge","subscription begins"],"color":"#2980b9","icon":"🔄"},
    "Misdirection":          {"keywords":["recommended","most popular","best value","our pick","selected for you","pre-selected"],"color":"#16a085","icon":"👁️"},
    "Roach Motel":           {"keywords":["call to cancel","contact support to cancel","email to cancel","cancellation requires","to unsubscribe call"],"color":"#8e44ad","icon":"🪤"},
    "Social Proof Manip.":   {"keywords":["people are viewing","others are looking","someone just bought","trending now","high demand"],"color":"#d35400","icon":"👥"},
}

# In-memory analytics store
analytics = {
    "total_analyses": 0,
    "pattern_counts": defaultdict(int),
    "risk_counts": {"Low":0,"Medium":0,"High":0},
    "recent": [],  # last 10 analyses
    "scores": []
}

def detect_patterns(text):
    text_lower = text.lower()
    found = []
    for name, data in DARK_PATTERNS.items():
        matched = [kw for kw in data["keywords"] if kw in text_lower]
        if matched:
            found.append({"pattern":name,"color":data["color"],"icon":data["icon"],
                          "matched":matched,"severity":"High" if len(matched)>=2 else "Medium"})
    return found

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('text','').strip()
    if not text: return jsonify({'error':'No text'}),400
    findings = detect_patterns(text)
    score = min(100, len(findings)*20)
    risk = "Low" if score<20 else "Medium" if score<60 else "High"

    # Update analytics
    analytics['total_analyses'] += 1
    analytics['risk_counts'][risk] += 1
    analytics['scores'].append(score)
    for f in findings:
        analytics['pattern_counts'][f['pattern']] += 1
    analytics['recent'].insert(0, {
        "text": text[:80]+('...' if len(text)>80 else ''),
        "score": score, "risk": risk, "count": len(findings),
        "time": datetime.now().strftime('%H:%M:%S')
    })
    analytics['recent'] = analytics['recent'][:10]

    highlighted = text
    for f in findings:
        for kw in f['matched']:
            pat = re.compile(re.escape(kw), re.IGNORECASE)
            highlighted = pat.sub(f'<mark style="background:{f["color"]};color:white;padding:2px 4px;border-radius:3px;">{kw}</mark>', highlighted)

    return jsonify({'findings':findings,'highlighted':highlighted,'score':score,'risk':risk,'count':len(findings)})

@app.route('/analytics')
def get_analytics():
    avg_score = round(sum(analytics['scores'])/max(len(analytics['scores']),1),1)
    top_patterns = sorted(analytics['pattern_counts'].items(), key=lambda x:-x[1])[:5]
    return jsonify({
        'total': analytics['total_analyses'],
        'risk_counts': analytics['risk_counts'],
        'avg_score': avg_score,
        'top_patterns': [{"pattern":p,"count":c} for p,c in top_patterns],
        'recent': analytics['recent'],
        'scores': analytics['scores'][-20:]
    })

if __name__ == '__main__':
    app.run(debug=True, port=5004)
