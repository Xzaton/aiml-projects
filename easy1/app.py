from flask import Flask, render_template, request, jsonify
import re

app = Flask(__name__)

# ── Rule-based dark pattern detection ──
DARK_PATTERNS = {
    "Scarcity Pressure": {
        "keywords": ["only left", "limited stock", "hurry", "almost gone", "selling fast",
                     "only remaining", "few left", "last chance", "stock running out"],
        "color": "#e74c3c",
        "description": "Creates artificial urgency by implying limited availability."
    },
    "Urgency / Countdown": {
        "keywords": ["offer expires", "today only", "ends in", "limited time", "deal ends",
                     "expires soon", "act now", "don't miss", "time is running out"],
        "color": "#e67e22",
        "description": "Pressures users with time-limited offers to prevent careful thinking."
    },
    "Confirm Shaming": {
        "keywords": ["no thanks i don't", "no i don't want", "i prefer to pay more",
                     "no thanks i hate", "i don't want to save", "decline and miss"],
        "color": "#9b59b6",
        "description": "Uses guilt-inducing language on the reject/decline option."
    },
    "Hidden Costs": {
        "keywords": ["service fee", "booking fee", "processing fee", "convenience fee",
                     "additional charge", "taxes not included", "fees apply"],
        "color": "#c0392b",
        "description": "Conceals extra charges until late in the purchase flow."
    },
    "Forced Continuity": {
        "keywords": ["free trial", "auto renew", "automatically charged", "cancel anytime",
                     "recurring charge", "subscription begins", "billed after trial"],
        "color": "#2980b9",
        "description": "Silently converts free trials into paid subscriptions."
    },
    "Misdirection": {
        "keywords": ["recommended", "most popular", "best value", "our pick",
                     "selected for you", "pre-selected"],
        "color": "#16a085",
        "description": "Steers attention toward the company-preferred option."
    },
    "Roach Motel": {
        "keywords": ["call to cancel", "contact support to cancel", "cancellation requires",
                     "to unsubscribe call", "email to cancel", "visit store to cancel"],
        "color": "#8e44ad",
        "description": "Makes it easy to sign up but extremely difficult to cancel."
    },
    "Social Proof Manipulation": {
        "keywords": ["people are viewing", "others are looking", "someone just bought",
                     "trending now", "popular right now", "high demand"],
        "color": "#d35400",
        "description": "Uses fake or exaggerated social activity to pressure decisions."
    },
}

def detect_patterns(text):
    text_lower = text.lower()
    found = []
    for pattern_name, data in DARK_PATTERNS.items():
        matched_keywords = [kw for kw in data["keywords"] if kw in text_lower]
        if matched_keywords:
            found.append({
                "pattern": pattern_name,
                "color": data["color"],
                "description": data["description"],
                "matched": matched_keywords,
                "severity": "High" if len(matched_keywords) >= 2 else "Medium"
            })
    return found

def highlight_text(text, findings):
    highlighted = text
    for finding in findings:
        for kw in finding["matched"]:
            pattern = re.compile(re.escape(kw), re.IGNORECASE)
            highlighted = pattern.sub(
                f'<mark style="background:{finding["color"]}; color:white; '
                f'padding:2px 4px; border-radius:3px;">{kw}</mark>',
                highlighted
            )
    return highlighted

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    findings = detect_patterns(text)
    highlighted = highlight_text(text, findings)
    score = min(100, len(findings) * 20)
    risk = "Low" if score < 20 else "Medium" if score < 60 else "High"
    return jsonify({
        "findings": findings,
        "highlighted": highlighted,
        "score": score,
        "risk": risk,
        "count": len(findings)
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
