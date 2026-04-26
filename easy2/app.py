from flask import Flask, render_template, request, jsonify
import re
from collections import Counter

app = Flask(__name__)

# Lexicon-based sentiment analysis (no external ML needed)
POSITIVE_WORDS = set([
    "good","great","excellent","amazing","wonderful","fantastic","love","best",
    "happy","positive","nice","beautiful","perfect","awesome","brilliant","superb",
    "outstanding","pleased","satisfied","enjoy","helpful","recommend","easy","fast",
    "reliable","trustworthy","honest","clear","fair","quality","value","impressive"
])

NEGATIVE_WORDS = set([
    "bad","terrible","awful","horrible","hate","worst","poor","useless","broken",
    "slow","expensive","difficult","annoying","frustrating","disappointed","scam",
    "misleading","confusing","tricky","hidden","fake","manipulative","deceptive",
    "forced","unfair","cheated","dishonest","unclear","complicated","overpriced"
])

INTENSIFIERS = {"very","extremely","really","absolutely","completely","totally","so","quite"}
NEGATORS = {"not","no","never","neither","nor","don't","doesn't","didn't","isn't","wasn't"}

def analyze_sentiment(text):
    words = re.findall(r'\b\w+\b', text.lower())
    pos_score = 0; neg_score = 0
    pos_found = []; neg_found = []
    for i, word in enumerate(words):
        prev = words[i-1] if i > 0 else ""
        prev2 = words[i-2] if i > 1 else ""
        multiplier = 1.5 if prev in INTENSIFIERS or prev2 in INTENSIFIERS else 1.0
        negated = prev in NEGATORS or prev2 in NEGATORS
        if word in POSITIVE_WORDS:
            if negated: neg_score += multiplier; neg_found.append(word)
            else: pos_score += multiplier; pos_found.append(word)
        elif word in NEGATIVE_WORDS:
            if negated: pos_score += multiplier; pos_found.append(word)
            else: neg_score += multiplier; neg_found.append(word)
    total = pos_score + neg_score
    if total == 0:
        compound = 0.0; label = "Neutral"; emoji = "😐"; color = "#95a5a6"
    else:
        compound = (pos_score - neg_score) / total
        if compound > 0.2: label = "Positive"; emoji = "😊"; color = "#27ae60"
        elif compound < -0.2: label = "Negative"; emoji = "😞"; color = "#e74c3c"
        else: label = "Neutral"; emoji = "😐"; color = "#f39c12"
    word_count = len(words)
    sentences = len(re.split(r'[.!?]+', text.strip()))
    return {
        "label": label, "emoji": emoji, "color": color,
        "compound": round(compound, 3),
        "pos_score": round(pos_score, 2), "neg_score": round(neg_score, 2),
        "pos_pct": round(pos_score / max(total,1) * 100, 1),
        "neg_pct": round(neg_score / max(total,1) * 100, 1),
        "pos_words": list(set(pos_found))[:8],
        "neg_words": list(set(neg_found))[:8],
        "word_count": word_count, "sentences": sentences,
        "avg_word_len": round(sum(len(w) for w in words) / max(len(words),1), 1)
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    texts = data.get("texts", [])
    if not texts:
        return jsonify({"error": "No text"}), 400
    results = [analyze_sentiment(t) for t in texts if t.strip()]
    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(debug=True, port=5001)
