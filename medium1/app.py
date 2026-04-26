from flask import Flask, render_template, request, jsonify
import pickle, os, re

app = Flask(__name__)

# Load or auto-train model
def load_model():
    model_path = 'model/pipeline.pkl'
    if not os.path.exists(model_path):
        print("Model not found. Training now...")
        from train import train_and_save
        return train_and_save()
    with open(model_path, 'rb') as f:
        return pickle.load(f)

pipeline = load_model()

# Heuristic signals for explainability
FAKE_SIGNALS = {
    "ALL CAPS words": r'\b[A-Z]{4,}\b',
    "Excessive exclamation": r'!{2,}|!.*!',
    "Clickbait phrases": r'\b(won\'t believe|shocking|bombshell|exposed|secret|leaked|banned|they don\'t want)\b',
    "Urgency language": r'\b(urgent|breaking|share now|before they delete)\b',
    "Conspiracy language": r'\b(deep state|shadow government|cover.up|illuminati|new world order|microchip|mind control)\b',
}
REAL_SIGNALS = {
    "Attribution present": r'\b(according to|study shows|researchers|officials|government|university)\b',
    "Measured language": r'\b(suggests|indicates|reports|announces|approximately|percent)\b',
    "Institutional reference": r'\b(court|parliament|ministry|commission|department|agency)\b',
}

def preprocess(text):
    t = text.lower()
    t = re.sub(r'[^a-z\s]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()

def explain(text):
    found_fake = {}
    found_real = {}
    for name, pattern in FAKE_SIGNALS.items():
        m = re.findall(pattern, text, re.IGNORECASE)
        if m: found_fake[name] = list(set(m))[:3]
    for name, pattern in REAL_SIGNALS.items():
        m = re.findall(pattern, text, re.IGNORECASE)
        if m: found_real[name] = list(set(m))[:3]
    return found_fake, found_real

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    processed = preprocess(text)
    proba = pipeline.predict_proba([processed])[0]
    pred = int(pipeline.predict([processed])[0])
    fake_prob = float(proba[1])
    real_prob = float(proba[0])
    fake_signals, real_signals = explain(text)
    label = 'FAKE' if pred == 1 else 'REAL'
    confidence = max(fake_prob, real_prob)
    risk = 'High' if fake_prob > 0.75 else 'Medium' if fake_prob > 0.45 else 'Low'
    return jsonify({
        'label': label, 'confidence': round(confidence*100, 1),
        'fake_prob': round(fake_prob*100, 1),
        'real_prob': round(real_prob*100, 1),
        'risk': risk,
        'fake_signals': fake_signals,
        'real_signals': real_signals,
    })

if __name__ == '__main__':
    app.run(debug=True, port=5002)
