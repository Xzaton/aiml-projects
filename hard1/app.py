"""
AI Chatbot Flask App — Intent Classification Chatbot
"""

from flask import Flask, render_template, request, jsonify, session
import pickle, json, os, re, random
from datetime import datetime
from train import train_and_save

app = Flask(__name__)
app.secret_key = 'darkpatterns_chatbot_secret_2024'

# ── Load or train model ──
def load_model():
    try:
        with open('model/chatbot_pipeline.pkl','rb') as f: pipeline = pickle.load(f)
        with open('model/label_encoder.pkl','rb') as f: le = pickle.load(f)
        with open('model/intents.json','r') as f: intents = json.load(f)
        return pipeline, le, intents
    except:
        print("Training model...")
        return train_and_save()

pipeline, le, intents = load_model()

# ── Entity extractor ──
def extract_entities(text):
    entities = {}
    urls = re.findall(r'https?://\S+|www\.\S+', text)
    if urls: entities['url'] = urls[0]
    prices = re.findall(r'₹\d+|\$\d+|\d+ rupees|\d+ dollars', text)
    if prices: entities['price'] = prices[0]
    return entities

# ── Context manager ──
class ConversationContext:
    def __init__(self):
        self.history = []
        self.last_intent = None
        self.turn_count = 0

    def update(self, user_msg, bot_response, intent):
        self.history.append({
            'user': user_msg, 'bot': bot_response,
            'intent': intent, 'time': datetime.now().strftime('%H:%M')
        })
        self.last_intent = intent
        self.turn_count += 1
        if len(self.history) > 20:
            self.history = self.history[-20:]

contexts = {}  # session_id -> ConversationContext

def get_context(session_id):
    if session_id not in contexts:
        contexts[session_id] = ConversationContext()
    return contexts[session_id]

def predict_intent(text):
    processed = text.lower().strip()
    proba = pipeline.predict_proba([processed])[0]
    pred_idx = proba.argmax()
    confidence = float(proba[pred_idx])
    intent = le.inverse_transform([pred_idx])[0]
    return intent, confidence

def generate_response(intent, confidence, context, user_text):
    # Low confidence → fallback
    if confidence < 0.25:
        fallbacks = [
            "I'm not sure I understood that. Could you rephrase? I can help with dark patterns, UX ethics, legal info, and design best practices.",
            "I didn't quite catch that. Try asking about: dark pattern types, detection methods, GDPR, or ethical design.",
            f"Hmm, I'm not confident about '{user_text}'. Could you be more specific? Type 'help' to see what I can do."
        ]
        return random.choice(fallbacks), "fallback", confidence

    # Context-aware responses
    intent_data = intents.get(intent, {})
    responses = intent_data.get('responses', ["I can help with dark patterns and UX ethics topics."])

    # Vary response based on turn count to avoid repetition
    response = responses[context.turn_count % len(responses)]

    # Add follow-up suggestions
    suggestions = {
        "dark_pattern_info": "Would you like to know about specific types? Try asking 'what are examples of dark patterns'",
        "greeting": "You can ask me about dark patterns, GDPR, ethical design, or detection methods.",
        "dark_pattern_examples": "Want to know more about any specific pattern? Ask about 'roach motel' or 'confirm shaming'.",
        "legal_info": "Want to know how to design ethically to avoid these issues?",
        "how_to_detect": "Would you like recommendations for ethical design practices?",
    }
    follow_up = suggestions.get(intent, "")
    if follow_up and context.turn_count > 0:
        response = f"{response}\n\n💡 *{follow_up}*"

    return response, intent, confidence

@app.route('/')
def index():
    if 'session_id' not in session:
        session['session_id'] = os.urandom(8).hex()
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_text = data.get('message', '').strip()
    if not user_text:
        return jsonify({'error': 'Empty message'}), 400

    session_id = session.get('session_id', 'default')
    ctx = get_context(session_id)

    intent, confidence = predict_intent(user_text)
    entities = extract_entities(user_text)
    response, final_intent, conf = generate_response(intent, confidence, ctx, user_text)
    ctx.update(user_text, response, final_intent)

    return jsonify({
        'response': response,
        'intent': final_intent,
        'confidence': round(conf * 100, 1),
        'entities': entities,
        'turn': ctx.turn_count
    })

@app.route('/history')
def history():
    session_id = session.get('session_id', 'default')
    ctx = get_context(session_id)
    return jsonify({'history': ctx.history})

@app.route('/reset', methods=['POST'])
def reset():
    session_id = session.get('session_id', 'default')
    if session_id in contexts:
        del contexts[session_id]
    return jsonify({'status': 'reset'})

if __name__ == '__main__':
    app.run(debug=True, port=5003)
