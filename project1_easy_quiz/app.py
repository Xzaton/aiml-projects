"""
Project 1: Dark Pattern Awareness Quiz App
Difficulty: EASY
Tech: Python + Flask + HTML/CSS
Run: pip install flask && python app.py
"""

from flask import Flask, render_template_string, request, session, jsonify
import random

app = Flask(__name__)
app.secret_key = "darkpatterns_quiz_2024"

QUESTIONS = [
    {
        "id": 1,
        "question": "A website shows '⚠️ Only 1 seat left!' even though 50 seats are available. What dark pattern is this?",
        "options": ["Roach Motel", "Scarcity Pressure / False Urgency", "Confirm Shaming", "Hidden Costs"],
        "answer": 1,
        "explanation": "Scarcity Pressure creates artificial urgency or false scarcity to rush users into decisions they might not make with more time."
    },
    {
        "id": 2,
        "question": "You signed up for a free trial with your credit card. 30 days later you're charged ₹999/month with no reminder. This is called:",
        "options": ["Bait and Switch", "Misdirection", "Forced Continuity", "Trick Question"],
        "answer": 2,
        "explanation": "Forced Continuity silently converts free trials into paid subscriptions without adequate warning."
    },
    {
        "id": 3,
        "question": "A popup says: 'Yes, sign me up for amazing deals!' / 'No thanks, I hate saving money.' What is this?",
        "options": ["Confirm Shaming", "Roach Motel", "Privacy Zuckering", "Disguised Ads"],
        "answer": 0,
        "explanation": "Confirm Shaming guilt-trips users who try to opt out by making the rejection option embarrassing or self-deprecating."
    },
    {
        "id": 4,
        "question": "A shopping site adds travel insurance to your cart automatically. You have to manually uncheck it. This is:",
        "options": ["Hidden Costs", "Sneak into Basket", "Roach Motel", "Misdirection"],
        "answer": 1,
        "explanation": "Sneak into Basket adds items to the cart without explicit user action, exploiting inertia and inattention."
    },
    {
        "id": 5,
        "question": "Creating an account takes 2 clicks. Deleting your account requires a 5-page form + email verification + 30-day wait. This is:",
        "options": ["Forced Continuity", "Bait and Switch", "Roach Motel", "Trick Questions"],
        "answer": 2,
        "explanation": "Roach Motel: easy to get in, nearly impossible to leave. Asymmetric ease of entry vs. exit is the hallmark."
    },
    {
        "id": 6,
        "question": "A cookie banner has a large green 'Accept All' button and a tiny grey 'Manage Preferences' link. This is:",
        "options": ["Scarcity Pressure", "Misdirection / Visual Interference", "Hidden Costs", "Confirm Shaming"],
        "answer": 1,
        "explanation": "Misdirection uses visual weight, color, and size to steer users toward the business-preferred option."
    },
    {
        "id": 7,
        "question": "A checkbox says: 'Uncheck this box if you do not wish to not receive our emails.' This uses:",
        "options": ["Roach Motel", "Forced Continuity", "Trick Questions / Double Negative", "Bait and Switch"],
        "answer": 2,
        "explanation": "Trick Questions use double negatives or confusing grammar to make opt-out accidentally opt users IN."
    },
    {
        "id": 8,
        "question": "A flight booking site shows ₹3,499 throughout but reveals ₹5,899 (with taxes, fees, seat charges) only at payment. This is:",
        "options": ["Hidden Costs", "Scarcity Pressure", "Confirm Shaming", "Privacy Zuckering"],
        "answer": 0,
        "explanation": "Hidden Costs withhold the full price until the user has invested time and intent, exploiting the sunk cost fallacy."
    },
    {
        "id": 9,
        "question": "An app's privacy settings have data sharing turned ON by default and require 7 sub-menus to turn OFF. This is:",
        "options": ["Misdirection", "Privacy Zuckering", "Forced Continuity", "Roach Motel"],
        "answer": 1,
        "explanation": "Privacy Zuckering (named after Facebook's founder) tricks users into sharing more personal data than intended."
    },
    {
        "id": 10,
        "question": "Which law requires that cookie consent be as easy to withdraw as to give?",
        "options": ["CCPA (California)", "GDPR (European Union)", "IT Act 2000 (India)", "COPPA (USA)"],
        "answer": 1,
        "explanation": "The EU's GDPR Article 7(3) states that withdrawal of consent must be as easy as giving it, directly targeting dark pattern consent flows."
    },
]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dark Pattern Awareness Quiz</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', sans-serif; background: #0f0f1a; color: #e0e0e0; min-height: 100vh; }
  .header { background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 24px 40px; border-bottom: 2px solid #e94560; }
  .header h1 { font-size: 1.8rem; color: #fff; } .header p { color: #aaa; margin-top: 4px; }
  .container { max-width: 780px; margin: 40px auto; padding: 0 20px; }
  .progress-bar { background: #1e1e2e; border-radius: 8px; height: 8px; margin-bottom: 28px; }
  .progress-fill { background: linear-gradient(90deg, #e94560, #f5a623); height: 100%; border-radius: 8px; transition: width 0.4s; }
  .progress-text { display: flex; justify-content: space-between; color: #888; font-size: 0.85rem; margin-bottom: 8px; }
  .card { background: #1a1a2e; border: 1px solid #2a2a4a; border-radius: 16px; padding: 32px; margin-bottom: 20px; }
  .q-number { color: #e94560; font-size: 0.85rem; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 12px; }
  .q-text { font-size: 1.15rem; line-height: 1.6; color: #fff; margin-bottom: 28px; }
  .options { display: grid; gap: 12px; }
  .option-btn { background: #0f0f1a; border: 1.5px solid #2a2a4a; border-radius: 10px; padding: 14px 18px;
    color: #ccc; font-size: 1rem; cursor: pointer; text-align: left; transition: all 0.2s; display: flex; align-items: center; gap: 12px; }
  .option-btn:hover { border-color: #e94560; color: #fff; background: #1e1e2e; }
  .option-btn.correct { background: #0d2e1a; border-color: #2ecc71; color: #2ecc71; }
  .option-btn.wrong { background: #2e0d0d; border-color: #e74c3c; color: #e74c3c; }
  .option-btn.reveal { border-color: #2ecc71; color: #2ecc71; opacity: 0.6; }
  .option-label { background: #2a2a4a; color: #888; border-radius: 6px; padding: 3px 9px; font-size: 0.8rem; font-weight: 700; }
  .explanation { background: #0d2818; border-left: 4px solid #2ecc71; border-radius: 8px; padding: 16px 20px;
    margin-top: 20px; color: #a8e6c1; font-size: 0.95rem; line-height: 1.6; display: none; }
  .next-btn { background: linear-gradient(135deg, #e94560, #c0392b); color: #fff; border: none; border-radius: 10px;
    padding: 14px 32px; font-size: 1rem; font-weight: 600; cursor: pointer; margin-top: 20px; width: 100%; display: none; }
  .next-btn:hover { opacity: 0.9; transform: translateY(-1px); }
  .score-card { background: #1a1a2e; border: 2px solid #e94560; border-radius: 20px; padding: 48px 32px; text-align: center; }
  .score-big { font-size: 5rem; font-weight: 900; color: #f5a623; }
  .score-label { color: #888; font-size: 1rem; margin-top: 8px; margin-bottom: 24px; }
  .score-msg { font-size: 1.3rem; color: #fff; margin-bottom: 8px; }
  .score-sub { color: #aaa; font-size: 0.95rem; margin-bottom: 32px; }
  .restart-btn { background: linear-gradient(135deg, #e94560, #c0392b); color: #fff; border: none; border-radius: 12px;
    padding: 16px 40px; font-size: 1.05rem; font-weight: 700; cursor: pointer; }
  .badge { display: inline-block; background: #e94560; color: #fff; border-radius: 20px; padding: 6px 18px; font-size: 0.8rem; font-weight: 700; margin: 4px; }
  .badge.green { background: #27ae60; } .badge.blue { background: #2980b9; }
</style>
</head>
<body>
<div class="header">
  <h1>🎯 Dark Pattern Awareness Quiz</h1>
  <p>Project 1 &nbsp;|&nbsp; AIML &nbsp;|&nbsp; Piyush Mishra — SAP 590024892</p>
</div>
<div class="container" id="app">
  <div id="quiz-area"></div>
</div>

<script>
const questions = {{ questions|tojson }};
let current = 0, score = 0, answered = false;

function render() {
  if (current >= questions.length) { showScore(); return; }
  const q = questions[current];
  const pct = Math.round((current / questions.length) * 100);
  const labels = ['A','B','C','D'];
  document.getElementById('quiz-area').innerHTML = `
    <div class="progress-text"><span>Question ${current+1} of ${questions.length}</span><span>Score: ${score}</span></div>
    <div class="progress-bar"><div class="progress-fill" style="width:${pct}%"></div></div>
    <div class="card">
      <div class="q-number">Question ${current+1}</div>
      <div class="q-text">${q.question}</div>
      <div class="options">
        ${q.options.map((o,i)=>`
          <button class="option-btn" id="opt${i}" onclick="choose(${i})">
            <span class="option-label">${labels[i]}</span> ${o}
          </button>`).join('')}
      </div>
      <div class="explanation" id="exp">${q.explanation}</div>
      <button class="next-btn" id="nxt" onclick="next()">
        ${current+1 < questions.length ? 'Next Question →' : 'See Results 🏆'}
      </button>
    </div>`;
  answered = false;
}

function choose(i) {
  if (answered) return;
  answered = true;
  const q = questions[current];
  const btns = document.querySelectorAll('.option-btn');
  btns.forEach((b,idx)=>{
    if (idx === q.answer) b.classList.add('correct');
    else if (idx === i && i !== q.answer) b.classList.add('wrong');
    b.disabled = true;
  });
  if (i === q.answer) score++;
  document.getElementById('exp').style.display = 'block';
  document.getElementById('nxt').style.display = 'block';
}

function next() { current++; render(); }

function showScore() {
  const pct = Math.round((score/questions.length)*100);
  let msg, sub, badges;
  if (pct >= 90) { msg="🏆 Expert Level!"; sub="You have mastered dark pattern awareness."; badges='<span class="badge green">Expert</span><span class="badge blue">HCI Pro</span>'; }
  else if (pct >= 70) { msg="🎯 Well Done!"; sub="Strong understanding of deceptive design."; badges='<span class="badge green">Proficient</span>'; }
  else if (pct >= 50) { msg="📚 Keep Learning"; sub="Review the explanations and try again."; badges='<span class="badge">Intermediate</span>'; }
  else { msg="🔍 Just Getting Started"; sub="Dark patterns are tricky — study up!"; badges='<span class="badge">Beginner</span>'; }
  document.getElementById('quiz-area').innerHTML = `
    <div class="score-card">
      <div class="score-big">${score}/${questions.length}</div>
      <div class="score-label">${pct}% Correct</div>
      <div class="score-msg">${msg}</div>
      <div class="score-sub">${sub}</div>
      <div style="margin-bottom:28px">${badges}</div>
      <button class="restart-btn" onclick="restart()">🔄 Retake Quiz</button>
    </div>`;
}

function restart() { current=0; score=0; render(); }
render();
</script>
</body>
</html>
"""

@app.route("/")
def index():
    q = random.sample(QUESTIONS, len(QUESTIONS))
    return render_template_string(HTML_TEMPLATE, questions=q)

if __name__ == "__main__":
    print("🚀 Dark Pattern Quiz running at http://localhost:5000")
    app.run(debug=True, port=5000)
