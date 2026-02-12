import json
import csv
import random
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import google.generativeai as genai

# --------------------------------------------------
# 🔑 GEMINI SETUP
# --------------------------------------------------
# It pulls the key safely from the variable you added to Render
api_key = os.environ.get("AIzaSyB6LQlmgH11j6bz8PR9kDH-O633y6pisEQ")

if not api_key:
    print("⚠️ Error: Gemini_API_Key not found in environment!")
else:
    genai.configure(api_key=api_key)
    # Using the latest 1.5-flash model for fast, efficient responses
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# --------------------------------------------------
# 🚀 FLASK APP
# --------------------------------------------------
app = Flask(__name__)
CORS(app)

# --------------------------------------------------
# 📚 LOAD KNOWLEDGE BASE & DATASET
# --------------------------------------------------
with open("knowledge_base.json", "r", encoding="utf-8") as f:
    knowledge_base = json.load(f)

training_data = []
with open("dataset.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        if len(row) >= 2:
            training_data.append((row[0].lower().strip(), row[1].strip()))

X_train = [x[0] for x in training_data]
y_train = [x[1] for x in training_data]

# --------------------------------------------------
# 🧠 TRAIN INTENT MODEL
# --------------------------------------------------
model = make_pipeline(
    CountVectorizer(ngram_range=(1, 2)),
    LogisticRegression(max_iter=1000)
)
model.fit(X_train, y_train)

print(f"✅ Loaded {len(training_data)} samples")
print("✅ JeelBot ready")

# --------------------------------------------------
# 💾 SESSION MEMORY & HELPERS
# --------------------------------------------------
sessions = {}

def normalize_text(text):
    return text.lower().strip()

def random_from_list(items, last=None):
    if not items: return ""
    if last and len(items) > 1:
        items = [i for i in items if i != last]
    return random.choice(items)

def is_greeting(text):
    greetings = ["hi", "hello", "hey", "namaste", "morning", "evening"]
    return any(g in text for g in greetings)

def is_small_talk(text):
    return text in ["ok", "okay", "hmm", "yes", "yeah", "cool", "fine"]

def is_yoga_domain(text):
    yoga_keywords = [
        "yoga", "asana", "pose", "pranayama", "meditation", "breath", 
        "stress", "relax", "sleep", "flexibility", "sun salutation"
    ]
    return any(word in text for word in yoga_keywords)

# --------------------------------------------------
# 🤖 GEMINI ENHANCER (STRICT YOGA ONLY)
# --------------------------------------------------
def gemini_reply(prompt_text, context_data=""):
    try:
        # System prompt ensures the bot stays strictly within the yoga domain
        prompt = f"""
You are JeelBot, an intelligent yoga and wellness assistant.
CONTEXT INFORMATION: {context_data}

Rules:
- Only talk about yoga, meditation, breathing, sleep, or wellness.
- If the user asks something outside these topics, politely refuse.
- Use the CONTEXT INFORMATION to provide accurate details.
- Keep responses short (1–2 sentences) and supportive.
- Do not mention you are an AI.

User message: {prompt_text}
"""
        response = gemini_model.generate_content(prompt)
        return response.text.strip() if response.text else ""
    except Exception as e:
        print(f"❌ Gemini Error: {e}")
        return ""

# --------------------------------------------------
# 💬 CHAT ROUTE
# --------------------------------------------------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    raw_message = data.get("message", "")
    session_id = data.get("session_id", "default")

    message = normalize_text(raw_message)

    if session_id not in sessions:
        sessions[session_id] = {"last_intent": None, "last_reply": None}

    context = sessions[session_id]

    # 1. GREETING
    if is_greeting(message):
        base = random_from_list(knowledge_base["greeting"]["responses"], context.get("last_reply"))
        follow = gemini_reply("Add a friendly yoga-related follow-up question")
        reply = f"{base}\n{follow}" if follow else base
        context["last_reply"] = base
        return jsonify({"response": reply})

    # 2. STRICT DOMAIN FILTER
    if not is_yoga_domain(message) and not message.isdigit():
        return jsonify({"response": "I’m JeelBot 🌿 I only answer yoga and wellness questions."})

    # 3. SMALL TALK
    if is_small_talk(message):
        follow = gemini_reply("Respond casually within yoga context", "Keep it very brief.")
        return jsonify({"response": follow or "Alright 🌿"})

    # 4. FOLLOW-UP OPTIONS (Numeric menu)
    if context.get("last_intent") and message in ["1", "2", "3", "4", "5"]:
        intent = context["last_intent"]
        info = knowledge_base.get(intent, {})
        if message == "1":
            return jsonify({"response": f"⏰ Duration: {info.get('duration')}\n🕒 Best time: {info.get('best_time')}"})
        if message == "2":
            return jsonify({"response": "🌬️ Breathing:\n• " + "\n• ".join(info.get("breathing", []))})
        if message == "3":
            poses = ", ".join(info.get("poses", []))
            ai_desc = gemini_reply(f"Briefly describe benefits of: {poses}")
            return jsonify({"response": f"🧘 Poses:\n• " + "\n• ".join(info.get("poses", [])) + f"\n\n{ai_desc}"})
        if message == "4":
            return jsonify({"response": "⚠️ Safety Tips:\n• " + "\n• ".join(info.get("tips", []))})
        if message == "5":
            context["last_intent"] = None
            return jsonify({"response": "Sure 🌿 What would you like to explore next?"})

    # 5. INTENT PREDICTION
    probs = model.predict_proba([message])[0]
    if max(probs) > 0.3:  # Confidence threshold
        intent = model.classes_[probs.argmax()]
        if intent in knowledge_base:
            context["last_intent"] = intent
            intro = gemini_reply(message, str(knowledge_base[intent]))
            return jsonify({
                "response": f"{intro}\n\nWhat would you like to explore?\n1️⃣ Duration & Time\n2️⃣ Breathing\n3️⃣ Poses\n4️⃣ Safety Tips\n5️⃣ Another topic"
            })

    # 6. FALLBACK
    return jsonify({"response": "I'm not sure about that. Try asking about a specific yoga pose or wellness topic 🌿"})

if __name__ == "__main__":
    # Get port from environment for Render deployment
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)