import json
import csv
import random
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# 1. 🔑 SWITCH TO THE MODERN LIBRARY
from google import genai

# --------------------------------------------------
# 🔑 GEMINI SETUP
# --------------------------------------------------
# Use the exact key name you set in Render's dashboard
api_key = os.environ.get("Gemini_API_Key")

if not api_key:
    print("⚠️ Error: Gemini_API_Key not found in environment!")
    client = None
else:
    # New initialization style for the modern SDK
    client = genai.Client(api_key=api_key)
    print("✅ Gemini Client connected")

# --------------------------------------------------
# 🚀 FLASK APP
# --------------------------------------------------
app = Flask(__name__)
CORS(app)

# --------------------------------------------------
# 📚 LOAD KNOWLEDGE BASE & DATASET
# --------------------------------------------------
# Ensure these files are in your GitHub repository!
try:
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

    model = make_pipeline(
        CountVectorizer(ngram_range=(1, 2)),
        LogisticRegression(max_iter=1000)
    )
    model.fit(X_train, y_train)
    print(f"✅ Loaded {len(training_data)} samples")
except Exception as e:
    print(f"❌ Initialization Error: {e}")

print("✅ JeelBot ready")

# --------------------------------------------------
# 💾 HELPERS & DOMAIN FILTER
# --------------------------------------------------
sessions = {}

def is_yoga_domain(text):
    yoga_keywords = [
        "yoga", "asana", "pose", "pranayama", "meditation", "breath", 
        "stress", "relax", "sleep", "flexibility", "sun salutation", "wellness"
    ]
    return any(word in text.lower() for word in yoga_keywords)

# --------------------------------------------------
# 🤖 GEMINI ENHANCER (MODERN SDK)
# --------------------------------------------------
def gemini_reply(prompt_text, context_data=""):
    if not client:
        return "I'm having trouble connecting to my brain right now. 🌿"
    
    try:
        prompt = f"""
        You are JeelBot, an intelligent yoga and wellness assistant.
        CONTEXT: {context_data}
        Rules:
        - Only talk about yoga, meditation, breathing, sleep, or wellness.
        - Short responses (1–2 sentences).
        - No mention of being an AI.
        User: {prompt_text}
        """
        # Updated method call for google-genai
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=prompt
        )
        return response.text.strip()
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
    
    if session_id not in sessions:
        sessions[session_id] = {"last_intent": None}
    
    # 1. Check if user is asking for a specific menu item (1-5)
    if raw_message.strip() in ["1", "2", "3", "4", "5"] and sessions[session_id].get("last_intent"):
        intent = sessions[session_id]["last_intent"]
        info = knowledge_base.get(intent, {})
        
        responses = {
            "1": f"⏰ Duration: {info.get('duration')}\n🕒 Best time: {info.get('best_time')}",
            "2": f"🌬️ Breathing:\n• " + "\n• ".join(info.get("breathing", [])),
            "3": f"🧘 Poses:\n• " + "\n• ".join(info.get("poses", [])),
            "4": f"⚠️ Safety Tips:\n• " + "\n• ".join(info.get("tips", [])),
            "5": "Sure 🌿 What else can I help with?"
        }
        
        if raw_message.strip() == "5": sessions[session_id]["last_intent"] = None
        return jsonify({"response": responses.get(raw_message.strip())})

    # 2. General Yoga Check
    if not is_yoga_domain(raw_message):
        return jsonify({"response": "I’m JeelBot 🌿 I specialize in yoga and wellness. Ask me about poses or stress relief!"})

    # 3. Predict Intent
    probs = model.predict_proba([raw_message.lower().strip()])[0]
    confidence = max(probs)
    intent = model.classes_[probs.argmax()]

    if confidence > 0.4 and intent in knowledge_base:
        sessions[session_id]["last_intent"] = intent
        intro = gemini_reply(raw_message, str(knowledge_base[intent]))
        return jsonify({
            "response": f"{intro}\n\nExplore more:\n1️⃣ Time\n2️⃣ Breathing\n3️⃣ Poses\n4️⃣ Safety\n5️⃣ New Topic"
        })

    # 4. Fallback to Gemini
    ai_response = gemini_reply(raw_message)
    return jsonify({"response": ai_response or "I'm here for your wellness needs! 🌿"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)