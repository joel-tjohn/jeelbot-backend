import json
import csv
import random
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

app = Flask(__name__)
CORS(app)

# --------------------------------------------------
# 1. LOAD KNOWLEDGE BASE
# --------------------------------------------------
with open("knowledge_base.json", "r", encoding="utf-8") as f:
    knowledge_base = json.load(f)

# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------
csv_path = "dataset.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError("❌ dataset.csv not found!")

training_data = []
with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        if len(row) >= 2:
            training_data.append((row[0].lower().strip(), row[1].strip()))

print(f"✅ Loaded {len(training_data)} training samples")

X_train = [x[0] for x in training_data]
y_train = [x[1] for x in training_data]

# --------------------------------------------------
# 3. TRAIN MODEL
# --------------------------------------------------
model = make_pipeline(
    CountVectorizer(ngram_range=(1, 2)),
    LogisticRegression(max_iter=1000)
)
model.fit(X_train, y_train)

print("🌿 JeelBot AI Model Trained Successfully")

# --------------------------------------------------
# 4. SESSION MEMORY
# --------------------------------------------------
sessions = {}

# --------------------------------------------------
# 5. HELPERS
# --------------------------------------------------
def normalize_text(text: str) -> str:
    replacements = {
        "womens yoga": "women yoga",
        "women's yoga": "women yoga",
        "female yoga": "women yoga",
        "ladies yoga": "women yoga"
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def is_greeting(text: str) -> bool:
    greetings = [
        "hi", "hello", "hey", "hai", "hii",
        "good morning", "good evening", "good afternoon",
        "hello there", "hey bot", "hello bot",
        "how are you", "how r u"
    ]
    return any(text.startswith(g) for g in greetings)


def is_invalid_input(text: str) -> bool:
    return not text or len(text) <= 1 or all(not c.isalnum() for c in text)


def is_yoga_domain_query(text: str) -> bool:
    keywords = [
        "yoga", "pose", "asana", "asanas",
        "pranayama", "breathing", "meditation",
        "stress", "sleep", "women", "kids",
        "flexibility", "strength", "weight",
        "menstrual", "hormonal"
    ]
    return any(word in text for word in keywords)


# --------------------------------------------------
# 6. CHAT API
# --------------------------------------------------
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json or {}
        message = normalize_text(
            data.get("message", "").lower().strip()
        )
        session_id = data.get("session_id")

        if not session_id:
            return jsonify({"response": "Session ID missing."})

        if session_id not in sessions:
            sessions[session_id] = {"stage": "intro", "last_intent": None}

        context = sessions[session_id]

        # ---------- GREETING ----------
        if is_greeting(message):
            return jsonify({
                "response": "Hello 🌿 I’m JeelBot. How can I support your wellness today?"
            })

        # ---------- INVALID INPUT ----------
        if is_invalid_input(message):
            return jsonify({
                "response": "🤔 I didn’t quite get that. Try asking about yoga for stress, sleep, beginners, or women’s health 🌿"
            })

        # ---------- CONTEXT FOLLOW-UP (✅ FIX ADDED) ----------
        if context.get("last_intent") and message in ["1", "2", "3", "4", "yes"]:
            intent = context["last_intent"]
            info = knowledge_base[intent]

            if message in ["1", "yes"]:
                return jsonify({
                    "response": f"⏰ Duration: {info.get('duration')}\n🕒 Best time: {info.get('best_time')}"
                })

            if message == "2":
                return jsonify({
                    "response": "🧘 Step-by-step guidance:\n• " + "\n• ".join(info.get("poses", []))
                })

            if message == "3":
                return jsonify({
                    "response": "⚠️ Safety tips:\n• " + "\n• ".join(info.get("tips", []))
                })

            if message == "4":
                context["last_intent"] = None
                return jsonify({
                    "response": "Sure 🌿 What yoga topic would you like next?"
                })

        # ---------- INTENT CLASSIFICATION ----------
        probs = model.predict_proba([message])[0]
        confidence = max(probs)
        intent = model.classes_[probs.argmax()]

        print(f"User: {message} → {intent} ({confidence:.2f})")

        # ---------- CONFIDENCE GATE ----------
        if confidence < 0.30 and not is_yoga_domain_query(message):
            return jsonify({
                "response": "I can help only with yoga and wellness topics 🌿"
            })

        # ---------- KNOWLEDGE RESPONSE ----------
        if intent in knowledge_base:
            sessions[session_id]["last_intent"] = intent
            info = knowledge_base[intent]
            benefit = random.choice(info.get("benefits", ["overall wellness"]))

            return jsonify({
                "response": (
                    f"{intent.replace('_', ' ').title()} is especially helpful for **{benefit}** 🌿\n\n"
                    "Would you like:\n"
                    "1️⃣ Duration & best time\n"
                    "2️⃣ Step-by-step guidance\n"
                    "3️⃣ Safety tips\n"
                    "4️⃣ Another yoga topic?"
                )
            })

        # ---------- FALLBACK ----------
        return jsonify({
            "response": "I’m here to help with yoga, stress relief, sleep, and wellness 🌿"
        })

    except Exception as e:
        print("❌ Chat error:", e)
        return jsonify({
            "response": "⚠️ Something went wrong. Please try again 🌿"
        })


# --------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
