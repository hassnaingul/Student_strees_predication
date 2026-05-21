from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# ── Load Artifacts ─────────────────────────────────────────────────────────────
with open("model.pkl",   "rb") as f: model   = pickle.load(f)
with open("scaler.pkl",  "rb") as f: scaler  = pickle.load(f)
with open("encoder.pkl", "rb") as f: encoder = pickle.load(f)

FEATURE_COLS = [
    'SleepHours', 'StudyWorkHours', 'ScreenTime', 'PhysicalActivity',
    'SocialInteraction', 'AcademicWorkPressure', 'FinancialPressure',
    'AnxietyLevel', 'MoodScore', 'RelationshipIssues'
]

STRESS_INFO = {
    "Low": {
        "emoji": "😊",
        "color": "#22c55e",
        "message": "You're managing stress well! Keep up your healthy habits.",
        "tips": [
            "Maintain your current sleep schedule",
            "Continue regular physical activity",
            "Keep nurturing your social connections"
        ]
    },
    "Medium": {
        "emoji": "😐",
        "color": "#f59e0b",
        "message": "Moderate stress detected. Take some steps to manage it.",
        "tips": [
            "Try to get 7–9 hours of sleep per night",
            "Reduce screen time before bed",
            "Practice mindfulness or deep breathing",
            "Take short breaks during study/work sessions"
        ]
    },
    "High": {
        "emoji": "😟",
        "color": "#ef4444",
        "message": "High stress detected. Please prioritize your well-being.",
        "tips": [
            "Speak to a counselor or trusted person",
            "Break your workload into smaller tasks",
            "Ensure at least 7 hours of sleep",
            "Engage in physical activity daily",
            "Limit social media / screen time",
            "Consider professional mental health support"
        ]
    }
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form
        features = [float(data[col]) for col in FEATURE_COLS]
        X = np.array(features).reshape(1, -1)
        X_scaled = scaler.transform(X)
        pred_encoded = model.predict(X_scaled)[0]
        pred_proba   = model.predict_proba(X_scaled)[0]
        label = encoder.inverse_transform([pred_encoded])[0]

        probabilities = {
            cls: round(float(prob) * 100, 1)
            for cls, prob in zip(encoder.classes_, pred_proba)
        }

        result = {
            "prediction": label,
            "probabilities": probabilities,
            "info": STRESS_INFO[label]
        }
        return render_template("result.html", result=result, inputs=dict(data))
    except Exception as e:
        return render_template("index.html", error=f"Error: {str(e)}")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API endpoint for programmatic access."""
    try:
        data = request.get_json()
        features = [float(data[col]) for col in FEATURE_COLS]
        X = np.array(features).reshape(1, -1)
        X_scaled = scaler.transform(X)
        pred_encoded = model.predict(X_scaled)[0]
        pred_proba   = model.predict_proba(X_scaled)[0]
        label = encoder.inverse_transform([pred_encoded])[0]

        return jsonify({
            "stress_level": label,
            "probabilities": {
                cls: round(float(p) * 100, 1)
                for cls, p in zip(encoder.classes_, pred_proba)
            },
            "status": "success"
        })
    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"}), 400

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
