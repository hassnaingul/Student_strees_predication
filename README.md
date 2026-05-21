# 🧠 Student Stress Level Predictor

A machine learning web app that predicts student stress levels (**Low / Medium / High**) based on lifestyle and academic habits, built with **Flask + MLP Neural Network**.

---

## 📊 Dataset Features

| Feature | Description |
|---|---|
| SleepHours | Hours of sleep per night |
| StudyWorkHours | Daily study/work hours |
| ScreenTime | Daily screen time (hrs) |
| PhysicalActivity | Daily physical activity (hrs) |
| SocialInteraction | Daily social interaction (hrs) |
| AcademicWorkPressure | Academic pressure (1–10) |
| FinancialPressure | Financial stress (1–10) |
| AnxietyLevel | Anxiety score (1–10) |
| MoodScore | Mood rating (1–10) |
| RelationshipIssues | Relationship issues (0/1) |

**Target:** `StressLevel` → Low / Medium / High

---

## 🤖 Model

- **Algorithm:** MLPClassifier (Neural Network)
- **Architecture:** 10 → 128 → 64 → 3
- **Test Accuracy:** 85%
- **Scaler:** StandardScaler

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/student-stress-predictor.git
cd student-stress-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (creates model.pkl, scaler.pkl, encoder.pkl)
python train_model.py

# 4. Start the Flask app
python app.py
```

Open → http://localhost:5000

---

## 🌐 Deploy to Render (Free Hosting)

1. Push this repo to GitHub
2. Go to [https://render.com](https://render.com) → **New Web Service**
3. Connect your GitHub repo
4. Set:
   - **Build Command:** `pip install -r requirements.txt && python train_model.py`
   - **Start Command:** `gunicorn app:app`
   - **Environment:** Python 3
5. Click **Deploy** → get your live URL 🎉

---

## 🌐 Deploy to Railway

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

---

## 📁 Project Structure

```
student-stress-predictor/
├── app.py                  ← Flask application
├── train_model.py          ← Model training script
├── stress_dataset.csv      ← Dataset
├── model.pkl               ← Trained model (auto-generated)
├── scaler.pkl              ← Scaler (auto-generated)
├── encoder.pkl             ← Label encoder (auto-generated)
├── requirements.txt        ← Python dependencies
├── Procfile                ← For Render/Heroku deployment
├── templates/
│   ├── index.html          ← Input form
│   ├── result.html         ← Prediction result
│   └── dashboard.html      ← Model dashboard
└── static/
    ├── confusion_matrix.png
    └── loss_curve.png
```

---

## 🔌 API Usage

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "SleepHours": 6,
    "StudyWorkHours": 8,
    "ScreenTime": 5,
    "PhysicalActivity": 1,
    "SocialInteraction": 2,
    "AcademicWorkPressure": 7,
    "FinancialPressure": 6,
    "AnxietyLevel": 7,
    "MoodScore": 4,
    "RelationshipIssues": 0
  }'
```

Response:
```json
{
  "stress_level": "High",
  "probabilities": {"High": 78.2, "Low": 5.1, "Medium": 16.7},
  "status": "success"
}
```
