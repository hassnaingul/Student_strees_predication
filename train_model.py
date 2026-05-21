import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ── Load Data ──────────────────────────────────────────────────────────────────
df = pd.read_csv("stress_dataset.csv")
print("Dataset shape:", df.shape)
print("\nStress Level distribution:\n", df['StressLevel'].value_counts())

# ── Features & Target ─────────────────────────────────────────────────────────
feature_cols = [
    'SleepHours', 'StudyWorkHours', 'ScreenTime', 'PhysicalActivity',
    'SocialInteraction', 'AcademicWorkPressure', 'FinancialPressure',
    'AnxietyLevel', 'MoodScore', 'RelationshipIssues'
]

X = df[feature_cols].values
y = df['StressLevel'].values

# Encode labels: Low=1, Medium=2, High=0 (alphabetical by LabelEncoder)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("\nLabel classes:", le.classes_)

# ── Train / Test Split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ── Scale ──────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── Train MLP ─────────────────────────────────────────────────────────────────
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    max_iter=1000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20
)
mlp.fit(X_train_scaled, y_train)

# ── Evaluate ───────────────────────────────────────────────────────────────────
y_pred = mlp.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# ── Save Confusion Matrix Plot ─────────────────────────────────────────────────
os.makedirs("static", exist_ok=True)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix – Student Stress Level', fontsize=14)
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig("static/confusion_matrix.png", dpi=150)
plt.close()

# ── Save Loss Curve ────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(mlp.loss_curve_, color='dodgerblue', linewidth=2)
plt.title('MLP Training Loss Curve', fontsize=14)
plt.xlabel('Epochs'); plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("static/loss_curve.png", dpi=150)
plt.close()

# ── Persist Model Artifacts ───────────────────────────────────────────────────
with open("model.pkl",   "wb") as f: pickle.dump(mlp, f)
with open("scaler.pkl",  "wb") as f: pickle.dump(scaler, f)
with open("encoder.pkl", "wb") as f: pickle.dump(le, f)

print("\n✅ model.pkl, scaler.pkl, encoder.pkl saved.")
print(f"✅ Plots saved to static/")
print(f"\nFinal Model Accuracy: {acc*100:.2f}%")
