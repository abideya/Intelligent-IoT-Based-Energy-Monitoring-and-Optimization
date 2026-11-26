import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, f1_score

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("lstm_test.csv")
df = df.dropna(subset=["energy", "night_id"])

night_energy = df.groupby("night_id")["energy"].agg([
    "sum", "mean", "max", "min", "std"
]).reset_index()

# =========================
# LABELS (IMPORTANT FIX)
# =========================
threshold_label = night_energy["sum"].quantile(0.95)
night_energy["label"] = (night_energy["sum"] > threshold_label).astype(int)

# =========================
# LOAD MODEL
# =========================
model = load_model(r"C:\Users\MSI\Desktop\projetpfe\train\lstm_autoencoder.keras")
scaler = joblib.load(r"C:\Users\MSI\Desktop\projetpfe\train\scaler.save")
threshold = joblib.load(r"C:\Users\MSI\Desktop\projetpfe\train\threshold.save")

print("Threshold:", threshold)

# =========================
# FEATURES
# =========================
data = night_energy.drop(["night_id", "label"], axis=1).values
labels = night_energy["label"].values

data_scaled = scaler.transform(data)

# =========================
# SEQUENCES SAFE
# =========================
SEQ_LENGTH = 5

def create_sequences(data, labels, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(labels[i+seq_length-1])
    return np.array(X), np.array(y)

X_test, y_true = create_sequences(data_scaled, labels, SEQ_LENGTH)

# =========================
# DEBUG CHECK (IMPORTANT)
# =========================
print("X_test shape:", X_test.shape)
print("y_true shape:", y_true.shape)

if len(X_test) == 0:
    raise ValueError("❌ Pas assez de données pour SEQ_LENGTH = 5")

# =========================
# PREDICTION
# =========================
X_pred = model.predict(X_test)

loss = np.mean(np.abs(X_test - X_pred), axis=(1, 2))

y_pred = (loss > threshold).astype(int)

# =========================
# FINAL DEBUG
# =========================
print("y_true unique:", np.unique(y_true, return_counts=True))
print("y_pred unique:", np.unique(y_pred, return_counts=True))

# =========================
# EVALUATION
# =========================
print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(y_true, y_pred))

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_true, y_pred, zero_division=0))

print("\n===== F1 SCORE =====")
print(f1_score(y_true, y_pred))
