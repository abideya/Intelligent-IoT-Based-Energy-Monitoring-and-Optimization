
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# LOAD
# =========================

model = load_model(r"C:\Users\MSI\Desktop\projetpfe\train\autoencoder_lstm.keras")
scaler = joblib.load(r"C:\Users\MSI\Desktop\projetpfe\train\scaler.save")
threshold = joblib.load(r"C:\Users\MSI\Desktop\projetpfe\train\threshold_ae.save")

# =========================
# LOAD DATA
# =========================

df = pd.read_csv("lstm_test.csv")
df = df.sort_values(["night_id", "datetime"])

data = df["energy"].values.reshape(-1,1)
data_scaled = scaler.transform(data)

# =========================
# SEQUENCES
# =========================

seq_length = 24

def create_sequences(data):
    X = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
    return np.array(X)

X_test = create_sequences(data_scaled)

# =========================
# NIGHT ID MAPPING (IMPORTANT)
# =========================

night_ids = df["night_id"].values
window_night_ids = []

for i in range(len(data_scaled) - seq_length):
    window_night_ids.append(night_ids[i + seq_length - 1])

window_night_ids = np.array(window_night_ids)

# =========================
# PREDICTION
# =========================

X_pred = model.predict(X_test)

# =========================
# METRICS (MAE - MSE - R²)
# =========================

mae = mean_absolute_error(X_test.reshape(-1), X_pred.reshape(-1))
mse = mean_squared_error(X_test.reshape(-1), X_pred.reshape(-1))
r2 = r2_score(X_test.reshape(-1), X_pred.reshape(-1))

print("\n===== PERFORMANCE =====")
print("MAE :", mae)
print("MSE :", mse)
print("R²  :", r2)

# =========================
# LOSS
# =========================

loss = np.mean(np.abs(X_test - X_pred), axis=(1,2))

# =========================
# ANOMALY DETECTION
# =========================

anomalies = loss > threshold

smooth = pd.Series(anomalies).rolling(3).sum()
final_anomalies = smooth >= 2

# =========================
# RESULTS DATAFRAME
# =========================

results = pd.DataFrame({
    "night_id": window_night_ids,
    "loss": loss,
    "anomaly": final_anomalies
})

print(results.head())
print("\nTotal anomalies:", final_anomalies.sum())

# =========================
# 🟢 CLASSIFICATION PAR NUIT
# =========================

# 1. ratio d'anomalies par nuit
night_anomaly_ratio = results.groupby("night_id")["anomaly"].mean().reset_index()
night_anomaly_ratio.columns = ["night_id", "anomaly_ratio"]

# 2. consommation moyenne par nuit (sur données scalées)
df["energy_scaled"] = data_scaled
night_mean = df.groupby("night_id")["energy_scaled"].mean().reset_index()
night_mean.columns = ["night_id", "mean_energy"]

# 3. fusion
night_stats = night_mean.merge(night_anomaly_ratio, on="night_id", how="left")
night_stats["anomaly_ratio"] = night_stats["anomaly_ratio"].fillna(0)

# =========================
# Z-SCORE POUR TYPE DE CONSOMMATION
# =========================

global_mean = night_stats["mean_energy"].mean()
global_std = night_stats["mean_energy"].std()

night_stats["z_score"] = (night_stats["mean_energy"] - global_mean) / global_std

def classify(z):
    if z > 0.5:
        return "SURCONSOMMATION 🔴"
    elif z < -0.5:
        return "SOUS-CONSOMMATION 🔵"
    else:
        return "NORMAL 🟢"

night_stats["class"] = night_stats["z_score"].apply(classify)

# =========================
# FINAL DECISION (HYBRIDE)
# =========================

def final_status(row):
    if row["anomaly_ratio"] > 0.3:
        return "ANOMALIE ⚠️"
    return row["class"]

night_stats["final_status"] = night_stats.apply(final_status, axis=1)

# =========================
# RESULTATS FINAUX
# =========================

print("\n===== NIGHT CLASSIFICATION =====")
print(night_stats[["night_id", "mean_energy", "z_score", "anomaly_ratio", "class", "final_status"]])

print("\n===== SUMMARY =====")
print(night_stats["final_status"].value_counts())

# =========================
# PLOT
# =========================

plt.figure(figsize=(12,5))
plt.plot(loss, label="Reconstruction Error")
plt.axhline(threshold, color='red', linestyle='--')
plt.title("Autoencoder Anomaly Detection")
plt.legend()
plt.show()