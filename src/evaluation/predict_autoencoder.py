import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# =========================
# LOAD MODEL
# =========================

model = load_model(r"C:\Users\MSI\Desktop\projetpfe\train\autoencoder_lstm.keras")
scaler = joblib.load(r"C:\Users\MSI\Desktop\projetpfe\train\scaler.save")

# =========================
# LOAD DATA (TRAIN + TEST SANS NUIT 20)
# =========================

df = pd.read_csv("lstm_test.csv")
df = df.sort_values(["night_id", "datetime"])

# =========================
# SIMULATION : on enlève la nuit 20
# =========================

df_known = df[df["night_id"] != 20]

# =========================
# PRENDRE DERNIÈRE FENÊTRE (NIGHT 19)
# =========================

data = df_known["energy"].values.reshape(-1, 1)
data_scaled = scaler.transform(data)

seq_length = 24

last_sequence = data_scaled[-seq_length:]   # dernière fenêtre connue

# reshape pour LSTM
input_seq = last_sequence.reshape(1, seq_length, 1)

# =========================
# RECONSTRUCTION (PREDICTION)
# =========================

pred_scaled = model.predict(input_seq)

# inverse scaling
pred_energy = scaler.inverse_transform(pred_scaled.reshape(-1, 1))

# =========================
# FAIRE UNE NUIT COMPLÈTE (24 valeurs)
# =========================

predicted_night_20 = pred_energy.flatten()
# =========================
# TOTAL ENERGY NIGHT 20
# =========================

total_energy_night_20 = np.sum(predicted_night_20)

print("\n===== TOTAL CONSUMPTION - NIGHT 20 =====")
print("Total energy:", total_energy_night_20)
# =========================
# AFFICHAGE
# =========================

print("\n===== PREDICTION NUIT 20 =====")
for i, val in enumerate(predicted_night_20):
    print(f"Hour {i+1}: {val:.4f}")

# =========================
# PLOT
# =========================

plt.figure(figsize=(10,5))
plt.plot(predicted_night_20, marker='o')
plt.title("Predicted Energy Consumption - Night 20")
plt.xlabel("Hour")
plt.ylabel("Energy")
plt.grid()
plt.show()
# =========================
# TRUE VALUES (NIGHT 20)
# =========================

df_night20 = df[df["night_id"] == 20].sort_values("datetime")

true_night_20 = df_night20["energy"].values

# =========================
# ALIGNEMENT DES TAILLES
# =========================

min_len = min(len(true_night_20), len(predicted_night_20))

true_night_20 = true_night_20[:min_len]
pred_night_20 = predicted_night_20[:min_len]

# =========================
# METRICS
# =========================

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(true_night_20, pred_night_20)
mse = mean_squared_error(true_night_20, pred_night_20)
r2 = r2_score(true_night_20, pred_night_20)

print("\n===== NIGHT 20 COMPARISON =====")
print("MAE :", mae)
print("MSE :", mse)
print("R²  :", r2)

# =========================
# PLOT COMPARISON
# =========================

plt.figure(figsize=(10,5))

plt.plot(true_night_20, label="Real Consumption", marker='o')
plt.plot(pred_night_20, label="Predicted Consumption", marker='x')

plt.title("Night 20: Real vs Predicted Energy")
plt.xlabel("Hour")
plt.ylabel("Energy")
plt.legend()
plt.grid()
plt.show()