import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
import requests

# =========================
# LOAD MODEL + DATA
# =========================

model = load_model(r"C:\Users\MSI\Desktop\projetpfe\train\lstm_model.keras")
scaler = joblib.load(r"C:\Users\MSI\Desktop\projetpfe\train\scaler.save")
threshold = joblib.load(r"C:\Users\MSI\Desktop\projetpfe\train\threshold.save")

df_test = pd.read_csv("test2.csv")
df_test = df_test.sort_values(["night_id", "datetime"])

# =========================
# ENERGY PER NIGHT
# =========================

test_nights = df_test.groupby("night_id")["energy"].sum().reset_index()

test_nights["energy_scaled"] = scaler.transform(
    test_nights[["energy"]]
)

# =========================
# CREATE SEQUENCES
# =========================

def create_sequences(data, seq_length=5):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 5
data = test_nights["energy_scaled"].values

X_test, y_test = create_sequences(data, seq_length)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# =========================
# PREDICTION
# =========================

y_pred = model.predict(X_test)

y_test_real = scaler.inverse_transform(y_test.reshape(-1,1))
y_pred_real = scaler.inverse_transform(y_pred)

# =========================
# ERROR + ANOMALY LOGIC
# =========================

errors = y_test_real.flatten() - y_pred_real.flatten()

anomalies = np.abs(errors) > threshold

anomaly_type = []

for e in errors:
    if e > threshold:
        anomaly_type.append("Overconsumption")   # 🔴
    elif e < -threshold:
        anomaly_type.append("Underconsumption")  # 🔵
    else:
        anomaly_type.append("Normal")            # 🟢

# =========================
# RESULTS
# =========================

results = pd.DataFrame({
    "real": y_test_real.flatten(),
    "predicted": y_pred_real.flatten(),
    "error": errors,
    "anomaly": anomalies,
    "type": anomaly_type
})

print(results)

# =========================
# PLOT DESIGN
# =========================

plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(18,10))

fig.suptitle(
    "LSTM Anomaly Detection (Over / Under Consumption)",
    fontsize=22,
    fontweight='bold'
)

# =========================
# GRAPH 1
# =========================

ax1 = plt.subplot2grid((2,3), (0,0), colspan=2)

x = np.arange(len(results))

ax1.plot(x, results["real"], label="Real Energy", linewidth=3)
ax1.plot(x, results["predicted"], label="Predicted Energy", linestyle="--", linewidth=3)

# Colors for anomalies
colors = []

for t in results["type"]:
    if t == "Overconsumption":
        colors.append("red")
    elif t == "Underconsumption":
        colors.append("blue")
    else:
        colors.append("green")

# Scatter points
ax1.scatter(x, results["real"], c=colors, s=120, marker='o', label="Status")

ax1.set_title("Real vs Predicted Energy")
ax1.set_xlabel("Night Index")
ax1.set_ylabel("Energy (kWh)")
ax1.legend()

# =========================
# SUMMARY BOX
# =========================

ax2 = plt.subplot2grid((2,3), (0,2))
ax2.axis('off')

total_over = sum(np.array(results["type"]) == "Overconsumption")
total_under = sum(np.array(results["type"]) == "Underconsumption")

summary_text = f"""
SUMMARY

Overconsumption : {total_over}
Underconsumption : {total_under}

Threshold : {threshold:.3f}
"""

ax2.text(
    0.05, 0.5, summary_text,
    fontsize=14,
    bbox=dict(facecolor='whitesmoke', edgecolor='gray', boxstyle='round,pad=1')
)

# =========================
# ERROR BAR
# =========================

ax3 = plt.subplot2grid((2,3), (1,0))

bar_colors = []

for t in results["type"]:
    if t == "Overconsumption":
        bar_colors.append("red")
    elif t == "Underconsumption":
        bar_colors.append("blue")
    else:
        bar_colors.append("green")

ax3.bar(x, errors, color=bar_colors)
ax3.axhline(threshold, color='red', linestyle='--')
ax3.axhline(-threshold, color='blue', linestyle='--')

ax3.set_title("Error per Night")
ax3.set_xlabel("Night")
ax3.set_ylabel("Error")

# =========================
# HISTOGRAM
# =========================

ax4 = plt.subplot2grid((2,3), (1,1))

ax4.hist(errors, bins=10, edgecolor='black')
ax4.axvline(threshold, color='red')
ax4.axvline(-threshold, color='blue')

ax4.set_title("Error Distribution")
ax4.set_xlabel("Error")
ax4.set_ylabel("Frequency")

# =========================
# TABLE
# =========================

ax5 = plt.subplot2grid((2,3), (1,2))
ax5.axis('off')

anomaly_table = results[results["anomaly"]]

table_data = []

for idx, row in anomaly_table.iterrows():
    table_data.append([
        idx,
        round(row["real"],3),
        round(row["predicted"],3),
        round(row["error"],3),
        row["type"]
    ])

table = ax5.table(
    cellText=table_data,
    colLabels=["Night", "Real", "Predicted", "Error", "Type"],
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 2)

ax5.set_title("Detected Anomalies")

# =========================
# SEND ALERT TO TAGOIO
# =========================

if results["anomaly"].any():

    payload = {
        "variable": "anomaly",
        "value": 1,
        "metadata": {
            "message": "Over/Under consumption detected"
        }
    }

    requests.post(
        "https://api.tago.io/data",
        json=payload,
        headers={
            "Content-Type": "application/json",
            "Device-Token": "26c9c076-a10d-4e22-b193-6cff95a39bd4"
        }
    )

# =========================
# SHOW
# =========================

plt.tight_layout(rect=[0,0.04,1,0.95])
plt.show()