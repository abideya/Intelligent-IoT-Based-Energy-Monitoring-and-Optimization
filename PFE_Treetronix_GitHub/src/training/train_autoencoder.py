import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("lstm_train.csv")
df = df.dropna(subset=["energy", "night_id"])

# =========================
# GROUP BY NIGHT
# =========================
night_energy = df.groupby("night_id")["energy"].agg([
    "sum", "mean", "max", "min", "std"
]).reset_index()

# =========================
# LABEL (ANOMALY SIMULATION)
# =========================
threshold_label = night_energy["sum"].quantile(0.95)
night_energy["label"] = (night_energy["sum"] > threshold_label).astype(int)

print(night_energy.head())

# =========================
# FEATURES
# =========================
data = night_energy.drop(["night_id", "label"], axis=1).values
labels = night_energy["label"].values

# =========================
# SCALE
# =========================
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# =========================
# SEQUENCES
# =========================
SEQ_LENGTH = 5

def create_sequences(data, labels, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(labels[i+seq_length-1])  # label aligné
    return np.array(X), np.array(y)

X, y_true = create_sequences(data_scaled, labels, SEQ_LENGTH)

# =========================
# MODEL LSTM AUTOENCODER
# =========================
features = X.shape[2]

inputs = Input(shape=(SEQ_LENGTH, features))

encoded = LSTM(64, activation="relu")(inputs)
decoded = RepeatVector(SEQ_LENGTH)(encoded)
decoded = LSTM(64, return_sequences=True, activation="relu")(decoded)
outputs = TimeDistributed(Dense(features))(decoded)

model = Model(inputs, outputs)
model.compile(optimizer="adam", loss="mse")

model.fit(X, X, epochs=100, batch_size=16, shuffle=False)

# =========================
# SAVE MODEL + SCALER
# =========================
model.save("lstm_autoencoder.keras")
joblib.dump(scaler, "scaler.save")

# =========================
# THRESHOLD (IMPORTANT FIX)
# =========================
X_pred = model.predict(X)

loss = np.mean(np.abs(X - X_pred), axis=(1, 2))

# 🔥 MEILLEURE STRATÉGIE (IMPORTANT)
threshold = np.percentile(loss[y_true == 0], 95)

joblib.dump(threshold, "threshold.save")

print("Training terminé ✔")
print("Threshold:", threshold)