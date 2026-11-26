import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# =========================
# LOAD DATA (NORMAL ONLY)
# =========================

df_train = pd.read_csv("lstm_train.csv")
df_train = df_train.sort_values(["night_id", "datetime"])

train_nights = df_train.groupby("night_id")["energy"].sum().reset_index()

# =========================
# SCALING
# =========================

scaler = MinMaxScaler()
train_nights["energy_scaled"] = scaler.fit_transform(train_nights[["energy"]])

# =========================
# SEQUENCES
# =========================

def create_sequences(data, seq_length=9):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 9
data = train_nights["energy_scaled"].values

X_train, y_train = create_sequences(data, seq_length)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

# =========================
# MODEL
# =========================

model = Sequential([
    LSTM(128, activation='tanh', input_shape=(seq_length, 1)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=30, batch_size=8, shuffle=False)

# =========================
# SAVE MODEL + SCALER
# =========================

model.save("lstm_model.keras")
joblib.dump(scaler, "scaler.save")

# =========================
# THRESHOLD (IMPORTANT FIX)
# =========================

train_pred = model.predict(X_train)

y_train_real = scaler.inverse_transform(y_train.reshape(-1,1))
train_pred_real = scaler.inverse_transform(train_pred)

train_errors = y_train_real.flatten() - train_pred_real.flatten()

mean_err = np.mean(train_errors)
std_err = np.std(train_errors)

threshold = 2.5 * std_err   # ✔ FIXED

joblib.dump(threshold, "threshold.save")

print("Model + scaler + threshold saved")