import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# LOAD DATA
# =========================

df = pd.read_csv("lstm_train.csv")
df = df.sort_values(["night_id", "datetime"])

# SOMME PAR NIGHT_ID
night_energy = df.groupby("night_id")["energy"].sum().reset_index()

night_energy.columns = ["night_id", "total_energy"]

print(night_energy.head())

# =========================
# SCALING
# =========================

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# =========================
# SEQUENCES
# =========================

def create_sequences(data, seq_length):
    X = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
    return np.array(X)

SEQ_LENGTH = 5
X = create_sequences(data_scaled, SEQ_LENGTH)

# =========================
# AUTOENCODER MODEL
# =========================

inputs = Input(shape=(SEQ_LENGTH, 1))

encoded = LSTM(64, activation='relu', return_sequences=False)(inputs)
decoded = RepeatVector(SEQ_LENGTH)(encoded)
decoded = LSTM(64, activation='relu', return_sequences=True)(decoded)
outputs = TimeDistributed(Dense(1))(decoded)

autoencoder = Model(inputs, outputs)
autoencoder.compile(optimizer='adam', loss='mse')

# =========================
# CALLBACKS
# =========================

early_stop = EarlyStopping(
    monitor='loss',
    patience=10,
    restore_best_weights=True
)

# =========================
# TRAIN
# =========================

history = autoencoder.fit(
    X, X,
    epochs=100,
    batch_size=16,
    shuffle=False,
    callbacks=[early_stop]
)

# =========================
# SAVE MODEL + SCALER
# =========================

autoencoder.save("autoencoder_lstm.keras")
joblib.dump(scaler, "scaler.save")

# =========================
# THRESHOLD (TRAIN ERROR)
# =========================

X_pred = autoencoder.predict(X)

loss = np.mean(np.abs(X - X_pred), axis=(1, 2))  # reconstruction error

threshold = np.percentile(loss, 95)

joblib.dump(threshold, "threshold_ae.save")

print("Autoencoder trained successfully")
print("Threshold:", threshold)