import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

early = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3
)
df = pd.read_csv("lstm_train.csv")
df = df.sort_values(["night_id", "datetime"])

nights = []

for night_id in df["night_id"].unique():
    night_data = df[df["night_id"] == night_id]["energy"].values
    nights.append(night_data)

nights = np.array(nights)

print("Shape nights:", nights.shape)

scaler = MinMaxScaler()
nights_scaled = scaler.fit_transform(nights.reshape(-1, 1)).reshape(nights.shape)

X, y = [], []

for i in range(len(nights_scaled) - 1):
    X.append(nights_scaled[i])
    y.append(nights_scaled[i + 1])

X = np.array(X)
y = np.array(y)

X = X.reshape((X.shape[0], X.shape[1], 1))
y = y.reshape((y.shape[0], y.shape[1], 1))

seq_length = X.shape[1]

inputs = Input(shape=(seq_length, 1))

x = LSTM(128, return_sequences=True)(inputs)
x = LSTM(64, return_sequences=True)(x)
x = LSTM(32)(x)

outputs = Dense(seq_length)(x)  # amélioration clé

model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='mse')

model.summary()

history = model.fit(
    X, y,
    epochs=100,
    batch_size=4,
    validation_split=0.2,
    shuffle=False,
    callbacks=[early, lr]
)

model.save("lstm_night_forecasting.keras")
joblib.dump(scaler, "scaler.save")

print("Model trained and saved")
