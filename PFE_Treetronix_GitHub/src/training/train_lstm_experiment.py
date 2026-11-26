import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

#Load train data (NORMAL only)
df_train = pd.read_csv("lstm_train.csv")
df_train = df_train.sort_values(["night_id", "datetime"])

train_nights = df_train.groupby("night_id")["energy"].sum().reset_index()
#Scaling
scaler = MinMaxScaler()
train_nights["energy_scaled"] = scaler.fit_transform(train_nights[["energy"]])
#Sequences
def create_sequences(data, seq_length=5):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 5
data = train_nights["energy_scaled"].values

X_train, y_train = create_sequences(data, seq_length)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
#Model LSTM
model = Sequential()

model.add(LSTM(128, activation='tanh', input_shape=(seq_length, 1)))
model.add(Dropout(0.2))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))   # ✅ ajout important

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=30, batch_size=8, shuffle=False)
#SAVE MODEL + SCALER
model.save("lstm_model.keras")

import joblib

joblib.dump(scaler, "scaler.save")
#calcul seuil 
train_pred = model.predict(X_train)

y_train_real = scaler.inverse_transform(y_train.reshape(-1,1))
train_pred_real = scaler.inverse_transform(train_pred)

train_errors = np.abs(y_train_real - train_pred_real)

threshold = train_errors.mean() + 10 * train_errors.std()

joblib.dump(threshold, "threshold.save")

print("Model + scaler + threshold saved")
