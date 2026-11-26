import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# 1. CHARGEMENT DATASET
# =========================
df = pd.read_csv("high_performance_energy_dataset.csv")

df['Date et l\'heure'] = pd.to_datetime(df['Date et l\'heure'])
df = df.sort_values('Date et l\'heure')

features = ['energy', 'voltage', 'current', 'power_kw', 'hour', 'minute', 'day']
data = df[features]

# =========================
# 2. NORMALISATION
# =========================
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# =========================
# 3. CREATION SEQUENCES
# =========================
sequence_length = 20

X, y = [], []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i, 0])  # energy

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

# =========================
# 4. TRAIN / TEST SPLIT
# =========================
train_size = int(len(X) * 0.8)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# =========================
# 5. MODELE LSTM
# =========================
model = Sequential()

model.add(LSTM(64, return_sequences=True,
               input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))

model.add(LSTM(32))
model.add(Dropout(0.2))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mae')

model.summary()

# =========================
# 6. TRAINING
# =========================
early_stop = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=150,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# =========================
# 7. PREDICTION
# =========================
predictions = model.predict(X_test)

# =========================
# 8. INVERSE TRANSFORM CORRECT
# =========================
def inverse_energy(values):
    temp = np.zeros((len(values), data.shape[1]))
    temp[:, 0] = values
    return scaler.inverse_transform(temp)[:, 0]

y_test_inv = inverse_energy(y_test)
predictions_inv = inverse_energy(predictions.flatten())

# =========================
# 9. METRICS
# =========================
mae = mean_absolute_error(y_test_inv, predictions_inv)
rmse = np.sqrt(mean_squared_error(y_test_inv, predictions_inv))
r2 = r2_score(y_test_inv, predictions_inv)

print("\n===== METRICS =====")
print("MAE :", mae)
print("RMSE :", rmse)
print("R² Score :", r2)

accuracy = (1 - mae / np.mean(y_test_inv)) * 100
print("Accuracy approx : {:.2f}%".format(accuracy))

# =========================
# 10. LOSS CURVES
# =========================
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Learning Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# =========================
# 11. PREDICTION VS REAL
# =========================
plt.figure(figsize=(12,6))
plt.plot(y_test_inv, label='Real Energy')
plt.plot(predictions_inv, label='Predicted Energy')
plt.title("Energy Prediction with LSTM")
plt.xlabel("Time")
plt.ylabel("Energy")
plt.legend()
plt.show()

# =========================
# 12. ANOMALY DETECTION (SMART)
# =========================
errors = np.abs(y_test_inv - predictions_inv)

threshold = np.mean(errors) + 3 * np.std(errors)

print("\n===== ANOMALIES =====")
for i in range(len(errors)):
    if errors[i] > threshold:
        print("Anomaly detected at index:", i, "Error:", errors[i])