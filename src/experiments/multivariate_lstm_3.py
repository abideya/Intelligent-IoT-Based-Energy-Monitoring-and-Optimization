import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("high_performance_energy_dataset.csv")

df['Date et l\'heure'] = pd.to_datetime(df['Date et l\'heure'])
df = df.sort_values('Date et l\'heure')

features = ['energy', 'voltage', 'current', 'power_kw']
data = df[features]

# =========================
# 2. NORMALISATION
# =========================
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# =========================
# 3. SEQUENCES
# =========================
SEQ_LENGTH = 20

X, y = [], []

for i in range(SEQ_LENGTH, len(scaled_data)):
    X.append(scaled_data[i-SEQ_LENGTH:i])
    y.append(scaled_data[i, 0])  # energy

X = np.array(X)
y = np.array(y)

# =========================
# 4. TRAIN / TEST
# =========================
split = int(0.8 * len(X))

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# =========================
# 5. MODEL LSTM
# =========================
model = Sequential()

model.add(LSTM(64, return_sequences=True,
               input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))

model.add(LSTM(32))
model.add(Dropout(0.2))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mae')

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# =========================
# 6. TRAINING
# =========================
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
# 8. INVERSE SCALE
# =========================
def inverse(values):
    temp = np.zeros((len(values), data.shape[1]))
    temp[:,0] = values
    return scaler.inverse_transform(temp)[:,0]

y_test_inv = inverse(y_test)
predictions_inv = inverse(predictions.flatten())

# =========================
# 9. METRICS
# =========================
mae = mean_absolute_error(y_test_inv, predictions_inv)
rmse = np.sqrt(mean_squared_error(y_test_inv, predictions_inv))
r2 = r2_score(y_test_inv, predictions_inv)

print("\n===== RESULTS =====")
print("MAE :", mae)
print("RMSE :", rmse)
print("R² :", r2)

# =========================
# 10. ANOMALY DETECTION (IMPORTANT POUR JURY)
# =========================

errors = np.abs(y_test_inv - predictions_inv)

threshold = np.mean(errors) + 0.75 * np.std(errors)  # plus sensible

anomalies = errors > threshold

print("Nombre d'anomalies :", np.sum(anomalies))

for i in range(len(errors)):
    if anomalies[i]:
        print("Anomalie à l'index :", i, "Erreur:", errors[i])

# =========================
# 11. VISUALISATION
# =========================
plt.figure(figsize=(12,5))
plt.plot(y_test_inv, label="Réel")
plt.plot(predictions_inv, label="Prédit")
plt.legend()
plt.title("Prédiction énergétique")
plt.show()

plt.figure()
plt.plot(errors, label="Erreur")
plt.axhline(threshold, color='red', label='Seuil anomalie')
plt.legend()
plt.title("Détection d'anomalies")
plt.show()