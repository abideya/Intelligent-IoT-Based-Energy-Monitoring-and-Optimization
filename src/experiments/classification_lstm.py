# ============================================================
# LSTM Classification for Night Energy Anomaly Detection
# ============================================================

# =========================
# 1. IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt

# =========================
# 2. LOAD DATASET
# =========================
df = pd.read_csv("data.csv")

print(df.head())

# =========================
# 3. CALCULATE TOTAL ENERGY PER NIGHT
# =========================

# Sum all energy values for each night
nightly_data = df.groupby("day").agg({
    "energy": "sum",
    "anomaly": "max"
}).reset_index()

# Rename column
nightly_data.rename(columns={"energy": "total_energy_kWh"}, inplace=True)

print("\nNightly Dataset:")
print(nightly_data.head())

# =========================
# 4. NORMALIZATION
# =========================

scaler = MinMaxScaler()

nightly_data["scaled_energy"] = scaler.fit_transform(
    nightly_data[["total_energy_kWh"]]
)

# =========================
# 5. CREATE LSTM SEQUENCES
# =========================

sequence_length = 5

X = []
y = []

energy_values = nightly_data["scaled_energy"].values
labels = nightly_data["anomaly"].values

for i in range(len(energy_values) - sequence_length):

    X.append(energy_values[i:i+sequence_length])

    y.append(labels[i+sequence_length])

X = np.array(X)
y = np.array(y)

# reshape for LSTM
X = X.reshape((X.shape[0], X.shape[1], 1))

print("\nShape of X:", X.shape)
print("Shape of y:", y.shape)

# =========================
# 6. TRAIN / TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

# =========================
# 7. BUILD LSTM MODEL
# =========================

model = Sequential()

model.add(
    LSTM(
        32,
        input_shape=(sequence_length, 1)
    )
)

model.add(Dropout(0.2))


model.add(Dense(1, activation='sigmoid'))

# =========================
# 8. COMPILE MODEL
# =========================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================
# 9. TRAIN MODEL
# =========================

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=4,
    validation_data=(X_test, y_test),
    verbose=1
)

# =========================
# 10. PREDICTIONS
# =========================

y_pred_prob = model.predict(X_test)

# Convert probabilities to classes
y_pred = (y_pred_prob > 0.5).astype(int)

# =========================
# 11. EVALUATION
# =========================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# =========================
# 12. PLOT TRAINING CURVES
# =========================

plt.figure(figsize=(10,5))

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.show()

# =========================
# 13. VISUALIZE ANOMALIES
# =========================

plt.figure(figsize=(12,5))

plt.plot(
    nightly_data["total_energy_kWh"],
    label="Night Energy Consumption"
)

# Plot anomaly points
anomaly_indices = nightly_data[
    nightly_data["anomaly"] == 1
].index

plt.scatter(
    anomaly_indices,
    nightly_data.loc[anomaly_indices, "total_energy_kWh"],
    color='red',
    label='Anomaly'
)

plt.title("Night Energy Consumption with Anomalies")

plt.xlabel("Night")

plt.ylabel("Energy (kWh)")

plt.legend()

plt.show()