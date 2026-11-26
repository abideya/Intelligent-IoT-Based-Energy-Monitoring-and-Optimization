import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
# Charger le dataset

df = pd.read_csv("high_performance_energy_dataset.csv")

# Conversion de la colonne temporelle
df['Date et l\'heure'] = pd.to_datetime(df['Date et l\'heure'])

# Tri chronologique
df = df.sort_values('Date et l\'heure')
features = [
    'energy',
    'voltage',
    'current',
    'power_kw',
    'hour',
    'minute',
    'day'
]

# Dataset final
data = df[features]

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
sequence_length = 20

X = []
y = []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i, 0])

X = np.array(X)
y = np.array(y)

print(X.shape)
print(y.shape)
train_size = int(len(X) * 0.8)

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]

model = Sequential()

# Première couche LSTM
model.add(
    LSTM(
        units=64,
        return_sequences=True,
        input_shape=(X_train.shape[1], X_train.shape[2])
    )
)

model.add(Dropout(0.2))

# Deuxième couche LSTM
model.add(LSTM(units=32))

model.add(Dropout(0.2))

# Couche de sortie
model.add(Dense(1))

model.compile(
    optimizer='adam',
    loss='mae'
)

model.summary()
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
history = model.fit(
    X_train,
    y_train,
    epochs=150,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("MAE :", mae)
print("RMSE :", rmse)


mean_real_value = np.mean(y_test)

accuracy = (1 - (mae / mean_real_value)) * 100

print("Accuracy approximative : {:.2f}%".format(accuracy))



plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title('Courbe d’apprentissage du modèle LSTM')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.legend()
plt.show()

plt.figure(figsize=(12,6))

plt.plot(y_test, label='Valeurs réelles')
plt.plot(predictions, label='Valeurs prédites')

plt.title('Prédiction énergétique avec LSTM')
plt.xlabel('Temps')
plt.ylabel('Énergie')

plt.legend()
plt.show()
threshold = 0.15

for i in range(len(predictions)):

    error = abs(y_test[i] - predictions[i])

    if error > threshold:
        print("Anomalie détectée à l'indice :", i)
        
from sklearn.metrics import r2_score

predictions_inv = scaler.inverse_transform(
    np.concatenate([predictions, np.zeros((len(predictions), data.shape[1]-1))], axis=1)
)[:,0]

y_test_inv = scaler.inverse_transform(
    np.concatenate([y_test.reshape(-1,1), np.zeros((len(y_test), data.shape[1]-1))], axis=1)
)[:,0]

r2 = r2_score(y_test_inv, predictions_inv)