import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
#Load model + scaler + threshold
model = load_model(r"C:\Users\MSI\Desktop\projet pfe\lstm_model.keras")

scaler = joblib.load(r"C:\Users\MSI\Desktop\projet pfe\scaler.save")

threshold = joblib.load(r"C:\Users\MSI\Desktop\projet pfe\threshold.save")
# Load test data
df_test = pd.read_csv("lstm_test.csv")
df_test = df_test.sort_values(["night_id", "datetime"])

test_nights = df_test.groupby("night_id")["energy"].sum().reset_index()

#Scaling test
test_nights["energy_scaled"] = scaler.transform(test_nights[["energy"]])

#sequences
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

#prediction
y_pred = model.predict(X_test)
#inverser scaling
y_test_real = scaler.inverse_transform(y_test.reshape(-1,1))
y_pred_real = scaler.inverse_transform(y_pred)

errors = np.abs(y_test_real - y_pred_real)
#anomaly
anomalies = errors > threshold
#results
results = pd.DataFrame({
    "real": y_test_real.flatten(),
    "predicted": y_pred_real.flatten(),
    "error": errors.flatten(),
    "anomaly": anomalies.flatten()
})

print(results)
print("Total anomalies:", anomalies.sum())