from tensorflow.keras.models import load_model
import joblib
model = load_model(r"C:\Users\MSI\Desktop\projetpfe\train\lstm_night_forecasting.keras")
scaler = joblib.load(r"C:\Users\MSI\Desktop\projetpfe\train\scaler.save")
