import joblib
import tensorflow as tf
import pandas as pd
import numpy as np
import os

# 1. Load the files
print("--- LOADING FILES ---")
scaler = joblib.load('models/scaler.pkl')
model = tf.keras.models.load_model('models/heart_disease_nn_model.keras')
print("Files loaded.")

# 2. Define the "Healthy Button" Input from app.py
# 'age': 55, 'sex': 1, 'cp': 0, 'trestbps': 140, 'chol': 217,
# 'fbs': 'No', 'restecg': 1, 'thalach': 111, 'exang': 'Yes',
# 'oldpeak': 5.6, 'slope': 0, 'ca': 0, 'thal': 3

feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# Convert 'No'/'Yes' to 0/1 where applicable, based on app.py logic
# fbs: 'No' -> 0
# exang: 'Yes' -> 1
values = [55, 1, 0, 140, 217, 0, 1, 111, 1, 5.6, 0, 0, 3]

raw_input = pd.DataFrame([values], columns=feature_names)

print("\n--- RAW INPUT (Healthy Button Values) ---")
print(raw_input.to_string(index=False))

# 3. Apply Scaling
num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
try:
    raw_input[num_cols] = scaler.transform(raw_input[num_cols])
except Exception as e:
    print(f"\n❌ SCALING FAILED: {e}")
    exit()

# 4. Predict
prob = model.predict(raw_input.values)[0][0]
print(f"\n--- FINAL PREDICTION ---")
print(f"Probability: {prob:.4f}")
print(f"Result: {'HIGH RISK' if prob > 0.5 else 'LOW RISK'}")
