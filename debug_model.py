import joblib
import tensorflow as tf
import pandas as pd
import numpy as np
import os

# 1. Load the files directly
print("--- LOADING FILES ---")
scaler = joblib.load('models/scaler.pkl')
model = tf.keras.models.load_model('models/heart_disease_nn_model.keras')
print("✅ Files loaded.")

# 2. Define the "Super Healthy" Input
# Age 25, Chol 150, BP 110, etc.
feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
raw_input = pd.DataFrame([[25, 1, 0, 110, 150, 0, 0, 175, 0, 0.0, 2, 0, 2]], columns=feature_names)

print("\n--- RAW INPUT (Before Scaling) ---")
print(raw_input[['age', 'chol']].to_string(index=False))

# 3. Apply Scaling (The Critical Step)
num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
try:
    # Attempt to scale
    raw_input[num_cols] = scaler.transform(raw_input[num_cols])
    print("\n✅ SCALING SUCCESSFUL")
    print("--- SCALED INPUT (What the AI sees) ---")
    print(raw_input[['age', 'chol']].to_string(index=False))
except Exception as e:
    print(f"\n❌ SCALING FAILED: {e}")

# 4. Predict
prob = model.predict(raw_input.values)[0][0]
print(f"\n--- FINAL PREDICTION ---")
print(f"Probability: {prob:.4f}")
print(f"Result: {'HIGH RISK' if prob > 0.5 else 'LOW RISK'}")