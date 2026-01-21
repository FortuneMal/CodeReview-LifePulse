import joblib
import tensorflow as tf
import pandas as pd
import numpy as np

# 1. Load the files
print("Loading files...")
scaler = joblib.load('models/scaler.pkl')
model = tf.keras.models.load_model('models/heart_disease_nn_model.keras')
print("Files loaded.")

# 2. Define the "High Risk Button" Input from app.py
# 'age': 65, 'sex': 1, 'cp': 0, 'trestbps': 160, 'chol': 300,
# 'fbs': 'Yes', 'restecg': 2, 'thalach': 100, 'exang': 'Yes',
# 'oldpeak': 2.5, 'slope': 1, 'ca': 2, 'thal': 1
# fbs='Yes' -> 1
# exang='Yes' -> 1

values = [65, 1, 0, 160, 300, 1, 2, 100, 1, 2.5, 1, 2, 1]
feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

raw_input = pd.DataFrame([values], columns=feature_names)

print("\n--- HIGH RISK BUTTON INPUT ---")
print(raw_input.to_string(index=False))

# 3. Apply Scaling
num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
try:
    raw_input[num_cols] = scaler.transform(raw_input[num_cols])
except Exception as e:
    print(f"\nScaling failed: {e}")
    exit()

# 4. Predict
prob = model.predict(raw_input.values)[0][0]
print(f"\n--- PREDICTION ---")
print(f"Probability: {prob:.4f}")
print(f"Result in App (Prob > 0.5): {'HIGH RISK' if prob > 0.5 else 'LOW RISK'}")
