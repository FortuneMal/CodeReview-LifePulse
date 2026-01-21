import joblib
import tensorflow as tf
import pandas as pd
import numpy as np

# 1. Load the files
print("Loading files...")
scaler = joblib.load('models/scaler.pkl')
model = tf.keras.models.load_model('models/heart_disease_nn_model.keras')
print("Files loaded.")

# 2. Define the PROPOSED "Healthy Button" Input
# Age: 50, Sex: 1, CP: 2, BP: 120, Chol: 200, FBS: 0, RestECG: 0, Thalach: 160, Exang: 0, Oldpeak: 0.0, Slope: 0, CA: 0, Thal: 1
values = [50, 1, 2, 120, 200, 0, 0, 160, 0, 0.0, 0, 0, 1]
feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

raw_input = pd.DataFrame([values], columns=feature_names)

print("\n--- NEW HEALTHY INPUT ---")
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
print(f"Result: {'HIGH RISK' if prob > 0.5 else 'LOW RISK'}")
