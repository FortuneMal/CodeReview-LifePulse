import joblib
import tensorflow as tf
import pandas as pd
import numpy as np

# 1. Load the files
print("Loading files...")
scaler = joblib.load('models/scaler.pkl')
model = tf.keras.models.load_model('models/heart_disease_nn_model.keras')
print("Files loaded.")

# helper function to simulate app logic
def get_app_result(prob):
    # New App Logic: < 0.5 is High Risk
    if prob < 0.5:
        return "HIGH RISK (Disease)"
    else:
        return "LOW RISK (Healthy)"

# Case A: New "Healthy" Button Values (Corrected)
# Age: 50, Sex: 1, CP: 2, BP: 120, Chol: 200, FBS: 0, RestECG: 0, Thalach: 160, Exang: 0, Oldpeak: 0.0, Slope: 0, CA: 0, Thal: 1
healthy_values = [50, 1, 2, 120, 200, 0, 0, 160, 0, 0.0, 0, 0, 1]

# Case B: High Risk Button Values (Existing)
# Age: 65, Sex: 1, CP: 0, BP: 160, Chol: 300, FBS: 1, RestECG: 2, Thalach: 100, Exang: 1, Oldpeak: 2.5, Slope: 1, CA: 2, Thal: 1
high_risk_values = [65, 1, 0, 160, 300, 1, 2, 100, 1, 2.5, 1, 2, 1]

feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

df = pd.DataFrame([healthy_values, high_risk_values], columns=feature_names)
num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
df[num_cols] = scaler.transform(df[num_cols])

print("\n--- RUNNING PREDICTIONS ---")
probs = model.predict(df.values)

prob_healthy = probs[0][0]
prob_risk = probs[1][0]

print(f"\n1. Healthy Button Input:")
print(f"   Probability: {prob_healthy:.4f}")
print(f"   App Result:  {get_app_result(prob_healthy)}")

print(f"\n2. High Risk Button Input:")
print(f"   Probability: {prob_risk:.4f}")
print(f"   App Result:  {get_app_result(prob_risk)}")
