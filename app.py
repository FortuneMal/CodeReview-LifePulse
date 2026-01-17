import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import os

# 1. Page Configuration
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("❤️ AI Heart Disease Predictor")
st.write("Enter the patient's clinical data below to check for heart disease risk.")

# 2. Load the Model
# We use the .keras file you saved in Phase 3
@st.cache_resource
def load_my_model():
    model_path = os.path.join('models', 'heart_disease_nn_model.keras')
    return tf.keras.models.load_model(model_path)

model = load_my_model()

# 3. Create the Input Form
with st.form("prediction_form"):
    st.subheader("Patient Health Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
        cp = st.slider("Chest Pain Type (0-3)", 0, 3, 1)
        trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
        chol = st.number_input("Cholesterol", 100, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "True" if x == 1 else "False")
        
    with col2:
        restecg = st.slider("Resting ECG Results (0-2)", 0, 2, 0)
        thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
        exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 6.0, 1.0)
        slope = st.slider("Slope of Peak Exercise ST", 0, 2, 1)
        ca = st.slider("Number of Major Vessels (0-4)", 0, 4, 0)
        thal = st.slider("Thalassemia (0-3)", 0, 3, 2)

    submit = st.form_submit_button("Predict Result")

# 4. Prediction Logic
if submit:
    # Arrange inputs into the 13-feature format the model expects
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    
    # Get probability
    prediction_prob = model.predict(input_data)[0][0]
    
    st.divider()
    
    if prediction_prob > 0.5:
        st.error(f"### Result: High Risk ({prediction_prob:.2%})")
        st.write("The model suggests a high probability of heart disease. Please consult a medical professional.")
    else:
        st.success(f"### Result: Low Risk ({prediction_prob:.2%})")
        st.write("The model suggests a low probability of heart disease.")

    # Optional: Risk Gauge
    st.progress(float(prediction_prob))