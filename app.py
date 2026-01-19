import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import os

# 1. Page Config & Custom "Cool" Styling
st.set_page_config(page_title="Pulse AI", page_icon="❤️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    h1 { color: #ff4b4b; }
    div.stButton > button:first-child {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-weight: bold;
        border: none;
    }
    .stNumberInput, .stSelectbox, .stSlider {
        background-color: #161b22;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Load Assets (Caching prevents reloading on every click)
@st.cache_resource
def load_assets():
    # Load the Neural Network
    model_path = os.path.join('models', 'heart_disease_nn_model.keras')
    model = tf.keras.models.load_model(model_path)
    
    # Load the Scaler
    scaler_path = os.path.join('models', 'scaler.pkl')
    scaler = joblib.load(scaler_path)
    
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Error loading files: {e}. Please check your /models folder.")
    st.stop()

# 3. Header Section
st.title("❤️ LifePulse: Heart Health Monitor")
st.markdown("### *Our AI is quite accurate. Your heart should be too.*")

# 4. Input Form (The "Clinical" Layout)
with st.container():
    col1, col2, col3 = st.columns([1, 1, 1], gap="medium")
    
    with col1:
        st.subheader("📋 Basics")
        age = st.number_input("Age", 1, 120, 25) # Default 25 (Low Risk)
        sex = st.selectbox("Biological Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], help="0: Asymptomatic, 1: Atypical, 2: Non-Anginal, 3: Typical Angina")
        trestbps = st.slider("Resting Blood Pressure", 80, 200, 110) # Default 110 (Low Risk)

    with col2:
        st.subheader("🧪 Labs")
        chol = st.number_input("Cholesterol Level", 100, 600, 150) # Default 150 (Low Risk)
        # Convert Yes/No to 1/0 immediately
        fbs_input = st.radio("Fasting Blood Sugar > 120 mg/dl?", ["No", "Yes"])
        fbs = 1 if fbs_input == "Yes" else 0
        
        restecg = st.selectbox("ECG Results", [0, 1, 2], help="0: Normal, 1: ST-T Wave Abnormality, 2: LV Hypertrophy")
        thalach = st.slider("Max Heart Rate", 60, 220, 175) # Default 175 (Healthy)

    with col3:
        st.subheader("🏃 Activity")
        # Convert Yes/No to 1/0
        exang_input = st.radio("Exercise Induced Angina?", ["No", "Yes"])
        exang = 1 if exang_input == "Yes" else 0
        
        oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 0.0)
        slope = st.selectbox("ST Slope", [0, 1, 2], index=2) # Default 2 (Upsloping/Healthy)
        ca = st.selectbox("Major Vessels Colored (CA)", [0, 1, 2, 3, 4])
        thal = st.selectbox("Thalassemia", [0, 1, 2, 3], index=2) # Default 2 (Normal)

# 5. The Prediction Logic
st.markdown("---")
if st.button("RUN DIAGNOSTIC SCAN"):
    
    # A. Create a DataFrame with the EXACT columns your notebook listed
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
                              columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
    
    # B. Scale ONLY the 5 numerical columns
    cols_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    input_data[cols_to_scale] = scaler.transform(input_data[cols_to_scale])
    
    # C. Predict
    with st.spinner('Analyzing vitals...'):
        prediction_prob = model.predict(input_data)[0][0]
    
    # D. Display Results
    st.divider()
    col_res1, col_res2 = st.columns([2, 1])
    
    with col_res1:
        if prediction_prob > 0.5:
            st.error(f"## ⚠️ High Risk Detected ({prediction_prob:.1%})")
            st.write("The AI has detected patterns associated with heart disease. Please consult a cardiologist.")
        else:
            st.success(f"## ✅ Low Risk / Clear Skies ({prediction_prob:.1%})")
            st.balloons()
            st.write("Your heart vitals look strong! Keep up the healthy lifestyle.")

    with col_res2:
        st.write("### Risk Gauge")
        st.progress(float(prediction_prob))