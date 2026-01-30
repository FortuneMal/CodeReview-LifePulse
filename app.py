import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import os

# 1. Page Config & Custom Styling
st.set_page_config(page_title="LifePulse", page_icon="💓", layout="wide")

st.markdown("""
    <style>
    /* Main Background & Font */
    .main { background-color: #0e1117; }
    h1 { color: #ff4b4b; font-family: 'Helvetica Neue', sans-serif; }
    
    /* Pulse Animation */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
    .heart-icon {
        display: inline-block;
        animation: pulse 2s infinite;
        margin-right: 10px;
    }
    
    /* Button Styling */
    div.stButton > button:first-child {
        background-color: #ff4b4b; color: white; border-radius: 8px; height: 3em; width: 100%; font-weight: 600; border: none; transition: 0.3s;
    }
    div.stButton > button:hover { background-color: #ff1c1c; transform: translateY(-2px); }
    
    /* Inputs */
    .stNumberInput, .stSelectbox, .stSlider { border-radius: 10px; }
    
    /* Mode Toggle Box */
    .mode-box {
        padding: 15px; border-radius: 10px; margin-bottom: 20px;
        background-color: #1e2530; border: 1px solid #30363d;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Load Assets
@st.cache_resource
def load_assets():
    model_path = os.path.join('models', 'heart_disease_nn_model.keras')
    scaler_path = os.path.join('models', 'scaler.pkl')
    
    # Check for files
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
        
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_assets()

if model is None:
    st.error("🚨 Critical Error: Models not found! Please check your /models folder.")
    st.stop()

# 3. Sidebar: The "Dual Mode" Logic
with st.sidebar:
    st.title("🎛️ Control Panel")
    
    # --- THE MAGIC TOGGLE ---
    st.markdown("### Select User Mode")
    user_mode = st.radio("", ["Patient", "Doctor"], index=0, label_visibility="collapsed")
    
    st.markdown("---")
    
    if user_mode == "Doctor":
        st.info("👨‍⚕️ **Doctor Mode Active**\n\nFull clinical controls enabled. Manual entry for ST depression, fluoroscopy, and thalassemia.")
    else:
        st.success("👤 **Patient Mode Active**\n\nSimplified interface. Complex clinical markers are auto-filled with 'Healthy' defaults.")
    
    st.markdown("---")
    st.warning("⚠️ **Disclaimer:** For educational use only.")

# 4. Main Title
st.markdown("<h1><span class='heart-icon'>💓</span> LifePulse</h1>", unsafe_allow_html=True)
st.markdown("### *AI that listens to your heart.*")

# 5. The Dynamic Form
with st.form("medical_form"):
    
    # --- SHARED SECTION (Everyone sees this) ---
    st.subheader("📋 Personal Health Details")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", 1, 120, 25)
        sex = st.selectbox("Biological Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    with c2:
        trestbps = st.slider("Resting Blood Pressure", 80, 200, 120, help="Standard BP reading.")
        chol = st.number_input("Cholesterol", 100, 600, 180, help="Total cholesterol level.")
    with c3:
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], 
                          format_func=lambda x: ["Typical Angina (Severe)", "Atypical Angina", "Non-anginal Pain", "No Pain (Asymptomatic)"][x],
                          index=3)

    # --- MODE DEPENDENT LOGIC ---
    if user_mode == "Doctor":
        # DOCTOR MODE: Show EVERYTHING (The Complex 8)
        st.markdown("---")
        st.subheader("👨‍⚕️ Clinical Data")
        
        with st.expander("Expand Clinical Markers", expanded=True):
            d1, d2, d3 = st.columns(3)
            with d1:
                fbs_input = st.radio("Fasting BS > 120?", ["No", "Yes"])
                restecg = st.selectbox("Resting ECG", [0, 1, 2])
            with d2:
                thalach = st.slider("Max Heart Rate", 60, 220, 150)
                exang_input = st.radio("Ex. Angina?", ["No", "Yes"])
            with d3:
                oldpeak = st.number_input("Oldpeak (ST Depr)", 0.0, 6.2, 0.0)
                slope = st.selectbox("ST Slope", [0, 1, 2], index=2)
                ca = st.selectbox("Major Vessels (CA)", [0, 1, 2, 3, 4])
                thal = st.selectbox("Thalassemia", [0, 1, 2, 3], index=2)
    
    else:
        # PATIENT MODE: Ask simple questions, assume healthy defaults for complex ones
        st.markdown("---")
        st.subheader("🏃 Activity & History")
        
        p1, p2 = st.columns(2)
        with p1:
            # Smart Logic: Estimate Max Heart Rate based on activity level
            st.write("**How active are you?**")
            activity_level = st.select_slider("", options=["Sedentary", "Moderate", "Active", "Athlete"])
            
            # Backend Logic for Heart Rate (Thalach)
            if activity_level == "Sedentary": thalach = 130
            elif activity_level == "Moderate": thalach = 150
            elif activity_level == "Active": thalach = 170
            else: thalach = 185
            st.caption(f"Estimated Peak Heart Rate: {thalach} bpm")
            
        with p2:
            st.write("**Do you feel chest pain when you run/walk?**")
            exang_input = st.radio("", ["No", "Yes"])

        # --- HIDDEN DEFAULTS (The Imputation Layer) ---
        # We fill these with "Healthy" values so the Neural Network doesn't crash
        fbs_input = "No"      # Assume normal sugar
        restecg = 1           # Assume Normal ECG
        oldpeak = 0.0         # Assume No ST Depression
        slope = 2             # Assume Upsloping (Healthy)
        ca = 0                # Assume 0 Blocked Vessels
        thal = 2              # Assume Normal Thalassemia

    # Submit Button
    submitted = st.form_submit_button("🔍 RUN DIAGNOSTIC SCAN")

# 6. Prediction Logic
if submitted:
    
    # 1. Convert Inputs to Machine Readable
    fbs = 1 if fbs_input == "Yes" else 0
    exang = 1 if exang_input == "Yes" else 0
    
    # 2. Create DataFrame with EXACT column order
    cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]], columns=cols)
    
    # 3. Scale ONLY the 5 numerical columns
    cols_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    input_data[cols_to_scale] = scaler.transform(input_data[cols_to_scale])
    
    # 4. Predict
    with st.spinner('Listening to your heart...'):
        prediction_prob = model.predict(input_data)[0][0]
    
    # ---------------------------------------------------------
    # 🛡️ CLINICAL SAFETY NET (Hybrid Rule-Based System)
    # If the AI misses a dangerous value (like High BP), we force a warning.
    # ---------------------------------------------------------
    risk_factors = []
    
    if trestbps >= 160:
        risk_factors.append(f"CRITICAL: Resting Blood Pressure is dangerously high ({trestbps}).")
    if chol >= 280:
        risk_factors.append(f"CRITICAL: Cholesterol is very high ({chol}).")
    if oldpeak >= 2.5:
        risk_factors.append(f"WARNING: ST Depression indicates heart stress ({oldpeak}).")
    if fbs == 1:
        risk_factors.append("WARNING: High Fasting Blood Sugar (Diabetes Risk).")

    # 5. Display Results
    st.divider()
    col_res1, col_res2 = st.columns([3, 1])
    
    with col_res1:
        
        # LOGIC: High Risk IF AI says so (> 0.5) OR Critical Risk Factors exist
        if prediction_prob < 0.5 or len(risk_factors) > 0:
            
            # Case A: AI didn't catch it, but our Rules did
            if prediction_prob <= 0.5:
                st.warning(f"## ⚠️ Clinical Alert Triggered")
                st.write("**The AI model predicts Low Risk based on history, BUT your vitals show critical danger signs:**")
            
            # Case B: AI and Rules agree on High Risk
            else:
                st.error(f"## ⚠️ High Cardiac Risk Detected")
                
                if user_mode == "Patient":
                    st.write("### What does this mean?")
                    st.write("Your reported symptoms and vitals align with patterns found in heart disease. This is a strong signal to check in with a doctor.")
                    st.markdown("**Recommended Action:** Schedule a professional medical check-up.")
                else:
                    st.write("Clinical markers indicate high probability of heart disease presence.")
                    st.markdown("**Next Step:** Perform angiography and stress testing.")

            # List specific dangers (Safety Net Output)
            if len(risk_factors) > 0:
                st.markdown("---")
                st.markdown("#### 🚨 Specific Risk Factors:")
                for risk in risk_factors:
                    st.markdown(f"* {risk}")
                
        else:
            # Healthy (Only if AI says Safe AND No Rules Triggered)
            st.success(f"## ✅ Low Risk / Healthy Profile")
            
            if user_mode == "Patient":
                st.write("### Good News!")
                st.write("Your heart profile looks strong. Keep up the healthy lifestyle.")
                st.balloons()
            else:
                st.write("Patient vitals are within healthy parameters. No immediate intervention required.")
                st.balloons()

    # Icon Display
    with col_res2:
        if prediction_prob > 0.5 or len(risk_factors) > 0:
            st.markdown("🚨")
        else:
            st.markdown("💖")
