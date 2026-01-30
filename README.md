# ❤️ LifePulse: Heart Health Monitor

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Accuracy](https://img.shields.io/badge/Model%20Accuracy-99.35%25-brightgreen)

**LifePulse** is a deep learning application designed to predict the risk of heart disease based on clinical patient data. Powered by a custom-trained Artificial Neural Network (ANN), the model achieves **99.35% accuracy** on the test set, assisting medical professionals in early diagnosis.

---

## 🚀 Key Features

* **High-Accuracy AI:** Utilizes a multi-layer Neural Network trained on clinical heart disease data.
* **Real-Time Predictions:** Instant inference using a deployed Streamlit dashboard.
* **Smart Scaling:** Automatically standardizes user inputs (Age, BP, Cholesterol) to match the model's training distribution.
* **User-Friendly Interface:** Clear, visual feedback ("Clear Skies" vs. "High Risk") with probability scores.

---

## 🧠 Model Architecture

The core "brain" of the application is a **Sequential Neural Network** built with TensorFlow/Keras:

* **Input Layer:** 13 Clinical Features (Age, Sex, Chest Pain, etc.)
* **Hidden Layer 1:** 32 Neurons (ReLU Activation)
* **Hidden Layer 2:** 16 Neurons (ReLU Activation)
* **Hidden Layer 3:** 8 Neurons (ReLU Activation)
* **Output Layer:** 1 Neuron (Sigmoid Activation for Binary Classification)

**Performance:**
* **Training Accuracy:** ~99.4%
* **Test Set Accuracy:** 98-99%

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit (Python)
* **Backend:** TensorFlow, Keras
* **Data Processing:** Pandas, NumPy, Scikit-Learn
* **Model Persistence:** Joblib (for Scaler), Keras (for Model)

---

## 💻 Installation & Usage

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/YourUsername/Pulse-AI.git](https://github.com/YourUsername/AI-Health.git)
    cd Pulse-AI
    ```

2.  **Create a Virtual Environment**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .\.venv\Scripts\Activate.ps1
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Application**
    ```bash
    streamlit run app.py
    ```

---

## 📂 Project Structure

```text
LifePulse/
├── app.py                   # Main Streamlit Dashboard application
├── data/                    # Raw and Processed CSV data
├── models/                  # Saved AI Models
│   ├── heart_disease_nn_model.keras  # The Trained Neural Network
│   ├── baseline_heart_model.pkl      # Logistic Regression Baseline
│   └── scaler.pkl                    # StandardScaler for data normalization
├── notebooks/               # Jupyter Notebooks for training
│   ├── data_preparation.ipynb
│   └── model_integration.ipynb
└── requirements.txt         # Project dependencies
