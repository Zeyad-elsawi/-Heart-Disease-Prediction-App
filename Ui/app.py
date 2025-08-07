import streamlit as st
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
class RFETransformer(BaseEstimator, TransformerMixin):
    def __init__(self, rfe_model):
        self.rfe_model = rfe_model
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return self.rfe_model.transform(X)


# Load the complete pipeline (must include encoder, scaler, PCA, RFE, model)
model = joblib.load('../NoteBooks/heart_disease_model.pkl')  # adjust path if needed

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")

st.title("❤️ Heart Disease Risk Prediction")
st.markdown("This AI model is trained using Logistic Regression, PCA, RFE, and one-hot encoded data. It predicts whether a patient is likely to have heart disease.")

# --- User Input Form ---
st.header("📝 Enter Your Medical Data")

age = st.number_input("Age", min_value=20, max_value=100, step=1)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, step=1)
chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, step=1)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
restecg = st.selectbox("Resting ECG", ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=70, max_value=220, step=1)
exang = st.selectbox("Exercise-induced Angina", ["Yes", "No"])
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=6.0, step=0.1)
slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

# --- Encoding Inputs ---
sex = 1 if sex == "Male" else 0
cp_dict = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
cp = cp_dict[cp]
fbs = 1 if fbs == "Yes" else 0
restecg_dict = {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}
restecg = restecg_dict[restecg]
exang = 1 if exang == "Yes" else 0
slope_dict = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
slope = slope_dict[slope]
thal_dict = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}
thal = thal_dict[thal]

columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
           'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# Construct a DataFrame, not a NumPy array
features = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs,
                          restecg, thalach, exang, oldpeak,
                          slope, ca, thal]], columns=columns)

# Now prediction will work!


# --- Prediction ---
if st.button("🔍 Predict"):
    
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1] * 100

    st.markdown("### 📊 Result")
    if prediction == 1:
        st.error(f"⚠️ Likely to have Heart Disease\n\n**Model Confidence:** {probability:.2f}%")
    else:
        st.success(f"✅ Unlikely to have Heart Disease\n\n**Model Confidence:** {100 - probability:.2f}%")
