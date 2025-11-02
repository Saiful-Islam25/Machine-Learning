# Save as app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load Pre-trained Model
model = joblib.load("hypertension_rf_model.pkl")
st.title("🩺 Hypertension Risk Prediction")
st.write("Predict your risk of hypertension based on health parameters.")

# User Inputs
age = st.number_input("Age", 1, 120, 30)
gender = st.selectbox("Gender", ["Male", "Female"])
bmi = st.number_input("BMI", 10.0, 50.0, 22.0)
systolic_bp = st.number_input("Systolic BP", 80, 200, 120)
diastolic_bp = st.number_input("Diastolic BP", 60, 130, 80)
cholesterol = st.number_input("Cholesterol", 100, 400, 200)
smoking = st.selectbox("Smoking", ["No", "Yes"])
alcohol = st.selectbox("Alcohol", ["No", "Yes"])
physical_activity = st.selectbox("Physical Activity", ["Low", "Moderate", "High"])
family_history = st.selectbox("Family History", ["No", "Yes"])
stress_level = st.selectbox("Stress Level", ["Low", "Moderate", "High"])

# Preprocessing
le_gender = LabelEncoder(); le_gender.fit(["Male","Female"]); gender_val = le_gender.transform([gender])[0]
le_binary = LabelEncoder(); le_binary.fit(["No","Yes"])
smoking_val = le_binary.transform([smoking])[0]; alcohol_val = le_binary.transform([alcohol])[0]; family_val = le_binary.transform([family_history])[0]
le_activity = LabelEncoder(); le_activity.fit(["Low","Moderate","High"]); activity_val = le_activity.transform([physical_activity])[0]
le_stress = LabelEncoder(); le_stress.fit(["Low","Moderate","High"]); stress_val = le_stress.transform([stress_level])[0]

input_data = np.array([[age, gender_val, bmi, systolic_bp, diastolic_bp, cholesterol,
                        smoking_val, alcohol_val, activity_val, family_val, stress_val]])

# Scaling (replace with training scaler values)
scaler = StandardScaler()
scaler.mean_ = np.array([40,0.5,25,120,80,200,0.2,0.3,1,0.2,1])
scaler.scale_ = np.array([15,0.5,5,15,10,40,0.4,0.45,0.8,0.4,0.8])
input_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict Hypertension Risk"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    if prediction == 1:
        st.error(f"⚠️ High Risk ({probability*100:.2f}% probability)")
    else:
        st.success(f"✅ Low Risk ({probability*100:.2f}% probability)")
