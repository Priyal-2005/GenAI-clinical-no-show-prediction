import streamlit as st
import pandas as pd
import pickle
import os
from datetime import datetime

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Clinical No-Show Predictor", layout="centered")
st.title("🏥 Clinical Appointment No-Show Prediction")
st.markdown("Predict the likelihood of a patient missing their appointment.")

# -----------------------------
# Load Model & Scaler
# -----------------------------
@st.cache_resource
def load_model():
    try:
        if os.path.exists("models/best_model.pkl"):
            model = pickle.load(open("models/best_model.pkl", "rb"))
            scaler = pickle.load(open("models/scaler.pkl", "rb"))
        else:
            model = pickle.load(open("best_model.pkl", "rb"))
            scaler = pickle.load(open("scaler.pkl", "rb"))
        return model, scaler
    except:
        st.error("Model files not found.")
        st.stop()

model, scaler = load_model()

st.markdown("---")

# -----------------------------
# User Inputs
# -----------------------------
st.subheader("📋 Patient Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 0, 120, 35)
    gender = st.selectbox("Gender", ["Female", "Male"])
    scholarship = st.selectbox("Scholarship Program", ["No", "Yes"])
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    diabetes = st.selectbox("Diabetes", ["No", "Yes"])

with col2:
    alcoholism = st.selectbox("Alcoholism", ["No", "Yes"])
    handcap = st.selectbox("Disability Level", [0,1,2,3,4])
    sms_received = st.selectbox("SMS Reminder Sent", ["Yes", "No"])

st.markdown("---")
st.subheader("📅 Appointment Details")

scheduled_date = st.date_input("Scheduled Date")
appointment_date = st.date_input("Appointment Date")

if appointment_date < scheduled_date:
    st.error("Appointment date cannot be before scheduled date.")
    st.stop()

waiting_days = (appointment_date - scheduled_date).days
day_of_week = appointment_date.weekday()

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):

    # Encode
    input_data = pd.DataFrame({
        "Gender": [1 if gender=="Male" else 0],
        "Age": [age],
        "Scholarship": [1 if scholarship=="Yes" else 0],
        "Hipertension": [1 if hypertension=="Yes" else 0],
        "Diabetes": [1 if diabetes=="Yes" else 0],
        "Alcoholism": [1 if alcoholism=="Yes" else 0],
        "Handcap": [handcap],
        "SMS_received": [1 if sms_received=="Yes" else 0],
        "waiting_days": [waiting_days],
        "appointment_day_of_week": [day_of_week]
    })

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.markdown("---")
    st.subheader("📊 Results")

    if prediction == 1:
        st.error("⚠️ Likely to No-Show")
    else:
        st.success("✅ Likely to Attend")

    st.metric("No-Show Probability", f"{probability*100:.2f}%")

    st.markdown("---")

    # Simple Recommendation
    if probability > 0.6:
        st.warning("Recommended Action: Send reminder call + SMS.")
    else:
        st.info("Standard reminder is sufficient.")

st.markdown("---")
st.caption("Decision Tree Model | Streamlit Deployment | NST Sonipat")