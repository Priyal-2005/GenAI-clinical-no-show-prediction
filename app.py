import streamlit as st
import pandas as pd
import pickle
import os

# ==========================
# Load Model and Scaler
# ==========================

# ADD ERROR HANDLING HERE (before loading):
if not os.path.exists("best_model.pkl"):
    st.error("Model file not found!")
    st.stop()

if not os.path.exists("scaler.pkl"):
    st.error("Scaler file not found!")
    st.stop()

model = pickle.load(open("models/best_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

# ==========================
# App Title
# ==========================

st.title("Clinical Appointment No-Show Prediction")

st.write("Enter patient details to predict no-show probability.")

# ==========================
# User Inputs
# ==========================

age = st.number_input("Age", min_value=0, max_value=100, value=30)

gender = st.selectbox("Gender", ["Female", "Male"])
hypertension = st.selectbox("Hypertension", ["No", "Yes"])
diabetes = st.selectbox("Diabetes", ["No", "Yes"])
alcoholism = st.selectbox("Alcoholism", ["No", "Yes"])
handcap = st.selectbox("Handicap", ["No", "Yes"])
sms_received = st.selectbox("SMS Received", ["No", "Yes"])

lead_time = st.number_input("Waiting Days (Lead Time)", min_value=0, max_value=365, value=5)

# ==========================
# Prediction Button
# ==========================

if st.button("Predict"):

    # Encode categorical variables
    gender = 1 if gender == "Male" else 0
    hypertension = 1 if hypertension == "Yes" else 0
    diabetes = 1 if diabetes == "Yes" else 0
    alcoholism = 1 if alcoholism == "Yes" else 0
    handcap = 1 if handcap == "Yes" else 0
    sms_received = 1 if sms_received == "Yes" else 0

    # Create input dataframe
    input_data = pd.DataFrame([[
        gender,
        age,
        hypertension,
        diabetes,
        alcoholism,
        handcap,
        sms_received,
        lead_time
    ]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # Show results
    st.subheader("Prediction Result")

    if prediction == 1:
        st.write("Prediction: No-Show")
    else:
        st.write("Prediction: Will Attend")

    st.write("Probability of No-Show:", round(probability * 100, 2), "%")