import streamlit as st
import pickle
import pandas as pd

# Load trained model
with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Hospital Appointment No-Show Prediction")

st.write("Enter patient details below to predict whether they will miss the appointment.")

# ---- INPUT FIELDS ----

gender = st.selectbox("Gender", ["Female", "Male"])
age = st.number_input("Age", min_value=0, max_value=100, value=30)

hypertension = st.selectbox("Hypertension", [0, 1])
diabetes = st.selectbox("Diabetes", [0, 1])
alcoholism = st.selectbox("Alcoholism", [0, 1])
handicap = st.selectbox("Handicap", [0, 1])

sms_received = st.selectbox("SMS Received", [0, 1])

waiting_days = st.number_input("Waiting Days", min_value=0, max_value=365, value=5)

# ---- PREDICTION ----

if st.button("Predict"):

    # Convert categorical inputs
    gender = 1 if gender == "Male" else 0

    # Create dataframe EXACTLY same order as training
    input_data = pd.DataFrame([{
        "Gender": gender,
        "Age": age,
        "Hypertension": hypertension,
        "Diabetes": diabetes,
        "Alcoholism": alcoholism,
        "Handicap": handicap,
        "SMS_received": sms_received,
        "waiting_days": waiting_days
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠ Patient likely to MISS appointment")
    else:
        st.success(f"✅ Patient likely to ATTEND appointment")

    st.write(f"Probability of No-Show: {round(probability*100, 2)}%")